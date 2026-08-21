"""Gymnasium DRL environment wrapping the rigid PROCS simulator."""

import math
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FrameStackObservation
from xuance.environment import RawEnvironment

from .procs_core import (
    N_SKIP_SPANS,
    WEAR_WAVELENGTH,
    ProcsCore,
    rigid_overhead_contact_system_params,
)

DIVERGENCE_PENALTY = 600.0
MAX_CONTACT_FORCE_N = 300.0
CONSTRAINT_WEIGHT = 5.0


class ProcsEnv(gym.Env):
    """受电弓主动控制 Gymnasium 环境。

    观测：弓头位移 y1、速度 v1、加速度 a1、列车速度 v_train、接触力 Fc。
    动作：施加于下框架 m3 的主动控制力 f3（范围 ±action_max N）。
    决策频率 control_freq Hz；每个决策保持约
    round(1/(control_freq*dt_base)) 个内部 Newmark 步。
    每个回合只覆盖稳定段（剔除首尾各 skip_spans 跨）：reset 时无控制
    滑行至稳定段起点，到达稳定段终点即 terminated（自然终止，非 truncated）。
    truncated 仅用于外部时间步上限等硬性截断。
    """

    metadata: ClassVar[dict[str, Any]] = {'render_modes': []}

    def __init__(
        self,
        pantograph: int = 1,
        rigid_overhead_contact_system: int = 3,
        NM: int = 200,
        N_spans: int = 30,
        dt_base: float = 1e-4,
        speed_kmh: float = 200.0,
        speed_range_kmh: tuple[float, float] | None = (200.0, 300.0),
        irregularity: bool = False,
        wear_range_m: tuple[float, float] | None = None,
        wear_wavelength: float = WEAR_WAVELENGTH,
        pantograph_param_scale: dict[str, float] | None = None,
        skip_spans: int = N_SKIP_SPANS,
        control_freq: float = 100.0,
        action_max: float = 100.0,
        action_rate_weight: float = 0.001,
        max_newton_iter: int = 10,
    ):
        """speed_range_kmh / wear_range_m 给 (lo, hi) 元组即每回合均匀随机；None 表示固定值。"""
        super().__init__()

        self.pantograph = pantograph
        self.rigid_overhead_contact_system = rigid_overhead_contact_system
        self.NM = NM
        self.N_spans = N_spans
        self.dt_base = dt_base
        self.speed_kmh = speed_kmh
        self.speed_range_kmh = speed_range_kmh
        self.irregularity = bool(irregularity)
        self.wear_range_m = wear_range_m
        self.wear_wavelength = wear_wavelength
        self.pantograph_param_scale = pantograph_param_scale
        self.skip_spans = skip_spans
        self.control_freq = control_freq
        self.action_max = float(action_max)
        self.action_rate_weight = float(action_rate_weight)
        self.max_newton_iter = max_newton_iter

        if not np.isfinite(dt_base) or dt_base <= 0:
            raise ValueError(f'dt_base must be finite and positive, got {dt_base}')
        if not np.isfinite(speed_kmh) or speed_kmh <= 0:
            raise ValueError(f'speed_kmh must be finite and positive, got {speed_kmh}')
        if not np.isfinite(control_freq) or control_freq <= 0:
            raise ValueError(f'control_freq must be finite and positive, got {control_freq}')
        if control_freq * dt_base > 1.0:
            raise ValueError('control_freq cannot exceed the internal simulation frequency')
        if not np.isfinite(self.action_max) or self.action_max <= 0:
            raise ValueError(f'action_max must be finite and positive, got {action_max}')
        if not np.isfinite(self.action_rate_weight) or self.action_rate_weight < 0:
            raise ValueError(f'action_rate_weight must be finite and non-negative, got {action_rate_weight}')
        self._validate_range('speed_range_kmh', speed_range_kmh, lower=0.0, inclusive=False)
        self._validate_range('wear_range_m', wear_range_m, lower=0.0)

        self.n_inner = max(1, round(1.0 / (control_freq * dt_base)))

        # 观测空间：y1[m], v1[m/s], a1[m/s²], v_train[m/s], Fc[N]
        obs_low = np.array([-1.0, -50.0, -500.0, -100.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 50.0, 500.0, 100.0, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)

        # 动作空间：归一化后 [-1, 1]，训练时再映射为 ±action_max N
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        # 最慢车速，用于估算需要最多的控制步
        slowest_speed = speed_range_kmh[0] if speed_range_kmh is not None else speed_kmh
        self.max_episode_steps = self._control_steps_at_speed(float(slowest_speed))

        self._core: ProcsCore | None = None
        self._episode_steps = 0
        self._episode_return = 0.0
        self._previous_action = 0.0

    @staticmethod
    def target_contact_force(speed_kmh: float) -> float:
        return 0.00097 * speed_kmh**2 + 70.0

    @staticmethod
    def _validate_range(
        name: str,
        value: tuple[float, float] | None,
        *,
        lower: float,
        inclusive: bool = True,  # 决定下界是大于等于还是严格大于
    ) -> None:
        if value is None:
            return
        try:
            lo, hi = value
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be a (lo, hi) pair or None') from exc
        try:
            invalid = (
                not np.isfinite(lo) or not np.isfinite(hi) or lo < lower or (not inclusive and lo <= lower) or lo > hi
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must contain two finite numbers') from exc
        if invalid:
            bound = f'>= {lower}' if inclusive else f'> {lower}'
            raise ValueError(f'{name} must satisfy finite lo {bound} and lo <= hi, got {value}')

    def _sample_speed(self) -> float:
        if self.speed_range_kmh is None:
            return float(self.speed_kmh)
        lo, hi = self.speed_range_kmh
        return float(self.np_random.uniform(lo, hi))

    def _sample_wear(self) -> float:
        if self.wear_range_m is None:
            return 0.0
        lo, hi = self.wear_range_m
        return float(self.np_random.uniform(lo, hi))

    def _control_steps_at_speed(self, speed_kmh: float) -> int:
        """Return the episode length at a fixed speed without running the simulator."""
        span_length, n_spans, *_ = rigid_overhead_contact_system_params(
            self.rigid_overhead_contact_system,
            self.N_spans,
        )
        velocity = speed_kmh / 3.6
        total_length = span_length * n_spans
        n_steps = int(np.floor(total_length / velocity / self.dt_base)) + 1
        positions = velocity * (np.arange(n_steps) * self.dt_base)
        k_start = int(np.searchsorted(positions, self.skip_spans * span_length, side='left'))
        k_end = int(np.searchsorted(positions, total_length - self.skip_spans * span_length, side='right'))
        return math.ceil((k_end - max(1, k_start)) / self.n_inner)

    def _observation(self) -> np.ndarray:
        return self._core.current_observation().astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        speed = self._sample_speed()
        wear_amp = self._sample_wear()

        if self._core is None:
            self._core = ProcsCore(
                rigid_overhead_contact_system=self.rigid_overhead_contact_system,
                pantograph=self.pantograph,
                speed_kmh=speed,
                NM=self.NM,
                N_spans=self.N_spans,
                dt_base=self.dt_base,
                irregularity=self.irregularity,
                wear_amplitude=wear_amp,
                wear_wavelength=self.wear_wavelength,
                pantograph_param_scale=self.pantograph_param_scale,
                max_newton_iter=self.max_newton_iter,
                skip_spans=self.skip_spans,
            )
        else:
            self._core.reset(speed_kmh=speed, wear_amplitude=wear_amp)

        # 无控制滑行至稳定段起点（启动瞬态不计入训练）
        burn = self._core.k_start - 1
        if burn > 0:
            self._core.advance(burn, 0.0)

        self._episode_steps = 0
        self._episode_return = 0.0
        self._previous_action = 0.0

        obs = self._observation()
        remaining = self._core.k_end - self._core.k
        info = {
            'F0_N': float(self._core.F0),
            'speed_kmh': float(self._core.speed_kmh),
            'target_Fc_N': self.target_contact_force(self._core.speed_kmh),
            'n_control_steps': math.ceil(remaining / self.n_inner),
            'n_inner': int(self.n_inner),
        }
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._core is None:
            raise RuntimeError('step() called before reset()')

        action = np.asarray(action, dtype=np.float32)
        if action.size != 1 or not np.isfinite(action).all():
            raise ValueError('action must contain exactly one finite value')
        normalized_action = float(np.clip(action.reshape(-1)[0], -1.0, 1.0))
        f3 = normalized_action * self.action_max

        try:
            hist = self._core.advance(self.n_inner, f3)
        except FloatingPointError:
            obs = self._observation()
            if not np.isfinite(obs).all():
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            self._episode_steps += 1
            self._episode_return += -DIVERGENCE_PENALTY
            return (
                obs,
                -DIVERGENCE_PENALTY,
                True,
                False,
                {
                    'diverged': True,
                    'reason': 'contact_nonconvergence',
                    'episode_step': self._episode_steps,
                    'episode_return': self._episode_return,
                },
            )
        fc = hist['contact_force']  # 控制步内的接触力
        obs = self._observation()

        # 发散 / NaN：仿真状态已失效，统一终止并给大惩罚（离线 fc=0 不算发散）
        if (fc.size and not np.isfinite(fc).all()) or not np.isfinite(obs).all():
            if not np.isfinite(obs).all():
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            self._episode_steps += 1
            self._episode_return += -DIVERGENCE_PENALTY
            return (
                obs,
                -DIVERGENCE_PENALTY,
                True,
                False,
                {
                    'diverged': True,
                    'episode_step': self._episode_steps,
                    'episode_return': self._episode_return,
                },
            )

        if hist['n_actual'] == 0:
            # step() 在到达终点后被继续调用
            return obs, 0.0, True, False, {}

        target_force = self.target_contact_force(self._core.speed_kmh)
        n_actual = int(hist['n_actual'])
        mean_fc = float(fc.mean())
        std_fc = float(fc.std())
        force_mse = float(np.mean(((fc - target_force) / target_force) ** 2))  # 力跟踪误差
        n_loss = int((fc <= 0.0).sum())  # 离线步数
        n_overforce = int((fc >= MAX_CONTACT_FORCE_N).sum())  # 超过300N的步数
        loss_fraction = n_loss / n_actual  # 离线比
        overforce_fraction = n_overforce / n_actual  # 超力比
        min_fc = float(fc.min())
        max_fc = float(fc.max())
        action_penalty = 0.01 * normalized_action**2  # 惩罚动作幅度大
        action_delta = normalized_action - self._previous_action
        action_rate_penalty = self.action_rate_weight * action_delta**2  # 惩罚动作变化太快

        reward = (
            -force_mse - CONSTRAINT_WEIGHT * (loss_fraction + overforce_fraction) - action_penalty - action_rate_penalty
        )

        self._previous_action = normalized_action

        at_end = self._core.done()
        terminated = at_end
        truncated = False

        self._episode_steps += 1
        self._episode_return += reward

        info = {
            'mean_Fc_N': mean_fc,
            'std_Fc_N': std_fc,
            'min_Fc_N': min_fc,
            'max_Fc_N': max_fc,
            'force_mse': force_mse,
            'loss_fraction': loss_fraction,
            'overforce_fraction': overforce_fraction,
            'n_loss': n_loss,
            'n_overforce': n_overforce,
            'f3_N': f3,
            'n_inner_actual': n_actual,
            'target_Fc_N': target_force,
            'action_delta': action_delta,
            'action_rate_penalty': action_rate_penalty,
            'episode_step': self._episode_steps,
            'episode_return': self._episode_return,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


def parse_history_length(value, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise TypeError(f'history_length must be an integer, got bool {value!r}')
    if type(value) is not int:
        raise TypeError(f'history_length must be a non-negative integer or None, got {type(value).__name__}: {value!r}')
    if value < 0:
        raise ValueError(f'history_length must be non-negative, got {value}')
    return value


def _copy_env_kwargs(env_config) -> dict:
    raw_kwargs = getattr(env_config, 'env_kwargs', None)
    if raw_kwargs is None:
        return {}
    if not isinstance(raw_kwargs, dict):
        raise TypeError(f'env_config.env_kwargs must be None or a dictionary, got {type(raw_kwargs).__name__}')
    return dict(raw_kwargs)


# XuanCe 适配器
class ProcsXuanceEnv(RawEnvironment):
    """XuanCe adapter for the Gymnasium-compatible PROCS environment."""

    def __init__(self, env_config):
        super().__init__()
        env_kwargs = _copy_env_kwargs(env_config)

        self.env_id = getattr(env_config, 'env_id', 'PROCS-v0')
        self.render_mode = getattr(env_config, 'render_mode', None)
        history_length = parse_history_length(env_kwargs.pop('history_length', None), default=0)
        base_env = ProcsEnv(**env_kwargs)
        self.max_episode_steps = base_env.max_episode_steps
        if history_length > 0:
            self.env = FrameStackObservation(base_env, stack_size=history_length, padding_type='reset')
        else:
            self.env = base_env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self._initial_seed = getattr(env_config, 'env_seed', None)

    def reset(self, **kwargs):
        if 'seed' not in kwargs and self._initial_seed is not None:
            kwargs['seed'] = self._initial_seed
            self._initial_seed = None
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self, *args, **kwargs):
        return self.env.render()

    def close(self):
        return self.env.close()
