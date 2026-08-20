"""Rigid overhead contact line / pantograph simulation core."""

import operator

import numpy as np
from scipy.linalg import cho_factor, cho_solve

PANTOGRAPH = 1  # 1 = DSA380; 2 = DSA250; 3 = TSG18F
RIGID_OVERHEAD_CONTACT_SYSTEM = 3
SPEED_KMH = 200  # [km/h]
NM = 200
DT_BASE = 1e-4  # [s]
G = 9.8  # [m/s²]

# KS 为受电弓–接触网之间的接触弹簧刚度
KS = 82300  # 接触刚度 [N/m]
ALPHA_C = 0.0125
BETA_C = 0.0001

N_SKIP_SPANS = 8  # 统计稳定段时剔除首尾各 N_SKIP_SPANS 跨
# 预应力参考位移采样窗口独立于 episode 的 skip_spans；两者默认采用同一稳定段约定。
PRELOAD_CALIBRATION_SKIP_SPANS = N_SKIP_SPANS

MAX_NEWTON_ITER = 10  # 每步接触 Newton / 有效集迭代最大次数

WEAR_AMPLITUDE = 1.0e-3  # [0.2e-3, 3e-3] m
WEAR_WAVELENGTH = 0.6  # [0.4, 1.2] m


BUSBAR_JOINT_OFFSET = 0.0  # [m]


def _require_int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be an integer >= {minimum}, got {value!r}')
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be an integer >= {minimum}, got {value!r}') from exc
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {value}')
    return int(value)


def contact_wire_wear(
    x,
    A_w: float = WEAR_AMPLITUDE,
    lambda_w: float = WEAR_WAVELENGTH,
):
    """接触线磨耗深度 W_cw = A_w/2 · (1 - cos(2πl/λ_w))。"""
    if not np.isfinite(A_w) or A_w < 0:
        raise ValueError(f'A_w must be finite and non-negative, got {A_w}')
    if not np.isfinite(lambda_w) or lambda_w <= 0:
        raise ValueError(f'lambda_w must be finite and positive, got {lambda_w}')
    return 0.5 * A_w * (1.0 - np.cos(2.0 * np.pi * x / lambda_w))


def contact_penetration(y_pantograph: float, y_contact_wire: float, wear_depth: float = 0.0) -> float:
    """穿透量 = 弓头位移 - 接触线位移 - 磨耗深度，向上为正。"""
    return y_pantograph - y_contact_wire - wear_depth


def rigid_overhead_contact_system_params(rigid_overhead_contact_system: int, N_spans: int | None = None):
    """返回指定刚性接触网预设的结构参数。

    返回 (L, N, rhoA, EI, KEQ, MEQ, MZ, L_MZ).
    """
    table = {
        1: (8.0, 30, 8.1, 1.7e5, 6.7e7, 7.0, 2.84, 12.0),  # case 1
        2: (8.5, 30, 7.1, 2.7e5, 6e5, 7.0, 2.84, 12.0),  # case 2
        3: (8.0, 30, 7.25, 2.69e5, 6e4, 7.0, 2.84, 12.0),  # case 3
    }
    if rigid_overhead_contact_system not in table:
        raise ValueError(
            f'RIGID_OVERHEAD_CONTACT_SYSTEM must be 1-3, the current value is {rigid_overhead_contact_system}.'
        )
    L, N_default, rhoA, EI, KEQ, MEQ, MZ, L_MZ = table[rigid_overhead_contact_system]
    N = _require_int('N_spans', N_spans, 1) if N_spans is not None else N_default
    return L, N, rhoA, EI, KEQ, MEQ, MZ, L_MZ


def pantograph_params(ptype: int, scale: dict[str, float] | None = None):
    """返回所选受电弓的 (m1,m2,m3, k1,k2,k3, c1,c2,c3, F0)。"""
    table = {
        1: (7.12, 6.00, 5.80, 9430.0, 14100.0, 0.1, 0, 0, 70.0, 120.0),  # DSA380
        2: (7.51, 5.855, 4.645, 8380.0, 6200.0, 80.0, 0, 0, 70, 120.0),  # DSA250
        3: (10.31, 6.15, 15.30, 53415.8, 6754.2, 40687.0, 29.60, 0.20, 1.14, 110.0),  # TSG18F
    }
    if ptype not in table:
        raise ValueError(f'PANTOGRAPH must be 1-3, the current value is {ptype}')
    values = list(table[ptype])
    names = ('m1', 'm2', 'm3', 'k1', 'k2', 'k3', 'c1', 'c2', 'c3')
    if scale:
        unknown = set(scale) - set(names)
        if unknown:
            raise ValueError(f'unknown pantograph parameter(s): {", ".join(sorted(unknown))}')
        for name, factor in scale.items():
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError(f'pantograph scale for {name} must be finite and positive, got {factor}')
            values[names.index(name)] *= float(factor)
    return tuple(values)


def compute_busbar_positions(LS: float, l_mz: float, offset: float = BUSBAR_JOINT_OFFSET) -> np.ndarray:
    """返回等间距布置下的汇流排接头位置。"""
    if l_mz <= 0:
        raise ValueError(f'l_mz must be positive, got {l_mz}')
    if not 0 <= offset < LS:
        raise ValueError(f'offset must satisfy 0 <= offset < LS, got offset={offset}, LS={LS}')
    return np.arange(offset, LS, l_mz, dtype=float)


class ProcsCore:
    """可步进的刚性接触网 / 受电弓动力学仿真核。

    使用 Newmark-β + 罚函数接触 + Woodbury/Sherman-Morrison 快速求解。
    外部控制器通过 `advance(n_inner, f3)` 施加作用于下框架 m3 的力 f3。
    结构矩阵与初始状态与车速 / 磨耗无关，逐回合仿真用 `reset()` 复用。
    `skip_spans > 0` 时，由调用方先滑行至稳定段起点，再推进至稳定段终点。
    """

    def __init__(
        self,
        rigid_overhead_contact_system: int = RIGID_OVERHEAD_CONTACT_SYSTEM,  # 刚性接触网型号
        pantograph: int = PANTOGRAPH,  # 受电弓型号
        speed_kmh: float = SPEED_KMH,  # 车速 [km/h]
        NM: int = NM,  # 接触网模态数
        N_spans: int = 30,  # 跨数
        dt_base: float = DT_BASE,  # 时间步长 [s]
        irregularity: bool = False,  # 是否启用接触线磨耗不平顺
        wear_amplitude: float = 0.0,  # 接触线磨耗幅值 [m]
        wear_wavelength: float = WEAR_WAVELENGTH,  # 接触线磨耗波长 [m]
        pantograph_param_scale: dict[str, float] | None = None,  # 缩放受电弓参数
        busbar_joint_offset: float = BUSBAR_JOINT_OFFSET,  # 汇流排接头偏移
        max_newton_iter: int = MAX_NEWTON_ITER,  # 每步接触 Newton / 有效集迭代最大次数
        skip_spans: int = N_SKIP_SPANS,  # 统计稳定段时剔除首尾各 N_SKIP_SPANS 跨
    ):
        NM = _require_int('NM', NM, 1)
        N_spans = _require_int('N_spans', N_spans, 1)
        if NM < N_spans - 1:
            raise ValueError(f'NM must be >= N_spans - 1 ({N_spans - 1}), got {NM}')
        max_newton_iter = _require_int('max_newton_iter', max_newton_iter, 2)
        skip_spans = _require_int('skip_spans', skip_spans, 0)
        if not np.isfinite(dt_base) or dt_base <= 0:
            raise ValueError(f'dt_base must be finite and positive, got {dt_base}')
        if not np.isfinite(speed_kmh) or speed_kmh <= 0:
            raise ValueError(f'speed_kmh must be finite and positive, got {speed_kmh}')
        if not np.isfinite(wear_amplitude) or wear_amplitude < 0:
            raise ValueError(f'wear_amplitude must be finite and non-negative, got {wear_amplitude}')
        if not np.isfinite(wear_wavelength) or wear_wavelength <= 0:
            raise ValueError(f'wear_wavelength must be finite and positive, got {wear_wavelength}')

        self.max_newton_iter = max_newton_iter
        self.skip_spans = skip_spans
        self.wear_amplitude = float(wear_amplitude)
        self.wear_wavelength = float(wear_wavelength)
        self.irregularity = bool(irregularity)
        self.dt = float(dt_base)

        self.L, self.N, self.rhoA, self.EI, self.KEQ, self.MEQ, self.MZ, self.L_MZ = (
            rigid_overhead_contact_system_params(rigid_overhead_contact_system, N_spans)
        )
        self.m1, self.m2, self.m3, self.k1, self.k2, self.k3, self.c1, self.c2, self.c3, self.F0 = pantograph_params(
            pantograph, pantograph_param_scale
        )

        self.LS = self.L * self.N
        if 2 * self.skip_spans * self.L >= self.LS:
            raise ValueError(f'skip_spans={self.skip_spans} leaves no active window (LS={self.LS} m, L={self.L} m)')

        # 受电弓质量 / 阻尼 / 刚度 / 外力
        self.M_p = np.diag([self.m1, self.m2, self.m3])
        self.C_p = np.array(
            [
                [self.c1, -self.c1, 0],
                [-self.c1, self.c1 + self.c2, -self.c2],
                [0, -self.c2, self.c2 + self.c3],
            ],
            dtype=float,
        )
        self.K_p = np.array(
            [
                [self.k1, -self.k1, 0],
                [-self.k1, self.k1 + self.k2, -self.k2],
                [0, -self.k2, self.k2 + self.k3],
            ],
            dtype=float,
        )
        self.F_p = np.array([0.0, 0.0, self.F0])

        # 接触网模态
        modes = np.arange(1, NM + 1, dtype=float)
        self.modes = modes
        self.norm_factor = np.sqrt(2.0 / (self.rhoA * self.LS))
        # 简支梁固有频率公式：$$\omega_n = \left(\frac{n\pi}{L_S}\right)^2\sqrt{\frac{EI}{\rho A}}$$
        self.omega_n = (modes * np.pi / self.LS) ** 2 * np.sqrt(self.EI / self.rhoA)

        x_j = self.L * np.arange(1, self.N, dtype=float)
        Phi_sup = self.norm_factor * np.sin(np.outer(modes * np.pi / self.LS, x_j))
        M_add_sup = self.MEQ * Phi_sup @ Phi_sup.T
        K_add_sup = self.KEQ * Phi_sup @ Phi_sup.T

        x_mz = compute_busbar_positions(self.LS, l_mz=self.L_MZ, offset=busbar_joint_offset)
        Phi_mz = self.norm_factor * np.sin(np.outer(modes * np.pi / self.LS, x_mz))
        M_add_mz = self.MZ * Phi_mz @ Phi_mz.T

        self.M_cat = np.eye(NM) + M_add_sup + M_add_mz
        self.K_cat = np.diag(self.omega_n**2) + K_add_sup
        self.C_cat = ALPHA_C * self.M_cat + BETA_C * self.K_cat

        int_sin = (self.LS / (modes * np.pi)) * (1.0 - np.cos(modes * np.pi))
        F_grav_beam = -self.rhoA * G * self.norm_factor * int_sin
        F_grav_sup = -self.MEQ * G * Phi_sup.sum(axis=1)
        F_grav_mz = -self.MZ * G * Phi_mz.sum(axis=1)
        F_gravity = F_grav_beam + F_grav_sup + F_grav_mz

        # 找形等效：悬挂点静态位移归零
        n_sup = Phi_sup.shape[1]
        KKT = np.block(
            [
                [np.diag(self.omega_n**2), Phi_sup],
                [Phi_sup.T, np.zeros((n_sup, n_sup))],
            ]
        )
        sol = np.linalg.solve(KKT, np.concatenate([F_gravity, np.zeros(n_sup)]))
        self.q_static = sol[:NM]
        F_pre = np.diag(self.omega_n**2) @ self.q_static - F_gravity
        F_gravity = F_gravity + F_pre

        n_dof = 3 + NM
        self.n_dof = n_dof
        self.M_sys = np.zeros((n_dof, n_dof))
        self.C_sys = np.zeros((n_dof, n_dof))
        self.K_sys = np.zeros((n_dof, n_dof))
        self.M_sys[:3, :3] = self.M_p
        self.M_sys[3:, 3:] = self.M_cat
        self.C_sys[:3, :3] = self.C_p
        self.C_sys[3:, 3:] = self.C_cat
        self.K_sys[:3, :3] = self.K_p
        self.K_sys[3:, 3:] = self.K_cat

        # 受电弓装配预应力：根据全线平均局部柔度和固定稳定段平均静态位移设置名义准静态参考位置。
        # F0 为下框架净抬升力；y3_reference 为接地弹簧 k3 的零力参考位置；
        K_cat_inv = np.linalg.solve(self.K_cat, np.eye(NM))
        c_cat_mean = np.trace(K_cat_inv) / (self.rhoA * self.LS)
        S_p = 1.0 / KS + 1.0 / self.k1 + 1.0 / self.k2 + c_cat_mean
        # u_mean：固定稳定段（不足则全线）4096 点平均静态位移，与车速无关，可随结构复用
        if self.LS > 2 * PRELOAD_CALIBRATION_SKIP_SPANS * self.L:
            x_grid = np.linspace(
                PRELOAD_CALIBRATION_SKIP_SPANS * self.L,
                self.LS - PRELOAD_CALIBRATION_SKIP_SPANS * self.L,
                4096,
            )
        else:
            x_grid = np.linspace(0.0, self.LS, 4096)
        phi_grid = self.norm_factor * np.sin(np.outer(modes * np.pi / self.LS, x_grid))
        u_mean = float(np.mean(phi_grid.T @ self.q_static))
        y3_reference = self.F0 * S_p + u_mean
        self.F_p_eff = self.F_p.copy()
        self.F_p_eff[2] += self.k3 * y3_reference

        self.F_base = np.zeros(n_dof)
        self.F_base[:3] = self.F_p_eff
        self.F_base[3:] = F_gravity

        # Newmark 常数 (平均加速度: beta=0.25, gamma=0.5)
        self.beta_nm, gamma_nm = 0.25, 0.5
        self.a0 = 1.0 / (self.beta_nm * self.dt * self.dt)
        self.a1 = gamma_nm / (self.beta_nm * self.dt)
        self.a2 = 1.0 / (self.beta_nm * self.dt)
        self.a3 = 1.0 / (2.0 * self.beta_nm) - 1.0
        self.a4 = gamma_nm / self.beta_nm - 1.0
        self.a5 = self.dt * (gamma_nm / (2.0 * self.beta_nm) - 1.0)
        self.a6 = self.dt * (1.0 - gamma_nm)
        self.a7 = gamma_nm * self.dt

        self.P = self.K_sys + self.a0 * self.M_sys + self.a1 * self.C_sys
        self.cP = cho_factor(self.P, check_finite=True)

        self.w_buf = np.zeros(n_dof)
        self.w_buf[0] = 1.0

        # 初始状态（与车速 / 磨耗无关，构造一次后由 reset 复用）
        Y = np.zeros(n_dof)
        Y[3:] = self.q_static

        phi_0 = self.phi_at(0.0)
        u_c_0 = phi_0 @ self.q_static
        K_p_static = self.K_p.copy()
        K_p_static[0, 0] += KS
        F_p_static = self.F_p_eff + np.array([KS * u_c_0, 0.0, 0.0])
        Y[:3] = np.linalg.solve(K_p_static, F_p_static)

        Kc0 = np.zeros((n_dof, n_dof))
        Kc0[0, 0] = KS
        Kc0[0, 3:] = -KS * phi_0
        Kc0[3:, 0] = -KS * phi_0
        Kc0[3:, 3:] = KS * np.outer(phi_0, phi_0)
        A = np.linalg.solve(self.M_sys, self.F_base - (self.K_sys + Kc0) @ Y)

        self._Y0 = Y
        self._V0 = np.zeros(n_dof)
        self._A0 = A

        self.reset(speed_kmh=speed_kmh)

    def reset(
        self,
        speed_kmh: float | None = None,
        wear_amplitude: float | None = None,
    ):
        """重置仿真状态；可选更换车速 / 磨耗幅值（结构矩阵与初始状态复用）。"""
        if wear_amplitude is not None:
            if not np.isfinite(wear_amplitude) or wear_amplitude < 0:
                raise ValueError(f'wear_amplitude must be finite and non-negative, got {wear_amplitude}')
            self.wear_amplitude = float(wear_amplitude)

        if speed_kmh is not None:
            if not np.isfinite(speed_kmh) or speed_kmh <= 0:
                raise ValueError(f'speed_kmh must be finite and positive, got {speed_kmh}')
            speed_kmh = float(speed_kmh)
            v = speed_kmh / 3.6
            n_steps = int(np.floor(self.LS / v / self.dt)) + 1
            t_vec = np.arange(n_steps, dtype=float) * self.dt
            x_vec = v * t_vec
            k_start = int(np.searchsorted(x_vec, self.skip_spans * self.L, side='left'))
            k_end = int(np.searchsorted(x_vec, self.LS - self.skip_spans * self.L, side='right'))
            if max(1, k_start) >= k_end:
                raise ValueError('speed_kmh and dt_base leave no simulation step in the active window')
            self.speed_kmh = speed_kmh
            self.v = v
            self.n_steps = n_steps
            self.t_vec = t_vec
            self.x_vec = x_vec
            self.k_start = k_start
            self.k_end = k_end

        self.Y = self._Y0.copy()
        self.V = self._V0.copy()
        self.A = self._A0.copy()
        # 推进指针：0 已初始化，下一步从 1 开始
        self.k = 1
        # x=0 处磨耗深度为 0，与是否启用磨耗无关
        phi_0 = self.phi_at(0.0)
        u_c_0 = float(phi_0 @ self.Y[3:])
        rel = contact_penetration(self.Y[0], u_c_0, 0.0)
        self.fc = KS * rel if rel > 0.0 else 0.0

    def phi_at(self, x: float) -> np.ndarray:
        return self.norm_factor * np.sin(self.modes * np.pi * x / self.LS)

    def current_observation(self) -> np.ndarray:
        """返回当前步观测 [y1, v1, a1, v_train, Fc]。"""
        return np.array(
            [self.Y[0], self.V[0], self.A[0], self.v, self.fc],
            dtype=np.float64,
        )

    def advance(self, n_inner: int, a: float = 0.0) -> dict:
        """推进 n_inner 个内步，期间对下框架 m3 施加恒定控制力 a。

        返回这 n_inner 步的 contact_force / y_pantograph / y_ocs 序列
        （到达稳定段终点时长度可能短于 n_inner）。
        """
        n_inner = _require_int('n_inner', n_inner, 0)
        if n_inner == 0:
            return self._empty_history()

        try:
            a = float(a)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'f3 must be finite, got {a!r}') from exc
        if not np.isfinite(a):
            raise ValueError(f'f3 must be finite, got {a!r}')

        F_base = self.F_base.copy()
        F_base[2] += a

        contact_force = np.empty(n_inner)
        y_pantograph = np.empty(n_inner)
        y_ocs = np.empty(n_inner)

        for i in range(n_inner):
            if self.k >= self.k_end:
                # 到达终点，截断返回
                return {
                    'n_actual': i,
                    'contact_force': contact_force[:i],
                    'y_pantograph': y_pantograph[:i],
                    'y_ocs': y_ocs[:i],
                }

            xc = self.x_vec[self.k]
            phi = self.phi_at(xc)
            w_cw = contact_wire_wear(xc, self.wear_amplitude, self.wear_wavelength) if self.irregularity else 0.0

            Ft_base = (
                F_base
                + self.M_sys @ (self.a0 * self.Y + self.a2 * self.V + self.a3 * self.A)
                + self.C_sys @ (self.a1 * self.Y + self.a4 * self.V + self.a5 * self.A)
            )

            # Newton / 有效集迭代
            Y_new = self.Y + self.dt * self.V + (0.5 - self.beta_nm) * self.dt * self.dt * self.A
            in_contact = contact_penetration(Y_new[0], phi @ Y_new[3:], w_cw) > 0.0
            z = None

            for _ in range(self.max_newton_iter):
                if in_contact:
                    self.w_buf[3:] = -phi
                    if w_cw != 0.0:
                        Ft = Ft_base.copy()
                        Ft[0] += KS * w_cw
                        Ft[3:] -= KS * phi * w_cw
                    else:
                        Ft = Ft_base
                    if z is None:
                        z = cho_solve(self.cP, self.w_buf, check_finite=True)
                    u = cho_solve(self.cP, Ft, check_finite=True)
                    Y_new = u - KS * (self.w_buf @ u) / (1.0 + KS * (self.w_buf @ z)) * z
                else:
                    Y_new = cho_solve(self.cP, Ft_base, check_finite=True)

                gap = contact_penetration(Y_new[0], phi @ Y_new[3:], w_cw)
                in_contact_new = gap > 0.0
                if in_contact == in_contact_new:
                    break
                in_contact = in_contact_new
            else:
                raise FloatingPointError(
                    f'contact active-set failed to converge at inner step k={self.k} '
                    f'after {self.max_newton_iter} iterations'
                )

            u_c_new = phi @ Y_new[3:]
            A_new = self.a0 * (Y_new - self.Y) - self.a2 * self.V - self.a3 * self.A
            self.V = self.V + self.a6 * self.A + self.a7 * A_new
            self.A = A_new
            self.Y = Y_new

            rel = contact_penetration(self.Y[0], u_c_new, w_cw)
            fc = KS * rel if rel > 0 else 0.0
            self.fc = fc

            contact_force[i] = fc
            y_pantograph[i] = self.Y[0]
            y_ocs[i] = u_c_new

            self.k += 1

        return {
            'n_actual': n_inner,
            'contact_force': contact_force,
            'y_pantograph': y_pantograph,
            'y_ocs': y_ocs,
        }

    def _empty_history(self) -> dict:
        return {
            'n_actual': 0,
            'contact_force': np.empty(0),
            'y_pantograph': np.empty(0),
            'y_ocs': np.empty(0),
        }

    def done(self) -> bool:
        """仿真是否已到达（稳定段）终点。"""
        return self.k >= self.k_end
