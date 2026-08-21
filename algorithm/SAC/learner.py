"""SAC learner aligned with XuanCe."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from xuance.torch.learners import Learner


class SACLearner(Learner):
    def __init__(self, config: Namespace, policy: nn.Module, callback):
        super().__init__(config, policy, callback)

        # 优化器设置
        self.optimizer = {
            'actor': torch.optim.Adam(self.policy.actor_parameters, self.config.learning_rate_actor),
            'critic': torch.optim.Adam(self.policy.critic_parameters, self.config.learning_rate_critic),
        }
        if self.use_linear_lr_decay:
            self.scheduler = {
                'actor': torch.optim.lr_scheduler.LinearLR(
                    self.optimizer['actor'],
                    start_factor=1.0,
                    end_factor=self.end_factor_lr_decay,
                    total_iters=self.total_iters,
                ),
                'critic': torch.optim.lr_scheduler.LinearLR(
                    self.optimizer['critic'],
                    start_factor=1.0,
                    end_factor=self.end_factor_lr_decay,
                    total_iters=self.total_iters,
                ),
            }
        else:
            self.scheduler = None
        self.mse_loss = nn.MSELoss()
        self.tau = config.tau
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            init_alpha = float(config.alpha)
            if not np.isfinite(init_alpha) or init_alpha <= 0.0:
                raise ValueError(f'config.alpha must be a finite positive number, got {config.alpha!r}')
            self.target_entropy = -np.prod(policy.action_space.shape).item()
            self.log_alpha = nn.Parameter(
                torch.tensor([np.log(init_alpha)], dtype=torch.float32, device=self.device, requires_grad=True)
            )
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate_actor)

    def save_model(self, model_path):
        super().save_model(model_path)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if self.use_automatic_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha.detach().cpu()
            checkpoint['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint['scheduler'] = {k: v.state_dict() for k, v in self.scheduler.items()}
        torch.save(checkpoint, model_path)

    def load_model(self, path, model=None):
        dir_name = super().load_model(path, model)
        checkpoint_path = self._resolve_checkpoint_path(path, model, dir_name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self._restore_sac_extras(checkpoint)
        return dir_name

    @staticmethod
    def _resolve_checkpoint_path(path, model, dir_name):
        path_obj = Path(path)
        if model is None and path_obj.is_file():
            return str(path_obj)
        if model is not None:
            candidate = path_obj / model
            if candidate.is_file():
                return str(candidate)
        model_names = sorted(Path(dir_name).glob('*.pth'))
        for model_path in model_names:
            if model_path.name == 'final_train_model.pth':
                return str(model_path)
        if len(model_names) == 1:
            return str(model_names[0])
        raise FileNotFoundError(f'cannot resolve checkpoint path in {dir_name}')

    def _restore_sac_extras(self, checkpoint):
        if self.use_automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha.data.copy_(checkpoint['log_alpha'].to(self.device))
            self.alpha = self.log_alpha.exp()
            if 'alpha_optimizer' in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        if self.scheduler is not None and 'scheduler' in checkpoint:
            for key, scheduler in self.scheduler.items():
                scheduler.load_state_dict(checkpoint['scheduler'][key])

    def update(self, **samples):
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        next_batch = torch.as_tensor(samples['obs_next'], device=self.device)
        rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
        ter_batch = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)
        info = self.callback.on_update_start(
            self.iterations,
            policy=self.policy,
            obs=obs_batch,
            act=act_batch,
            next_obs=next_batch,
            rew=rew_batch,
            termination=ter_batch,
        )

        # actor update
        # $log \pi(a | s)$ 和 $Q_{1,2}(s, a)$ 的计算
        log_pi, policy_q_1, policy_q_2 = self.policy.Qpolicy(obs_batch)
        policy_q = torch.min(policy_q_1, policy_q_2).reshape([-1])  # 减轻 Q 高估

        # 策略损失：$$J_{\pi} = \mathbb{E}_{s \sim D, a \sim \pi} [\alpha \log \pi(a | s) - Q(s, a)]$$
        p_loss = (self.alpha * log_pi.reshape([-1]) - policy_q).mean()
        self.optimizer['actor'].zero_grad()
        p_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters, self.grad_clip_norm)
        self.optimizer['actor'].step()

        # critic update
        # 用 replay 里的(s,a) 算的 Q 值
        action_q_1, action_q_2 = self.policy.Qaction(obs_batch, act_batch)

        # 下一个状态的 soft 目标
        log_pi_next, target_q = self.policy.Qtarget(next_batch)
        target_value = target_q - self.alpha * log_pi_next.reshape([-1])

        # 贝尔曼方程
        backup = rew_batch + (1 - ter_batch) * self.gamma * target_value

        # 双 Q 的 MSE 损失
        q_loss = self.mse_loss(action_q_1, backup.detach()) + self.mse_loss(action_q_2, backup.detach())

        # 反传与更新
        self.optimizer['critic'].zero_grad()
        q_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters, self.grad_clip_norm)
        self.optimizer['critic'].step()

        # automatic entropy tuning
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.zeros([])

        # 学习率更新
        if self.scheduler is not None:
            self.scheduler['actor'].step()
            self.scheduler['critic'].step()

        #  Polyak 更新 target critic
        self.policy.soft_update(self.tau)

        actor_lr = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        # 多卡训练
        if self.distributed_training:
            info.update(
                {
                    f'Qloss/rank_{self.rank}': q_loss.item(),
                    f'Ploss/rank_{self.rank}': p_loss.item(),
                    f'Qvalue/rank_{self.rank}': policy_q.mean().item(),
                    f'actor_lr/rank_{self.rank}': actor_lr,
                    f'critic_lr/rank_{self.rank}': critic_lr,
                }
            )
        else:
            info.update(
                {
                    'Qloss': q_loss.item(),
                    'Ploss': p_loss.item(),
                    'Qvalue': policy_q.mean().item(),
                    'actor_lr': actor_lr,
                    'critic_lr': critic_lr,
                }
            )
        if self.use_automatic_entropy_tuning:
            if self.distributed_training:
                info.update(
                    {
                        f'alpha_loss/rank_{self.rank}': alpha_loss.item(),
                        f'alpha/rank_{self.rank}': self.alpha.item(),
                    }
                )
            else:
                info.update({'alpha_loss': alpha_loss.item(), 'alpha': self.alpha.item()})

        info.update(
            self.callback.on_update_end(
                self.iterations,
                policy=self.policy,
                info=info,
                log_pi=log_pi,
                policy_q_1=policy_q_1,
                policy_q_2=policy_q_2,
                policy_q=policy_q,
                p_loss=p_loss,
                action_q_1=action_q_1,
                action_q_2=action_q_2,
                log_pi_next=log_pi_next,
                target_q=target_q,
                target_value=target_value,
                backup=backup,
                q_loss=q_loss,
                alpha_loss=alpha_loss,
                alpha=self.alpha,
            )
        )
        return info
