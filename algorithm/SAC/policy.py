"""Actor-Critic SAC policy aligned with XuanCe."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from copy import deepcopy

import torch
from gymnasium.spaces import Box
from xuance.torch import DistributedDataParallel, Module, Tensor
from xuance.torch.policies.core import CriticNet, GaussianActorNet_SAC
from xuance.torch.utils import ModuleType


class SACPolicy(Module):
    """Actor-Critic for SAC with Gaussian distributions (continuous Box only)."""

    def __init__(
        self,
        action_space: Box,
        representation: Module,  # 观测编码器
        actor_hidden_size: Sequence[int],
        critic_hidden_size: Sequence[int],
        normalize: ModuleType | None = None,
        initialize: Callable[..., Tensor] | None = None,  # 权重初始化函数
        activation: ModuleType | None = None,  # 激活函数
        activation_action: ModuleType | None = None,  # 动作激活函数
        device: str | int | torch.device | None = None,
        use_distributed_training: bool = False,
    ):
        super().__init__()
        if not isinstance(action_space, Box):
            raise TypeError('SACPolicy only supports continuous Box action spaces.')

        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self.representation_info_shape = representation.output_shapes

        self.actor_representation = representation
        self.actor = GaussianActorNet_SAC(
            representation.output_shapes['state'][0],
            self.action_dim,
            actor_hidden_size,
            normalize,
            initialize,
            activation,
            activation_action,
            device,
        )

        self.critic_1_representation = deepcopy(representation)
        self.critic_1 = CriticNet(
            representation.output_shapes['state'][0] + self.action_dim,
            critic_hidden_size,
            normalize,
            initialize,
            activation,
            device,
        )
        self.critic_2_representation = deepcopy(representation)
        self.critic_2 = CriticNet(
            representation.output_shapes['state'][0] + self.action_dim,
            critic_hidden_size,
            normalize,
            initialize,
            activation,
            device,
        )
        self.target_critic_1_representation = deepcopy(self.critic_1_representation)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2_representation = deepcopy(self.critic_2_representation)
        self.target_critic_2 = deepcopy(self.critic_2)

        self.actor_parameters = list(self.actor_representation.parameters()) + list(self.actor.parameters())
        self.critic_parameters = (
            list(self.critic_1_representation.parameters())
            + list(self.critic_1.parameters())
            + list(self.critic_2_representation.parameters())
            + list(self.critic_2.parameters())
        )

        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ['RANK'])
            if self.actor_representation._get_name() != 'Basic_Identical':
                self.actor_representation = DistributedDataParallel(self.actor_representation, device_ids=[self.rank])
            if self.critic_1_representation._get_name() != 'Basic_Identical':
                self.critic_1_representation = DistributedDataParallel(
                    self.critic_1_representation, device_ids=[self.rank]
                )
            if self.critic_2_representation._get_name() != 'Basic_Identical':
                self.critic_2_representation = DistributedDataParallel(
                    self.critic_2_representation, device_ids=[self.rank]
                )
            self.actor = DistributedDataParallel(module=self.actor, device_ids=[self.rank])
            self.critic_1 = DistributedDataParallel(module=self.critic_1, device_ids=[self.rank])
            self.critic_2 = DistributedDataParallel(module=self.critic_2, device_ids=[self.rank])

    def forward(self, observation: Tensor | dict, deterministic: bool = False):
        outputs = self.actor_representation(observation)
        act_dist = self.actor(outputs['state'])
        if deterministic:
            act_sample = act_dist.activation_fn(act_dist.deterministic_sample())
        else:
            act_sample = act_dist.activated_rsample()
        return outputs, act_sample

    # 输入：当前$s$
    # Critic: online $Q_1$, $Q_2$
    # 返回：$\log_\pi$, $Q_1$, $Q_2$
    # 用途：更新 actor
    def Qpolicy(self, observation: Tensor | dict):
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)

        act_dist = self.actor(outputs_actor['state'])
        act_sample, log_action_prob = act_dist.activated_rsample_and_logprob()

        q_1 = self.critic_1(torch.concat([outputs_critic_1['state'], act_sample], dim=-1))
        q_2 = self.critic_2(torch.concat([outputs_critic_2['state'], act_sample], dim=-1))
        return log_action_prob, q_1[:, 0], q_2[:, 0]

    # 输入：下一状态$s^\prime$
    # Critic: target $Q_1^\prime$, $Q_2^\prime$
    # 返回：$\log_\pi$, $\min Q^\prime$
    # 用途：构造critic 的 TD 目标
    def Qtarget(self, observation: Tensor | dict):
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.target_critic_1_representation(observation)
        outputs_critic_2 = self.target_critic_2_representation(observation)

        new_act_dist = self.actor(outputs_actor['state'])
        new_act_sample, log_action_prob = new_act_dist.activated_rsample_and_logprob()

        target_q_1 = self.target_critic_1(torch.concat([outputs_critic_1['state'], new_act_sample], dim=-1))
        target_q_2 = self.target_critic_2(torch.concat([outputs_critic_2['state'], new_act_sample], dim=-1))
        target_q = torch.min(target_q_1, target_q_2)
        return log_action_prob, target_q[:, 0]

    def Qaction(self, observation: Tensor | dict, action: Tensor):
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)
        q_1 = self.critic_1(torch.concat([outputs_critic_1['state'], action], dim=-1))
        q_2 = self.critic_2(torch.concat([outputs_critic_2['state'], action], dim=-1))
        return q_1[:, 0], q_2[:, 0]

    def soft_update(self, tau: float = 0.005):
        # ep：online/evaluation, tp: target
        for ep, tp in zip(self.critic_1_representation.parameters(), self.target_critic_1_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2_representation.parameters(), self.target_critic_2_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
