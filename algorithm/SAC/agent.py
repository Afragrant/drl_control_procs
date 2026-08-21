"""SAC agent wired to this package's Policy/Learner."""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch
from gymnasium.spaces import Box, Space
from xuance.common import BaseCallback, Optional
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.agents import OffPolicyAgent
from xuance.torch.utils import ActivationFunctions, NormalizeFunctions

from .learner import SACLearner
from .policy import SACPolicy


class SACAgent(OffPolicyAgent):
    """Off-policy SAC agent using algorithm.SAC Policy/Learner (continuous Box only)."""

    def __init__(
        self,
        config: Namespace,
        envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
        observation_space: Optional[Space] = None,
        action_space: Optional[Space] = None,
        callback: Optional[BaseCallback] = None,
    ):
        super().__init__(config, envs, observation_space, action_space, callback)

        if not isinstance(self.action_space, Box):
            raise TypeError('algorithm.SAC only supports continuous Box action spaces.')

        self.policy = self._build_policy()
        self.memory = self._build_memory()
        self.learner = SACLearner(self.config, self.policy, self.callback)

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, 'normalize') else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        activation_action = ActivationFunctions[self.config.activation_action]
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        return SACPolicy(
            action_space=self.action_space,
            representation=representation,
            actor_hidden_size=self.config.actor_hidden_size,
            critic_hidden_size=self.config.critic_hidden_size,
            normalize=normalize_fn,
            initialize=initializer,
            activation=activation,
            activation_action=activation_action,
            device=self.device,
            use_distributed_training=self.distributed_training,
        )

    def get_actions(self, observations: np.ndarray, test_mode: Optional[bool] = False):
        _, actions_output = self.policy(observations, deterministic=bool(test_mode))
        actions = actions_output.detach().cpu().numpy()
        return {'actions': actions}
