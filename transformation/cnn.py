
from functools import partial
from typing import Tuple, Optional

import gym
import numpy as np
import torch

from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn


class CnnFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, features_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features_dim, features_dim, 1),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)


class Sum(nn.Module):
    def __init__(self, dim: Optional[int | Tuple[int]] = 1):
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.sum(tensor, dim=self.dim)


class FcnDecoder(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        features_dim: int,
    ):
        if not isinstance(observation_space, spaces.Box):
            raise ValueError(f'Incorrect type for observation space: {type(observation_space)}')
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError(f'Incorrect type for action space: {type(action_space)}')

        super().__init__()

        circuit_area = observation_space.shape[1] * observation_space.shape[2]
        action_channels = action_space.n // circuit_area

        self.policy_net = nn.Sequential(
            nn.ConvTranspose2d(features_dim, features_dim, 3, padding=1, stride=2, output_padding=1),
            nn.ConvTranspose2d(features_dim, action_channels, 3, padding=1, stride=2, output_padding=1),
        )
        self.value_net = nn.Sequential(
            nn.ConvTranspose2d(features_dim, features_dim, 3, padding=1, stride=2, output_padding=1),
            nn.ConvTranspose2d(features_dim, 1, 3, padding=1, stride=2, output_padding=1),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class MaskableActorCriticFcnPolicy(MaskableActorCriticPolicy):
    def _build(self, lr_schedule: Schedule):
        self._build_mlp_extractor()

        self.action_net = nn.Sequential(
            nn.Flatten(),
        )
        self.value_net = nn.Sequential(
            nn.Flatten(),
            Sum(),
        )

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = FcnDecoder(self.observation_space, self.action_space, self.features_dim)
