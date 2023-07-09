
from typing import Any, Optional, Union

import numpy as np
import torch
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn


class MaskableQNetwork(QNetwork):
    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = True,
        action_masks: Optional[np.ndarray] = None,
    ) -> th.Tensor:
        q_values = self(observation)
        if action_masks is not None:
            action_masks = torch.tensor(action_masks, dtype=torch.bool, device=self.device)
            q_values = torch.where(action_masks, q_values, -torch.inf)
        return q_values.argmax(dim=1).reshape(-1)


class MaskableDQNPolicy(DQNPolicy):
    q_net: MaskableQNetwork
    q_net_target: MaskableQNetwork

    def make_q_net(self) -> MaskableQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskableQNetwork(**net_args).to(self.device)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = True,
        action_masks: Optional[np.ndarray] = None,
    ) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic, action_masks=action_masks)

    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = True,
        action_masks: Optional[np.ndarray] = None,
    ) -> th.Tensor:
        return self.q_net._predict(observation, deterministic=deterministic, action_masks=action_masks)

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic, action_masks=action_masks)
        # Convert to numpy, and reshape the original action space
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state


class MaskableMultiInputDQNPolicy(MaskableDQNPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
