
from typing import Any

import torch
from gymnasium import spaces
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import ModelConfigDict
from torch import TensorType
from torch import nn


class ActionMaskModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        if hasattr(obs_space, 'original_space'):
            original_space = obs_space.original_space
        else:
            raise ValueError(f'Could not obtain original observation space')

        if not (
            isinstance(original_space, spaces.Dict)
            and 'action_mask' in original_space.spaces
            and 'true_obs' in original_space.spaces
        ):
            raise ValueError(f'Invalid observation space: {original_space}')

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        preprocessor = DictFlatteningPreprocessor(original_space['true_obs'])

        self.internal_model = FullyConnectedNetwork(
            preprocessor.observation_space,
            action_space,
            num_outputs,
            model_config,
            name + '_internal',
        )

    def forward(
        self,
        input_dict: dict[str, Any],
        state: list[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, list[TensorType]):
        action_mask = input_dict['obs']['action_mask']

        obs_tensors = [torch.flatten(t, start_dim=1) for t in input_dict['obs']['true_obs'].values()]
        true_obs_flat = torch.cat(obs_tensors, dim=1)

        # Compute the unmasked logits
        logits, _ = self.internal_model({'obs': true_obs_flat})

        # Convert action_mask into a [0.0 || -inf]-type mask
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()
