
from typing import Any, Optional, cast

import numpy as np
import torch
from gymnasium import spaces
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import ModelConfigDict
from torch import TensorType
from torch import nn

class ActionMaskModel(TorchModelV2, nn.Module):
    embedding: Optional[nn.Embedding]

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):
        if hasattr(obs_space, 'original_space'):
            original_space: spaces.Dict = obs_space.original_space
        else:
            raise ValueError(f'Could not obtain original observation space')

        if not (
            isinstance(original_space, spaces.Dict)
            and 'action_mask' in original_space.spaces
            and 'true_obs' in original_space.spaces
        ):
            raise ValueError('Observation space must be a dictionary containing an action mask and the '
                             'true observation')

        true_obs_space = original_space['true_obs']
        if not isinstance(true_obs_space, spaces.Dict):
            raise ValueError(f'True observation space must be of type `Dict`, got {true_obs_space}')

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        embedding_dim = kwargs.get('embedding_dim')
        if 'circuit_matrix' in true_obs_space.spaces and embedding_dim is not None:
            circuit_matrix = cast(spaces.Box, true_obs_space['circuit_matrix'])
            self.embedding = nn.Embedding(circuit_matrix.high.flat[0] + 1, embedding_dim)
        else:
            self.embedding = None

        total_features = 0
        low = np.inf
        high = -np.inf

        for key, space in true_obs_space.items():
            features = np.prod(space.shape)
            if key == 'circuit_matrix' and self.embedding is not None:
                features *= self.embedding.embedding_dim

            total_features += features
            low = min(low, np.min(space.low))
            high = max(high, np.max(space.high))

        self.internal_model = FullyConnectedNetwork(
            spaces.Box(low, high, shape=(total_features,)),
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
        true_obs = input_dict['obs']['true_obs']

        if self.embedding is not None:
            true_obs['circuit_matrix'] = self.embedding(true_obs['circuit_matrix'].int())

        obs_tensors = [torch.flatten(t, start_dim=1) for t in true_obs.values()]
        true_obs_flat = torch.cat(obs_tensors, dim=1)

        # Compute the unmasked logits
        logits, _ = self.internal_model({'obs': true_obs_flat})

        # Convert action_mask into a [0.0 || -inf]-type mask
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()
