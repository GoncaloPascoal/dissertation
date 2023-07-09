
from typing import Any

from gymnasium import spaces
import torch
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import ModelConfigDict
from torch import TensorType


class TorchActionMaskModel(TorchModelV2):
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        if not (
            isinstance(obs_space, spaces.Dict)
            and 'action_mask' in obs_space.spaces
            and 'true_obs' in obs_space.spaces
        ):
            raise ValueError('Invalid observation space')

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            obs_space['true_obs'],
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

        # Compute the unmasked logits
        logits, _ = self.internal_model({'obs': input_dict['obs']['true_obs']})

        # Convert action_mask into a [0.0 || -inf]-type mask
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()
