
from gymnasium.spaces import Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.torch_utils import FLOAT_MIN

import torch
import torch.nn as nn


class TorchActionMaskModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    ):
        orig_space = getattr(obs_space, 'original_space', obs_space)
        assert isinstance(orig_space, Tuple) and len(orig_space.spaces) == 2

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.internal_model = FullyConnectedNetwork(
            orig_space[1],
            action_space,
            num_outputs,
            model_config,
            name + '_internal',
        )

        # Disable action masking, will likely lead to invalid actions
        self.no_masking = False
        if 'no_masking' in model_config['custom_model_config']:
            self.no_masking = model_config['custom_model_config']['no_masking']

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict['obs'][0]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({'obs': input_dict['obs'][1]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()
