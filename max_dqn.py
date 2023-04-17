from typing import Type, List, Union, Optional

import torch

from ray.rllib import SampleBatch, Policy
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.dqn import DQN, DQNTorchPolicy
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.algorithms.dqn.dqn_torch_policy import compute_q_values, F
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils import override
from ray.rllib.utils.torch_utils import FLOAT_MIN, huber_loss, l2_loss, softmax_cross_entropy_with_logits
from ray.rllib.utils.typing import TensorType


class MaxQLoss:
    def __init__(
        self,
        q_t_selected: TensorType,
        q_logits_t_selected: TensorType,
        q_tp1_best: TensorType,
        q_probs_tp1_best: TensorType,
        importance_weights: TensorType,
        rewards: TensorType,
        done_mask: TensorType,
        gamma: float = 0.99,
        n_step: int = 1,
        num_atoms: int = 1,
        v_min: float = -10.0,
        v_max: float = 10.0,
        loss_fn=huber_loss,
    ):
        if num_atoms > 1:
            # Distributional Q-learning which corresponds to an entropy loss
            z = torch.arange(0.0, num_atoms, dtype=torch.float32).to(rewards.device)
            z = v_min + z * (v_max - v_min) / float(num_atoms - 1)

            # (batch_size, 1) * (1, num_atoms) = (batch_size, num_atoms)
            r_tau = torch.unsqueeze(rewards, -1) + gamma**n_step * torch.unsqueeze(
                1.0 - done_mask, -1
            ) * torch.unsqueeze(z, 0)
            r_tau = torch.clamp(r_tau, v_min, v_max)
            b = (r_tau - v_min) / ((v_max - v_min) / float(num_atoms - 1))
            lb = torch.floor(b)
            ub = torch.ceil(b)

            # Indispensable judgement which is missed in most implementations
            # when b happens to be an integer, lb == ub, so pr_j(s', a*) will
            # be discarded because (ub-b) == (b-lb) == 0.
            floor_equal_ceil = ((ub - lb) < 0.5).float()

            # (batch_size, num_atoms, num_atoms)
            l_project = F.one_hot(lb.long(), num_atoms)
            # (batch_size, num_atoms, num_atoms)
            u_project = F.one_hot(ub.long(), num_atoms)
            ml_delta = q_probs_tp1_best * (ub - b + floor_equal_ceil)
            mu_delta = q_probs_tp1_best * (b - lb)
            ml_delta = torch.sum(l_project * torch.unsqueeze(ml_delta, -1), dim=1)
            mu_delta = torch.sum(u_project * torch.unsqueeze(mu_delta, -1), dim=1)
            m = ml_delta + mu_delta

            # Rainbow paper claims that using this cross entropy loss for
            # priority is robust and insensitive to `prioritized_replay_alpha`
            self.td_error = softmax_cross_entropy_with_logits(
                logits=q_logits_t_selected, labels=m.detach()
            )
            self.loss = torch.mean(self.td_error * importance_weights)
            self.stats = {
                # TODO: better Q stats for dist dqn
            }
        else:
            q_tp1_best_masked = (1.0 - done_mask) * q_tp1_best

            # Compute RHS of max-Bellman equation
            q_t_selected_target = torch.maximum(rewards, gamma**n_step * q_tp1_best_masked)

            # compute the error (potentially clipped)
            self.td_error = q_t_selected - q_t_selected_target.detach()
            self.loss = torch.mean(importance_weights.float() * loss_fn(self.td_error))
            self.stats = {
                "mean_q": torch.mean(q_t_selected),
                "min_q": torch.min(q_t_selected),
                "max_q": torch.max(q_t_selected),
            }


class MaxDQNTorchPolicy(DQNTorchPolicy):
    @override(DQNTorchPolicy)
    def loss(
        self,
        model: ModelV2,
        _: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        # Q-network evaluation.
        q_t, q_logits_t, q_probs_t, _ = compute_q_values(
            self,
            model,
            {'obs': train_batch[SampleBatch.CUR_OBS]},
            explore=False,
            is_training=True,
        )

        # Target Q-network evaluation.
        q_tp1, q_logits_tp1, q_probs_tp1, _ = compute_q_values(
            self,
            self.target_models[model],
            {'obs': train_batch[SampleBatch.NEXT_OBS]},
            explore=False,
            is_training=True,
        )

        # Q scores for actions which we know were selected in the given state.
        one_hot_selection = F.one_hot(
            train_batch[SampleBatch.ACTIONS].long(), self.action_space.n
        )
        q_t_selected = torch.sum(
            torch.where(q_t > FLOAT_MIN, q_t, torch.tensor(0.0, device=q_t.device))
            * one_hot_selection,
            1,
        )
        q_logits_t_selected = torch.sum(
            q_logits_t * torch.unsqueeze(one_hot_selection, -1), 1
        )

        # Compute estimate of best possible value starting from state at t + 1.
        if self.config['double_q']:
            (
                q_tp1_using_online_net,
                q_logits_tp1_using_online_net,
                q_dist_tp1_using_online_net,
                _,
            ) = compute_q_values(
                self,
                model,
                {'obs': train_batch[SampleBatch.NEXT_OBS]},
                explore=False,
                is_training=True,
            )
            q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)
            q_tp1_best_one_hot_selection = F.one_hot(
                q_tp1_best_using_online_net, self.action_space.n
            )
            q_tp1_best = torch.sum(
                torch.where(
                    q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=q_tp1.device)
                )
                * q_tp1_best_one_hot_selection,
                1,
            )
            q_probs_tp1_best = torch.sum(
                q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
            )
        else:
            q_tp1_best_one_hot_selection = F.one_hot(
                torch.argmax(q_tp1, 1), self.action_space.n
            )
            q_tp1_best = torch.sum(
                torch.where(
                    q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=q_tp1.device)
                )
                * q_tp1_best_one_hot_selection,
                1,
            )
            q_probs_tp1_best = torch.sum(
                q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
            )

        loss_fn = huber_loss if self.config['td_error_loss_fn'] == 'huber' else l2_loss

        max_q_loss = MaxQLoss(
            q_t_selected,
            q_logits_t_selected,
            q_tp1_best,
            q_probs_tp1_best,
            train_batch[PRIO_WEIGHTS],
            train_batch[SampleBatch.REWARDS],
            train_batch[SampleBatch.TERMINATEDS].float(),
            self.config['gamma'],
            self.config['n_step'],
            self.config['num_atoms'],
            self.config['v_min'],
            self.config['v_max'],
            loss_fn,
        )

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats['td_error'] = max_q_loss.td_error
        # TD-error tensor in final stats
        # will be concatenated and retrieved for each individual batch item.
        model.tower_stats['q_loss'] = max_q_loss

        return max_q_loss.loss


class MaxDQN(DQN):
    @classmethod
    @override(DQN)
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Optional[Type[Policy]]:
        if config['framework'] == 'torch':
            return MaxDQNTorchPolicy
        return None

