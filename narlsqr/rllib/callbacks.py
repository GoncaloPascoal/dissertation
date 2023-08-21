
from typing import Optional

from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID

from narlsqr.env.wrappers import TrainingWrapper

class RoutingCallbacks(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: dict[PolicyID, Policy],
        episode: EpisodeV2,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        if env_index is not None:
            training_env: TrainingWrapper = base_env.get_sub_environments()[env_index]

            if training_env.env.log_metrics:
                episode.custom_metrics.update(training_env.env.metrics)
                episode.hist_data.update({k: [v] for k, v in training_env.env.metrics.items()})
