from collections import defaultdict
from typing import Optional

import numpy as np
from ray.rllib import BaseEnv, RolloutWorker, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID

from routing.env import RoutingEnv


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
        envs: list[RoutingEnv] = base_env.get_sub_environments()

        metrics = defaultdict(list)
        for env in envs:
            for metric, value in env.numeric_metrics.items():
                metrics[metric].append(value)

        episode.custom_metrics.update({k: np.mean(v) for k, v in metrics.items()})
