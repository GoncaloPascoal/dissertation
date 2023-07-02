
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from gymnasium import spaces
from nptyping import NDArray

from routing.env import LayeredRoutingEnv, RoutingEnv


RoutingEnvType = TypeVar('RoutingEnvType', bound=RoutingEnv)


class ObsModule(ABC, Generic[RoutingEnvType]):
    @staticmethod
    @abstractmethod
    def key() -> str:
        raise NotImplementedError

    @abstractmethod
    def space(self, env: RoutingEnvType) -> spaces.Box:
        raise NotImplementedError

    @abstractmethod
    def obs(self, env: RoutingEnvType) -> NDArray:
        raise NotImplementedError


class LogReliabilities(ObsModule[RoutingEnv]):
    @staticmethod
    def key() -> str:
        return 'log_reliabilities'

    def space(self, env: RoutingEnv) -> spaces.Box:
        return spaces.Box(env.noise_config.min_log_reliability, 0.0, shape=env.log_reliabilities.shape)

    def obs(self, env: RoutingEnv) -> NDArray:
        return env.log_reliabilities.copy()


class LockedNodes(ObsModule[LayeredRoutingEnv]):
    @staticmethod
    def key() -> str:
        return 'locked_nodes'

    def space(self, env: LayeredRoutingEnv) -> spaces.Box:
        return spaces.Box(0, 4 if env.use_decomposed_actions else 1, shape=(env.num_qubits,), dtype=np.int32)

    def obs(self, env: LayeredRoutingEnv) -> NDArray:
        return np.array([env.scheduling_map.get(q, 0) for q in range(env.num_qubits)])
