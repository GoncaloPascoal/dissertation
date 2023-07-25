
from typing import TypeVar, Optional, Any

import gymnasium as gym
import numpy as np
from nptyping import NDArray

from routing.circuit_gen import CircuitGenerator
from routing.env import RoutingEnv, RoutingObsType
from routing.noise import NoiseGenerator

_RoutingEnvType = TypeVar('_RoutingEnvType', bound=RoutingEnv)


def _generate_random_mapping(num_qubits: int) -> NDArray:
    mapping = np.arange(num_qubits)
    np.random.shuffle(mapping)
    return mapping


class TrainingWrapper(gym.Wrapper[RoutingObsType, int, RoutingObsType, int]):
    """
    Wraps a :py:class:`RoutingEnv`, automatically generating circuits and gate error rates at fixed intervals to
    help train deep learning algorithms.

    :param env: :py:class:`RoutingEnv` to wrap.
    :param circuit_generator: Random circuit generator to be used during training.
    :param noise_generator: Generator for two-qubit gate error rates. Should be provided iff the environment is
                            noise-aware.
    :param training_iters: Number of episodes per generated circuit.

    :ivar iter: Current training iteration.
    """

    env: RoutingEnv
    noise_generator: Optional[NoiseGenerator]

    def __init__(
        self,
        env: RoutingEnv,
        circuit_generator: CircuitGenerator,
        noise_generator: Optional[NoiseGenerator] = None,
        training_iters: int = 1,
    ):
        if (noise_generator is not None) != env.noise_aware:
            raise ValueError('Noise-awareness mismatch between wrapper and env')

        self.circuit_generator = circuit_generator
        self.noise_generator = noise_generator
        self.training_iters = training_iters

        super().__init__(env)

        self.iter = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None
    ) -> tuple[RoutingObsType, dict[str, Any]]:
        self.env.initial_mapping = _generate_random_mapping(self.num_qubits)

        if self.iter % self.training_iters == 0:
            self.env.circuit = self.circuit_generator.generate()

        if self.noise_aware and self.iter % self.noise_generator.recalibration_interval == 0:
            error_rates = self.noise_generator.generate_error_rates(self.num_edges)
            self.env.calibrate(error_rates)

        self.iter += 1

        return super().reset(seed=seed, options=options)


class EvaluationWrapper(gym.Wrapper[RoutingObsType, int, RoutingObsType, int]):
    """
    Wraps a :py:class:`RoutingEnv`, automatically generating circuits to evaluate the performance of a reinforcement
    learning model.

    :param env: :py:class:`RoutingEnv` to wrap.
    :param circuit_generator: Random circuit generator to be used during training.
    :param noise_generator: Generator for two-qubit gate error rates. Should be provided iff the environment is
                            noise-aware. As this is an evaluation environment, the error rates are only generated once.
    :param evaluation_iters: Number of evaluation iterations per generated circuit.

    :ivar iter: Current training iteration.
    """

    env: RoutingEnv

    def __init__(
        self,
        env: RoutingEnv,
        circuit_generator: CircuitGenerator,
        noise_generator: Optional[NoiseGenerator] = None,
        evaluation_iters: int = 20,
    ):
        self.circuit_generator = circuit_generator
        self.evaluation_iters = evaluation_iters

        if noise_generator is not None:
            env.calibrate(noise_generator.generate_error_rates(env.num_edges))
        env.initial_mapping = _generate_random_mapping(env.num_qubits)

        super().__init__(env)

        self.iter = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None
    ) -> tuple[RoutingObsType, dict[str, Any]]:
        if self.iter % self.evaluation_iters == 0:
            self.env.circuit = self.circuit_generator.generate()

        self.iter += 1

        return super().reset(seed=seed, options=options)
