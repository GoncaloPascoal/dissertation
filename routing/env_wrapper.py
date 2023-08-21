
from typing import Optional, Any

import gymnasium as gym
import numpy as np
from qiskit import transpile

from routing.circuit_gen import CircuitGenerator
from routing.env import RoutingEnv, RoutingObs
from routing.noise import NoiseGenerator


class TrainingWrapper(gym.Wrapper[RoutingObs, int]):
    """
    Wraps a :py:class:`RoutingEnv`, automatically generating circuits and gate error rates at fixed intervals to
    help train deep learning algorithms.

    :param env: :py:class:`RoutingEnv` to wrap.
    :param circuit_generator: Random circuit generator to be used during training.
    :param noise_generator: Generator for two-qubit gate error rates. Should be provided iff the environment is
                            noise-aware.
    :param recalibration_interval: Error rates will be regenerated after routing this many circuits.
    :param episodes_per_circuit: Number of episodes per generated circuit.

    :ivar current_iter: Current training iteration.
    """

    env: RoutingEnv

    def __init__(
        self,
        env: RoutingEnv,
        circuit_generator: CircuitGenerator,
        noise_generator: NoiseGenerator,
        recalibration_interval: int = 64,
        episodes_per_circuit: int = 1,
    ):
        if recalibration_interval <= 0:
            raise ValueError(f'Recalibration interval must be positive, got {recalibration_interval}')

        self.circuit_generator = circuit_generator
        self.noise_generator = noise_generator
        self.recalibration_interval = recalibration_interval
        self.episodes_per_circuit = episodes_per_circuit

        env.initial_mapping = np.arange(env.num_qubits)

        super().__init__(env)

        self.current_iter = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[RoutingObs, dict[str, Any]]:
        if self.current_iter % self.episodes_per_circuit == 0:
            self.env.circuit = self.circuit_generator.generate()

        if self.current_iter % self.recalibration_interval == 0:
            error_rates = self.noise_generator.generate_error_rates(self.env.num_edges)
            self.env.calibrate(error_rates)

        self.current_iter += 1

        return super().reset(seed=seed, options=options)


class EvaluationWrapper(gym.Wrapper[RoutingObs, int]):
    """
    Wraps a :py:class:`RoutingEnv`, automatically generating circuits to evaluate the performance of a reinforcement
    learning model.

    :param env: :py:class:`RoutingEnv` to wrap.
    :param circuit_generator: Random circuit generator to be used during training.
    :param noise_generator: Generator for two-qubit gate error rates. Should be provided iff the environment is
                            noise-aware. As this is an evaluation environment, the error rates are only generated once.
    :param evaluation_iters: Number of evaluation iterations per generated circuit.

    :ivar current_iter: Current training iteration.
    """

    env: RoutingEnv

    def __init__(
        self,
        env: RoutingEnv,
        circuit_generator: CircuitGenerator,
        noise_generator: NoiseGenerator,
        evaluation_iters: int = 20,
    ):
        self.circuit_generator = circuit_generator
        self.evaluation_iters = evaluation_iters

        env.calibrate(noise_generator.generate_error_rates(env.num_edges))
        env.initial_mapping = np.arange(env.num_qubits)

        super().__init__(env)

        self.current_iter = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[RoutingObs, dict[str, Any]]:
        if self.current_iter % self.evaluation_iters == 0:
            # TODO: extract IBM gate set to constant
            self.env.circuit = transpile(
                self.circuit_generator.generate(),
                basis_gates=['cx', 'id', 'rz', 'sx', 'x'],
                optimization_level=0,
                seed_transpiler=0,
            )

        self.current_iter += 1

        return super().reset(seed=seed, options=options)
