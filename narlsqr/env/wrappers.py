
from math import inf
from typing import Optional, Any, SupportsFloat

import gymnasium as gym
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.models import BackendProperties
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import DenseLayout

from narlsqr.env import RoutingEnv, RoutingObs
from narlsqr.generators.circuit import CircuitGenerator
from narlsqr.generators.noise import NoiseGenerator, get_error_rates_from_backend_properties


class TrainingWrapper(gym.Wrapper[RoutingObs, int, RoutingObs, int]):
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
        if env.num_edges != noise_generator.num_edges:
            raise ValueError(f'Number of edges in environment and noise generator must be'
                             f'equal, got {env.num_edges = } and {noise_generator.num_edges = }')

        if recalibration_interval <= 0:
            raise ValueError(f'Recalibration interval must be positive, got {recalibration_interval}')

        if episodes_per_circuit <= 0:
            raise ValueError(f'Training episodes per circuit must be positive, got {episodes_per_circuit}')

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
            error_rates = self.noise_generator.generate()
            self.env.calibrate(error_rates)

        self.current_iter += 1

        return super().reset(seed=seed, options=options)


class StochasticPolicyWrapper(gym.Wrapper[RoutingObs, int, RoutingObs, int]):
    best_reward: float
    best_circuit: QuantumCircuit

    unwrapped: RoutingEnv

    def __init__(self, env: RoutingEnv):
        super().__init__(env)

        self.reset_best_circuit()
        self.total_reward = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[RoutingObs, dict[str, Any]]:
        self.total_reward = 0.0
        return super().reset(seed=seed, options=options)

    def step(self, action: int) -> tuple[RoutingObs, SupportsFloat, bool, bool, dict[str, Any]]:
        result = super().step(action)

        _, reward, terminated, *_ = result
        self.total_reward += reward

        if terminated and self.total_reward > self.best_reward:
            self.best_reward = self.total_reward
            self.best_circuit = self.unwrapped.routed_circuit()

        return result

    def reset_best_circuit(self):
        self.best_reward = -inf
        self.best_circuit = self.unwrapped.circuit.copy_empty_like()


class EvaluationWrapper(gym.Wrapper[RoutingObs, int, RoutingObs, int]):
    """
    Wraps a :py:class:`RoutingEnv`, automatically generating circuits to evaluate the performance of a reinforcement
    learning model.

    :param env: :py:class:`RoutingEnv` to wrap.
    :param circuit_generator: Random circuit generator to be used during training.
    :param backend_properties: ``BackendProperties`` object containing device calibration data.
    :param evaluation_episodes: Number of evaluation episodes per generated circuit.

    :ivar current_iter: Current training iteration.
    """

    env: StochasticPolicyWrapper
    unwrapped: RoutingEnv

    def __init__(
        self,
        env: RoutingEnv,
        circuit_generator: CircuitGenerator,
        backend_properties: BackendProperties,
        evaluation_episodes: int = 10,
    ):
        self.circuit_generator = circuit_generator
        self.evaluation_iters = evaluation_episodes

        error_rates = np.array(get_error_rates_from_backend_properties(backend_properties), copy=False)
        env.calibrate(error_rates)

        qiskit_coupling_map = CouplingMap(env.coupling_map.edge_list())
        env.layout_pass = DenseLayout(qiskit_coupling_map, backend_properties)

        super().__init__(StochasticPolicyWrapper(env))

        self.current_iter = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[RoutingObs, dict[str, Any]]:
        if self.current_iter % self.evaluation_iters == 0:
            self.unwrapped.circuit = self.circuit_generator.generate()
            self.env.reset_best_circuit()

        self.current_iter += 1

        return super().reset(seed=seed, options=options)
