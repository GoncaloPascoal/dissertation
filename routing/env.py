
import copy
import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import MutableMapping, Iterable
from dataclasses import dataclass, field
from math import e
from typing import Optional, Tuple, Dict, Any, List, SupportsFloat, Iterator, Self, TypeVar, Generic, Type

import gymnasium as gym
import numpy as np
import rustworkx as rx
from gymnasium import spaces
from nptyping import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate, CXGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode

from dag_utils import dag_layers
from routing.circuit_gen import CircuitGenerator
from utils import qubits_to_indices, indices_to_qubits


@dataclass
class NoiseConfig:
    mean: float
    std: float

    recalibration_interval: int = field(default=16, kw_only=True)

    def generate_error_rates(self, n: int) -> NDArray:
        return np.random.normal(self.mean, self.std, n).clip(0.0, 1.0)

    @staticmethod
    def calculate_log_reliabilities(
        error_rates: NDArray,
        log_base: float = e,
        min_log_reliability: float = -100,
    ) -> NDArray:
        if np.any((error_rates < 0.0) | (error_rates > 1.0)):
            raise ValueError('Got invalid values for error rates')

        return np.where(
            error_rates < 1.0,
            np.emath.logn(log_base, 1.0 - error_rates),
            -np.inf
        ).clip(min_log_reliability)


RoutingObsType = Dict[str, NDArray]
GateSchedulingList = List[Tuple[DAGOpNode, Tuple[int, ...]]]
BridgeArgs = Tuple[DAGOpNode, Tuple[int, int, int]]


class RoutingEnv(gym.Env[RoutingObsType, int], ABC):
    """
    Base qubit routing environment.

    :param circuit: Quantum circuit to compile.
    :param coupling_map: Graph representing the connectivity of the target device.
    :param initial_mapping: Initial mapping from physical nodes to logical qubits. If ``None``, a random initial mapping
                            will be used for each training iteration.
    :param allow_bridge_gate: Allow the use of BRIDGE gates when routing.
    :param error_rates: Array of two-qubit gate error rates. If ``None``, routing will be noise-unaware.
    :param obs_modules: Observation modules that define the key-value pairs in observations.

    :ivar node_to_qubit: Current mapping from physical nodes to logical qubits.
    :ivar qubit_to_node: Current mapping from logical qubits to physical nodes.
    :ivar routed_dag: ``DAGCircuit`` containing already routed gates.
    :ivar log_reliabilities: Array of logarithms of two-qubit gate reliabilities (``reliability = 1 - error_rate``)
    :ivar log_reliabilities_map: Dictionary that maps edges to their corresponding log reliability values.
    """

    observation_space: spaces.Dict
    action_space: spaces.Discrete

    initial_mapping: NDArray
    error_rates: Optional[NDArray]
    obs_modules: List['ObsModule']
    log_reliabilities: NDArray
    log_reliabilities_map: Dict[Tuple[int, int], float]

    def __init__(
        self,
        circuit: QuantumCircuit,
        coupling_map: rx.PyGraph,
        initial_mapping: NDArray,
        allow_bridge_gate: bool = True,
        error_rates: Optional[NDArray] = None,
        obs_modules: Optional[List['ObsModule']] = None,
    ):
        if initial_mapping.shape != (coupling_map.num_nodes(),):
            raise ValueError('Initial mapping has invalid shape for the provided coupling map')

        if error_rates is not None and error_rates.shape != (coupling_map.num_edges(),):
            raise ValueError('Error rates have invalid shape for the provided coupling map')

        self.circuit = circuit
        self.coupling_map = coupling_map
        self.initial_mapping = initial_mapping
        self.allow_bridge_gate = allow_bridge_gate
        self.error_rates = error_rates
        self.obs_modules = [] if obs_modules is None else obs_modules

        # Computations using coupling map
        self.num_qubits = coupling_map.num_nodes()
        self.num_edges = coupling_map.num_edges()
        self.edge_list = list(coupling_map.edge_list())  # type: ignore

        self.shortest_paths = rx.graph_all_pairs_dijkstra_shortest_paths(coupling_map, lambda _: 1.0)
        self.distance_matrix = rx.graph_distance_matrix(coupling_map).astype(np.int32)

        if self.allow_bridge_gate:
            self.bridge_pairs = sorted(
                (j, i) for i in range(self.num_qubits) for j in range(i) if self.distance_matrix[j, i] == 2
            )
        else:
            self.bridge_pairs = []

        # State information
        self.node_to_qubit = initial_mapping.copy()
        self.qubit_to_node = np.zeros_like(self.node_to_qubit)
        for node, qubit in enumerate(self.node_to_qubit):
            self.qubit_to_node[qubit] = node

        self.dag = circuit_to_dag(self.circuit)
        self.routed_dag = self.dag.copy_empty_like()

        # Noise-awareness information
        if self.noise_aware:
            self.obs_modules.append(LogReliabilities())
            self.calibrate(self.error_rates)

        bridge_circuit = QuantumCircuit(3)
        for _ in range(2):
            bridge_circuit.cx(1, 2)
            bridge_circuit.cx(0, 1)

        self.bridge_gate = bridge_circuit.to_gate(label='BRIDGE')
        self.bridge_gate.name = 'bridge'
        self.swap_gate = SwapGate()

    @property
    def noise_aware(self) -> bool:
        return self.error_rates is not None

    @property
    def terminated(self) -> bool:
        return self.dag.size() == 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[RoutingObsType, Dict[str, Any]]:
        self.dag = circuit_to_dag(self.circuit)
        self.routed_dag = self.dag.copy_empty_like()

        self._reset_state()

        return self.current_obs(), {}

    def calibrate(self, error_rates: NDArray):
        """
        Given an array of edge error rates, calculates the log reliabilities according to the noise
        configuration, which can, in turn, be used to obtain agent rewards.

        :param error_rates: Array of two-qubit gate error rates.
        """
        self.error_rates = error_rates.copy()
        self.log_reliabilities = NoiseConfig.calculate_log_reliabilities(error_rates)

        m = {}
        for edge, value in zip(self.edge_list, self.log_reliabilities):  # type: ignore
            m[edge] = value
            m[edge[::-1]] = value

        self.log_reliabilities_map = m

    def copy(self) -> Self:
        """
        Returns a copy of the environment. Attributes that are not mutated when stepping through the environment are
        copied by reference; other attributes are copied by value.
        """
        env = copy.copy(self)

        env.node_to_qubit = self.node_to_qubit.copy()
        env.qubit_to_node = self.qubit_to_node.copy()

        env.dag = self.dag.copy_empty_like()
        env.dag.compose(self.dag)

        env.routed_dag = self.routed_dag.copy_empty_like()
        env.routed_dag.compose(self.routed_dag)

        return env

    def current_obs(self) -> RoutingObsType:
        return {module.key(): module.obs(self) for module in self.obs_modules}

    def routed_circuit(self) -> QuantumCircuit:
        return dag_to_circuit(self.routed_dag)

    @abstractmethod
    def action_masks(self) -> NDArray:
        """
        Returns a boolean NumPy array where the ith element is true if the ith action is valid in the current state.
        """
        raise NotImplementedError

    @abstractmethod
    def _update_state(self):
        raise NotImplementedError

    def _obs_spaces(self) -> Dict[str, spaces.Space]:
        return {module.key(): module.space(self) for module in self.obs_modules}

    def _schedule_gates(self, gates: GateSchedulingList):
        for op_node, nodes in gates:
            qargs = indices_to_qubits(self.circuit, nodes)
            self.routed_dag.apply_operation_back(op_node.op, qargs)
            self.dag.remove_op_node(op_node)

    def _schedulable_gates(self, only_front_layer: bool = False) -> GateSchedulingList:
        gates = []
        layers = [self.dag.front_layer()] if only_front_layer else dag_layers(self.dag)
        stop = False

        for layer in layers:
            for op_node in layer:
                indices = qubits_to_indices(self.circuit, op_node.qargs)
                nodes = tuple(self.qubit_to_node[i] for i in indices)

                if len(nodes) != 2 or self.coupling_map.has_edge(nodes[0], nodes[1]):
                    gates.append((op_node, nodes))
                else:
                    stop = True

            if stop:
                break

        return gates

    def _bridge(self, control: int, middle: int, target: int):
        indices = (control, middle, target)
        qargs = indices_to_qubits(self.circuit, indices)

        self.routed_dag.apply_operation_back(self.bridge_gate, qargs)

    def _bridge_args(self, pair: Tuple[int, int]) -> Optional[BridgeArgs]:
        for op_node in self.dag.front_layer():
            if isinstance(op_node.op, CXGate):
                indices = qubits_to_indices(self.circuit, op_node.qargs)
                nodes = tuple(self.qubit_to_node[i] for i in indices)

                if tuple(sorted(nodes)) == pair:
                    control, target = nodes
                    return op_node, tuple(self.shortest_paths[control][target])

        return None

    def _swap(self, edge: Tuple[int, int]):
        qargs = indices_to_qubits(self.circuit, edge)

        self.routed_dag.apply_operation_back(self.swap_gate, qargs)

        # Update mappings
        node_a, node_b = edge
        qubit_a, qubit_b = (self.node_to_qubit[n] for n in edge)

        self.node_to_qubit[node_a], self.node_to_qubit[node_b] = qubit_b, qubit_a
        self.qubit_to_node[qubit_a], self.qubit_to_node[qubit_b] = node_b, node_a

    def _reset_state(self):
        self.node_to_qubit = self.initial_mapping.copy()

        for node, qubit in enumerate(self.node_to_qubit):
            self.qubit_to_node[qubit] = node

        self._update_state()


class SequentialRoutingEnv(RoutingEnv):
    """
    Sequential routing environment, where SWAP and BRIDGE operations are iteratively added to the routed circuit.
    Subclasses should override the :py:meth:`_current_obs` and :py:meth:`_obs_spaces` methods to provide their
    desired observation representations.

    :param restrict_swaps_to_front_layer: Restrict SWAP operations to edges involving qubits that are part of two-qubit
        operations in the front (first) layer of the original circuit.
    :param base_gate_reward: Base reward for scheduling a two-qubit gate when the environment isn't noise-aware.
    """

    _blocked_swap: Optional[int]

    def __init__(
        self,
        circuit: QuantumCircuit,
        coupling_map: rx.PyGraph,
        initial_mapping: NDArray,
        allow_bridge_gate: bool = True,
        error_rates: Optional[NDArray] = None,
        obs_modules: Optional[List['ObsModule']] = None,
        restrict_swaps_to_front_layer: bool = True,
        base_gate_reward: float = -1.0,
    ):
        super().__init__(circuit, coupling_map, initial_mapping, allow_bridge_gate, error_rates, obs_modules)

        self.restrict_swaps_to_front_layer = restrict_swaps_to_front_layer
        self.base_gate_reward = base_gate_reward

        self._blocked_swap = None
        self._scheduling_reward = 0.0

        self.observation_space = spaces.Dict(self._obs_spaces())

        num_actions = self.num_edges + len(self.bridge_pairs)
        self.action_space = spaces.Discrete(num_actions)

    def step(self, action: int) -> Tuple[RoutingObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if self.terminated:
            # Environment terminated immediately and does not require routing
            return self.current_obs(), 0.0, True, False, {}

        is_swap = action < self.num_edges

        if is_swap:
            # SWAP action
            edge = self.edge_list[action]
            self._swap(edge)
            reward = self._swap_reward(edge)
        else:
            # BRIDGE action
            pair = self.bridge_pairs[action - self.num_edges]
            op_node, nodes = self._bridge_args(pair)

            self.dag.remove_op_node(op_node)
            self._bridge(*nodes)
            reward = self._bridge_reward(*nodes)

        self._update_state()
        # reward += self._scheduling_reward
        # self._blocked_swap = action if is_swap and self._scheduling_reward == 0.0 else None

        return self.current_obs(), reward, self.terminated, False, {}

    def action_masks(self) -> NDArray:
        mask = np.ones(self.action_space.n, dtype=bool)

        # Swap actions
        if self.restrict_swaps_to_front_layer:
            front_layer_nodes = set()

            for op_node in self.dag.front_layer():
                indices = qubits_to_indices(self.circuit, op_node.qargs)
                nodes = (self.qubit_to_node[i] for i in indices)
                front_layer_nodes.update(nodes)

            invalid_swap_actions = [
                i for i, edge in enumerate(self.edge_list)
                if not front_layer_nodes.intersection(edge)
            ]
            mask[invalid_swap_actions] = False

        if self._blocked_swap is not None:
            mask[self._blocked_swap] = False

        # Bridge actions
        invalid_bridge_actions = [
            self.num_edges + i for i, pair in enumerate(self.bridge_pairs)
            if self._bridge_args(pair) is None
        ]
        mask[invalid_bridge_actions] = False

        # TODO: remove
        if np.all(mask == False):
            print(dag_to_circuit(self.dag))
            print(self.node_to_qubit, self.qubit_to_node)
            print(self.terminated)

        return mask

    def _update_state(self):
        gates = self._schedulable_gates()
        self._schedule_gates(gates)
        self._scheduling_reward = sum(self._gate_reward(nodes) for _, nodes in gates if len(nodes) == 2)

    def _gate_reward(self, edge: Tuple[int, int]) -> float:
        if self.noise_aware:
            return self.log_reliabilities_map[edge]
        else:
            return self.base_gate_reward

    def _swap_reward(self, edge: Tuple[int, int]) -> float:
        if self.noise_aware:
            return 3.0 * self.log_reliabilities_map[edge]
        else:
            return 3.0 * self.base_gate_reward

    def _bridge_reward(self, control: int, middle: int, target: int) -> float:
        if self.noise_aware:
            return 2.0 * (
                self.log_reliabilities_map[(middle, target)] +
                self.log_reliabilities_map[(control, middle)]
            )
        else:
            return 4.0 * self.base_gate_reward


class QcpRoutingEnv(SequentialRoutingEnv):
    """
    Environment using circuit representations from `Optimizing quantum circuit placement via machine learning
    <https://dl.acm.org/doi/10.1145/3489517.3530403>`_. The circuit observation consists of a matrix where each row
    corresponds to a physical node. The element at (i, j) corresponds to the qubit targeted by the logical
    qubit currently occupying the i-th node, at time step (layer) j. A value of -1 is used if the qubit is not involved
    in an interaction at that time step.

    :param depth: Number of two-qubit gate layers in the matrix circuit representation.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        coupling_map: rx.PyGraph,
        initial_mapping: NDArray,
        depth: int,
        allow_bridge_gate: bool = True,
        error_rates: Optional[NDArray] = None,
        restrict_swaps_to_front_layer: bool = True,
        base_gate_reward: float = -1.0,
    ):
        super().__init__(circuit, coupling_map, initial_mapping, allow_bridge_gate, error_rates,
                         [CircuitMatrix(depth)], restrict_swaps_to_front_layer, base_gate_reward)


class SchedulingMap(MutableMapping[int, int]):
    _map: Dict[int, int]

    def __init__(self):
        self._map = {}

    def __setitem__(self, k: int, v: int):
        if v == 0:
            if k in self._map:
                del self._map[k]
        else:
            self._map.__setitem__(k, v)

    def __delitem__(self, k: int):
        self._map.__delitem__(k)

    def __getitem__(self, k: int) -> int:
        return self._map.__getitem__(k)

    def __len__(self) -> int:
        return self._map.__len__()

    def __iter__(self) -> Iterator[int]:
        return self._map.__iter__()

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self._map}>'

    def decrement(self):
        for k in list(self._map):
            self[k] -= 1

    def update_all(self, keys: Iterable[int], value: int = 1):
        self.update((k, value) for k in keys)

    def copy(self) -> Self:
        scheduling_map = copy.copy(self)
        scheduling_map._map = self._map.copy()
        return scheduling_map


class LayeredRoutingEnv(RoutingEnv):
    def __init__(
        self,
        circuit: QuantumCircuit,
        coupling_map: rx.PyGraph,
        initial_mapping: NDArray,
        allow_bridge_gate: bool = True,
        error_rates: Optional[NDArray] = None,
        obs_modules: Optional[List['ObsModule']] = None,
        use_decomposed_actions: bool = False,
    ):
        super().__init__(circuit, coupling_map, initial_mapping, allow_bridge_gate, error_rates, obs_modules)

        self.use_decomposed_actions = use_decomposed_actions
        self.scheduling_map = SchedulingMap()

        self.observation_space = spaces.Dict(self._obs_spaces())

        # Account for COMMIT / FINISH action in addition to SWAP and BRIDGE gates
        num_actions = self.num_edges + len(self.bridge_pairs) + 1
        self.action_space = spaces.Discrete(num_actions)

    def step(self, action: int) -> Tuple[RoutingObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        # TODO: figure out how to implement rewards (maybe extract rewards to superclass)
        reward = 0.0

        if action < self.num_edges:
            # SWAP action
            edge = self.edge_list[action]

            self._swap(edge)
            reward = -3.0
        elif action < self.num_edges + len(self.bridge_pairs):
            # BRIDGE action
            pair = self.bridge_pairs[action - self.num_edges]
            op_node, nodes = self._bridge_args(pair)

            self.dag.remove_op_node(op_node)
            self._bridge(*nodes)
            reward = -2.0
        else:
            # COMMIT action
            self._update_state()

        return self.current_obs(), reward, self.terminated, False, {}

    def action_masks(self) -> NDArray:
        mask = np.ones(self.action_space.n, dtype=bool)

        # Swap actions
        invalid_swap_actions = [
            i for i, edge in enumerate(self.edge_list)
            if self.scheduling_map.keys() & edge
        ]

        mask[invalid_swap_actions] = False

        # Bridge actions
        invalid_bridge_actions = []
        for i, pair in enumerate(self.bridge_pairs):
            args = self._bridge_args(pair)
            if args is None or self.scheduling_map.keys() & args[1]:
                invalid_bridge_actions.append(self.num_edges + i)

        mask[invalid_bridge_actions] = False

        # Cannot COMMIT an empty layer
        if not self.scheduling_map:
            mask[-1] = False

        return mask

    def copy(self) -> Self:
        env = super().copy()
        env.scheduling_map = self.scheduling_map.copy()
        return env

    def _bridge(self, control: int, middle: int, target: int):
        super()._bridge(control, middle, target)
        self.scheduling_map.update_all((control, middle, target))

    def _swap(self, edge: Tuple[int, int]):
        super()._swap(edge)
        self.scheduling_map.update_all(edge)

    def _schedule_gates(self, gates: GateSchedulingList):
        super()._schedule_gates(gates)
        self.scheduling_map.update_all(itertools.chain(*(nodes for _, nodes in gates)))

    def _schedulable_gates(self, only_front_layer: bool = False) -> GateSchedulingList:
        gates = super()._schedulable_gates(only_front_layer)
        return [(op_node, nodes) for (op_node, nodes) in gates if not self.scheduling_map.keys() & nodes]

    def _reset_state(self):
        self.scheduling_map.clear()
        super()._reset_state()

    def _update_state(self):
        self.scheduling_map.decrement()

        gates = self._schedulable_gates(only_front_layer=True)
        self._schedule_gates(gates)

        # edge_vector_idx = self.diameter + 1
        #
        # for qubit in range(self.num_qubits):
        #     target = self.qubit_targets[qubit]
        #     if target != -1:
        #         # Distance vector
        #         qubit_node = self.qubit_to_node[qubit]
        #         target_node = self.qubit_to_node[target]
        #
        #         distance = self.distance_matrix[qubit_node, target_node]
        #         obs[distance] += 1
        #
        #         # Edge vector
        #         num_edges = 0
        #         for _, child_node, _ in self.coupling_map.out_edges(qubit):
        #             child_target = self.qubit_targets[target]
        #             if (
        #                 child_target not in {-1, qubit} and
        #                 self.shortest_paths[qubit][child_target][1] == child_node and
        #                 not self.protected_nodes.intersection({qubit, child_node})
        #             ):
        #                 num_edges += 1
        #         obs[edge_vector_idx + num_edges] += 1

    # def _schedule_swaps(self, action: NDArray) -> float:
    #     reward = 0.0
    #
    #     pre_swap_distances = self._qubit_distances()
    #
    #     to_swap = [self.edge_list[i] for i in np.flatnonzero(action)]
    #
    #     for edge in to_swap:
    #         if not self.protected_nodes.intersection(edge):
    #             self._swap(edge)
    #
    #     self._update_state()
    #
    #     post_swap_distances = self._qubit_distances()
    #     reward += np.sum(np.sign(pre_swap_distances - post_swap_distances)) * self.distance_reduction_reward
    #
    #     return reward
    #
    # def _qubit_distances(self) -> NDArray:
    #     distances = np.zeros(self.num_qubits)
    #
    #     for qubit, target in enumerate(self.qubit_targets):
    #         if target != -1:
    #             qubit_node = self.qubit_to_node[qubit]
    #             target_node = self.qubit_to_node[target]
    #
    #             distances[qubit] = self.distance_matrix[qubit_node, target_node]
    #
    #     return distances


RoutingEnvType = TypeVar('RoutingEnvType', bound=RoutingEnv)


def _generate_random_mapping(num_qubits: int) -> NDArray:
    mapping = np.arange(num_qubits)
    np.random.shuffle(mapping)
    return mapping


class TrainingWrapper(gym.Wrapper[RoutingObsType, int, RoutingObsType, int]):
    """
    Wraps a :py:class:`RoutingEnv`, automatically generating circuits and gate error rates at fixed intervals to
    help train deep learning algorithms.

    :param coupling_map: Graph representing the connectivity of the target device.
    :param circuit_generator: Random circuit generator to be used during training.
    :param env_class: :py:class:`RoutingEnv` subclass to wrap.
    :param noise_config: Noise configuration used to generate gate error rates.
    :param training_iters: Number of episodes per generated circuit.

    :ivar iter: Current training iteration.
    """

    env: RoutingEnv

    def __init__(
        self,
        circuit_generator: CircuitGenerator,
        coupling_map: rx.PyGraph,
        env_class: Type[RoutingEnvType],
        *args,
        noise_config: Optional[NoiseConfig] = None,
        training_iters: int = 1,
        **kwargs,
    ):
        self.circuit_generator = circuit_generator
        self.noise_config = noise_config
        self.training_iters = training_iters

        env = env_class(QuantumCircuit(), coupling_map, np.arange(coupling_map.num_nodes()), *args, **kwargs)
        super().__init__(env)

        self.iter = 0

    @property
    def noise_aware(self) -> bool:
        return self.noise_config is not None

    @property
    def num_qubits(self) -> bool:
        return self.env.num_qubits

    @property
    def num_edges(self) -> bool:
        return self.env.num_edges

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[RoutingObsType, Dict[str, Any]]:
        self.env.initial_mapping = _generate_random_mapping(self.num_qubits)

        if self.iter % self.training_iters == 0:
            self.env.circuit = self.circuit_generator.generate()

        if self.noise_aware and self.iter % self.noise_config.recalibration_interval == 0:
            error_rates = self.noise_config.generate_error_rates(self.num_edges)
            self.env.calibrate(error_rates)

        self.iter += 1

        return super().reset(seed=seed, options=options)


class EvaluationWrapper(gym.Wrapper[RoutingObsType, int, RoutingObsType, int]):
    env: RoutingEnv

    def __init__(
        self,
        circuit_generator: CircuitGenerator,
        coupling_map: rx.PyGraph,
        env_class: Type[RoutingEnvType],
        *args,
        noise_config: Optional[NoiseConfig] = None,
        evaluation_iters: int = 25,
        **kwargs,
    ):
        self.circuit_generator = circuit_generator
        self.evaluation_iters = evaluation_iters

        initial_mapping = _generate_random_mapping(coupling_map.num_nodes())
        error_rates = noise_config.generate_error_rates(coupling_map.num_edges()) if noise_config is not None else None

        env = env_class(QuantumCircuit(), coupling_map, initial_mapping, *args, error_rates=error_rates, **kwargs)
        super().__init__(env)

        self.iter = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[RoutingObsType, Dict[str, Any]]:
        if self.iter % self.evaluation_iters == 0:
            self.env.circuit = self.circuit_generator.generate()

        self.iter += 1

        return super().reset(seed=seed, options=options)


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
        # TODO: min_log_reliability
        return spaces.Box(-100.0, 0.0, shape=env.log_reliabilities.shape)

    def obs(self, env: RoutingEnv) -> NDArray:
        return env.log_reliabilities.copy()


class CircuitMatrix(ObsModule[SequentialRoutingEnv]):
    def __init__(self, depth: int):
        if depth <= 0:
            raise ValueError(f'Depth must be positive, got {depth}')

        self.depth = depth

    @staticmethod
    def key() -> str:
        return 'circuit'

    def space(self, env: SequentialRoutingEnv) -> spaces.Box:
        return spaces.Box(-1, env.num_qubits - 1, (env.num_qubits, self.depth), dtype=np.int32)

    def obs(self, env: SequentialRoutingEnv) -> NDArray:
        space = self.space(env)
        circuit = np.full(space.shape, -1, dtype=space.dtype)

        layer_idx = 0
        layer_qubits = set()

        for op_node in env.dag.two_qubit_ops():
            indices = qubits_to_indices(env.circuit, op_node.qargs)

            if layer_qubits.intersection(indices):
                layer_qubits = set(indices)
                layer_idx += 1

                if layer_idx == circuit.shape[1]:
                    break
            else:
                layer_qubits.update(indices)

            idx_a, idx_b = indices

            circuit[idx_a, layer_idx] = env.qubit_to_node[idx_b]
            circuit[idx_b, layer_idx] = env.qubit_to_node[idx_a]

        mapped_circuit = np.zeros_like(circuit)
        for q in range(env.num_qubits):
            mapped_circuit[q] = circuit[env.node_to_qubit[q]]

        return mapped_circuit


class QubitInteractions(ObsModule[LayeredRoutingEnv]):
    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth

    @staticmethod
    def key() -> str:
        return 'qubit_interactions'

    def space(self, env: LayeredRoutingEnv) -> spaces.Box:
        return spaces.Box(-1, self.max_depth, shape=(env.num_qubits * (env.num_qubits - 1) // 2,), dtype=np.int32)

    def obs(self, env: LayeredRoutingEnv) -> NDArray:
        qubit_interactions = OrderedDict()
        for i in range(env.num_qubits):
            for j in range(i):
                qubit_interactions[(j, i)] = -1

        for i, layer in enumerate(dag_layers(env.dag)[:self.max_depth]):
            for op_node in layer:
                if len(op_node.qargs) == 2:
                    indices = tuple(sorted(qubits_to_indices(env.circuit, op_node.qargs)))

                    if qubit_interactions[indices] == -1:
                        qubit_interactions[indices] = i

        return np.array(list(qubit_interactions.values()))


class LockedEdges(ObsModule[LayeredRoutingEnv]):
    @staticmethod
    def key() -> str:
        return 'locked_edges'

    def space(self, env: LayeredRoutingEnv) -> spaces.Box:
        return spaces.Box(0, 4 if env.use_decomposed_actions else 1, shape=(env.num_edges,), dtype=np.int32)

    def obs(self, env: LayeredRoutingEnv) -> NDArray:
        return np.array([max(env.scheduling_map.get(q, 0) for q in edge) for edge in env.edge_list])
