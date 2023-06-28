
import itertools
from abc import ABC, abstractmethod
from collections.abc import MutableMapping, Iterable
from dataclasses import dataclass, field
from math import e
from typing import Optional, Tuple, Dict, Any, List, SupportsFloat, Iterator

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
    min_log_reliability: float = field(default=-100, kw_only=True)
    log_base: float = field(default=e, kw_only=True)

    def generate_error_rates(self, n: int) -> NDArray:
        return np.random.normal(self.mean, self.std, n).clip(0.0, 1.0)

    def calculate_log_reliabilities(self, error_rates: NDArray) -> NDArray:
        if np.any((error_rates < 0.0) | (error_rates > 1.0)):
            raise ValueError('Got invalid values for error rates')

        return np.where(
            error_rates < 1.0,
            np.emath.logn(self.log_base, 1.0 - error_rates),
            -np.inf
        ).clip(self.min_log_reliability)


RoutingObsType = Dict[str, NDArray]
GateSchedulingList = List[Tuple[DAGOpNode, Tuple[int, ...]]]
BridgeArgs = Tuple[DAGOpNode, Tuple[int, int, int]]


class RoutingEnv(gym.Env[RoutingObsType, int], ABC):
    """
    Base qubit routing environment.

    :param coupling_map: Graph representing the connectivity of the target device.
    :param circuit_generator: Random circuit generator to be used during training.
    :param initial_mapping: Initial mapping from physical nodes to logical qubits. If ``None``, a random initial mapping
                            will be used for each training iteration.
    :param allow_bridge_gate: Allow the use of BRIDGE gates when routing.
    :param training: ``True`` if the environment is being used for training.
    :param training_iterations: Number of episodes per generated circuit.

    :ivar node_to_qubit: Current mapping from physical nodes to logical qubits.
    :ivar qubit_to_node: Current mapping from logical qubits to physical nodes.
    :ivar routed_dag: ``DAGCircuit`` containing already routed gates.
    :ivar error_rates: Array of two-qubit gate error rates.
    :ivar log_reliabilities: Array of logarithms of two-qubit gate reliabilities (``reliability = 1 - error_rate``)
    :ivar iter: Current training iteration.
    """

    observation_space: spaces.Dict
    action_space: spaces.Discrete

    initial_mapping: Optional[NDArray]
    error_rates: NDArray
    log_reliabilities: NDArray
    log_reliabilities_map: Dict[Tuple[int, int], float]

    def __init__(
        self,
        coupling_map: rx.PyGraph,
        circuit_generator: CircuitGenerator,
        initial_mapping: Optional[NDArray] = None,
        allow_bridge_gate: bool = True,
        noise_config: Optional[NoiseConfig] = None,
        training: bool = True,
        training_iterations: int = 1,
    ):
        if initial_mapping is not None and initial_mapping.shape != (coupling_map.num_nodes(),):
            raise ValueError('Initial mapping has invalid shape for the provided coupling map')

        if training_iterations < 1:
            raise ValueError(f'Number of training iterations must be greater than zero, got {training_iterations}')

        self.coupling_map = coupling_map
        self.circuit_generator = circuit_generator
        self.initial_mapping = initial_mapping
        self.allow_bridge_gate = allow_bridge_gate
        self.noise_config = noise_config
        self.training = training
        self.training_iterations = training_iterations

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
        self.node_to_qubit = np.zeros(self.num_qubits, dtype=np.int32)
        self.qubit_to_node = np.zeros(self.num_qubits, dtype=np.int32)

        self.circuit = QuantumCircuit(self.num_qubits)
        self.dag = circuit_to_dag(self.circuit)
        self.routed_dag = self.dag.copy_empty_like()

        # Noise-awareness information
        if self.noise_aware:
            self.calibrate(np.zeros(self.num_edges))

        bridge_circuit = QuantumCircuit(3)
        for _ in range(2):
            bridge_circuit.cx(1, 2)
            bridge_circuit.cx(0, 1)

        self.bridge_gate = bridge_circuit.to_gate(label='BRIDGE')
        self.bridge_gate.name = 'bridge'
        self.swap_gate = SwapGate()

        self.iter = 0

    @property
    def noise_aware(self) -> bool:
        return self.noise_config is not None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[RoutingObsType, Dict[str, Any]]:
        if self.training:
            if self.noise_aware and self.iter % self.noise_config.recalibration_interval == 0:
                self.recalibrate()

            if self.iter % self.training_iterations == 0:
                self.generate_circuit()

            self.iter += 1

        self._reset_dag()

        return self._current_obs(), {}

    def generate_circuit(self):
        self.circuit = self.circuit_generator.generate()

    def routed_circuit(self) -> QuantumCircuit:
        return dag_to_circuit(self.routed_dag)

    def calibrate(self, error_rates: NDArray):
        """
        Given an array of edge error rates, calculates the log reliabilities according to the noise
        configuration, which can, in turn, be used to obtain agent rewards.

        :param error_rates: Array of two-qubit gate error rates.
        """
        self.error_rates = error_rates.copy()
        self.log_reliabilities = self.noise_config.calculate_log_reliabilities(error_rates)

        m = {}
        for edge, value in zip(self.edge_list, self.log_reliabilities):  # type: ignore
            m[edge] = value
            m[edge[::-1]] = value

        self.log_reliabilities_map = m

    def recalibrate(self):
        """
        Performs calibration with random error rates determined by the noise configuration.
        """
        self.calibrate(self.noise_config.generate_error_rates(self.num_edges))

    def set_random_initial_mapping(self):
        """
        Generates and sets a random permutation for the initial mapping. Afterward, all circuits will be
        routed using this mapping.
        """
        self.initial_mapping = self._generate_random_mapping()

    @abstractmethod
    def action_masks(self) -> NDArray:
        raise NotImplementedError

    @abstractmethod
    def _update_state(self):
        raise NotImplementedError

    def _obs_spaces(self) -> Dict[str, spaces.Space]:
        obs_spaces = {}

        if self.noise_aware:
            obs_spaces['log_reliabilities'] = spaces.Box(
                self.noise_config.min_log_reliability,
                0.0,
                shape=self.log_reliabilities.shape,
            )

        return obs_spaces

    def _current_obs(self) -> RoutingObsType:
        obs = {}

        if self.noise_aware:
            obs['log_reliabilities'] = self.log_reliabilities.copy()

        return obs

    @property
    def terminated(self) -> bool:
        return self.dag.size() == 0

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

    def _swap(self, edge: Tuple[int, int]):
        qargs = indices_to_qubits(self.circuit, edge)

        self.routed_dag.apply_operation_back(self.swap_gate, qargs)

        # Update mappings
        node_a, node_b = edge
        qubit_a, qubit_b = (self.node_to_qubit[n] for n in edge)

        self.node_to_qubit[node_a], self.node_to_qubit[node_b] = qubit_b, qubit_a
        self.qubit_to_node[qubit_a], self.qubit_to_node[qubit_b] = node_b, node_a

    def _generate_random_mapping(self) -> NDArray:
        mapping = np.arange(self.num_qubits)
        np.random.shuffle(mapping)
        return mapping

    def _reset_dag(self):
        self.dag = circuit_to_dag(self.circuit)
        self.routed_dag = self.dag.copy_empty_like()

        self._reset_state()

    def _reset_state(self):
        self.node_to_qubit = (
            self._generate_random_mapping()
            if self.initial_mapping is None
            else self.initial_mapping.copy()
        )

        for node, qubit in enumerate(self.node_to_qubit):
            self.qubit_to_node[qubit] = node

        self._update_state()

    def _bridge_args(self, pair: Tuple[int, int]) -> Optional[BridgeArgs]:
        for op_node in self.dag.front_layer():
            if isinstance(op_node.op, CXGate):
                indices = qubits_to_indices(self.circuit, op_node.qargs)
                nodes = tuple(self.qubit_to_node[i] for i in indices)

                if tuple(sorted(nodes)) == pair:
                    control, target = nodes
                    return op_node, tuple(self.shortest_paths[control][target])

        return None


class SequentialRoutingEnv(RoutingEnv, ABC):
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
        coupling_map: rx.PyGraph,
        circuit_generator: CircuitGenerator,
        initial_mapping: Optional[NDArray] = None,
        allow_bridge_gate: bool = True,
        noise_config: Optional[NoiseConfig] = None,
        training: bool = True,
        training_iterations: int = 1,
        restrict_swaps_to_front_layer: bool = True,
        base_gate_reward: float = -1.0,
    ):
        super().__init__(coupling_map, circuit_generator, initial_mapping, allow_bridge_gate, noise_config, training,
                         training_iterations)

        self.restrict_swaps_to_front_layer = restrict_swaps_to_front_layer
        self.base_gate_reward = base_gate_reward

        self._blocked_swap = None
        self._scheduling_reward = 0.0

        num_actions = self.num_edges + len(self.bridge_pairs)
        self.action_space = spaces.Discrete(num_actions)

    def step(self, action: int) -> Tuple[RoutingObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if self.terminated:
            # Environment terminated immediately and does not require routing
            return self._current_obs(), 0.0, True, False, {}

        is_swap = action < self.num_edges

        if is_swap:
            # SWAP action
            edge = self.edge_list[action]
            self._swap(edge)
            reward = self._swap_reward(edge)
        else:
            # BRIDGE action
            pair = self.bridge_pairs[action - self.num_edges]

            try:
                op_node, nodes = self._bridge_args(pair)
            except TypeError:
                # TODO: remove
                print(self._current_obs())
                print(dag_to_circuit(self.dag))
                print(self.qubit_to_node, self.node_to_qubit)
                print(action, self.action_masks())
                raise

            self.dag.remove_op_node(op_node)
            self._bridge(*nodes)
            reward = self._bridge_reward(*nodes)

        self._update_state()
        # reward += self._scheduling_reward

        # self._blocked_swap = action if is_swap and self._scheduling_reward == 0.0 else None

        return self._current_obs(), reward, self.terminated, False, {}

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
        coupling_map: rx.PyGraph,
        circuit_generator: CircuitGenerator,
        depth: int,
        initial_mapping: Optional[NDArray] = None,
        allow_bridge_gate: bool = True,
        noise_config: Optional[NoiseConfig] = None,
        training: bool = True,
        training_iterations: int = 1,
        restrict_swaps_to_front_layer: bool = True,
        base_gate_reward: float = -1.0,
    ):
        super().__init__(coupling_map, circuit_generator, initial_mapping, allow_bridge_gate, noise_config, training,
                         training_iterations, restrict_swaps_to_front_layer, base_gate_reward)

        if depth <= 0:
            raise ValueError(f'Depth must be positive, got {depth}')

        self.depth = depth

        self.observation_space = spaces.Dict(self._obs_spaces())

    def _obs_spaces(self) -> Dict[str, spaces.Space]:
        obs_spaces = super()._obs_spaces()
        obs_spaces['circuit'] = spaces.Box(-1, self.num_qubits - 1, (self.num_qubits, self.depth), dtype=np.int32)
        return obs_spaces

    def _current_obs(self) -> RoutingObsType:
        obs = super()._current_obs()

        obs_space_circuit = self.observation_space.spaces['circuit']
        circuit = np.full(obs_space_circuit.shape, -1, dtype=obs_space_circuit.dtype)

        layer_idx = 0
        layer_qubits = set()

        for op_node in self.dag.two_qubit_ops():
            indices = qubits_to_indices(self.circuit, op_node.qargs)

            if layer_qubits.intersection(indices):
                layer_qubits = set(indices)
                layer_idx += 1

                if layer_idx == circuit.shape[1]:
                    break
            else:
                layer_qubits.update(indices)

            idx_a, idx_b = indices

            circuit[idx_a, layer_idx] = self.qubit_to_node[idx_b]
            circuit[idx_b, layer_idx] = self.qubit_to_node[idx_a]

        mapped_circuit = np.zeros_like(circuit)
        for q in range(self.num_qubits):
            mapped_circuit[q] = circuit[self.node_to_qubit[q]]

        obs['circuit'] = mapped_circuit

        return obs


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


class LayeredRoutingEnv(RoutingEnv):
    def __init__(
        self,
        coupling_map: rx.PyGraph,
        circuit_generator: CircuitGenerator,
        initial_mapping: Optional[NDArray] = None,
        allow_bridge_gate: bool = True,
        noise_config: Optional[NoiseConfig] = None,
        training: bool = True,
        training_iterations: int = 1,
        use_decomposed_actions: bool = False,
    ):
        super().__init__(coupling_map, circuit_generator, initial_mapping, allow_bridge_gate,
                         noise_config, training, training_iterations)

        self.use_decomposed_actions = use_decomposed_actions
        self.scheduling_map = SchedulingMap()

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
            reward = -3.0
        else:
            # COMMIT action
            self._update_state()

        return self._current_obs(), reward, self.terminated, False, {}

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
        self._schedule_gates(self._schedulable_gates(only_front_layer=True))


class QubitInteractionsRoutingEnv(LayeredRoutingEnv):
    def __init__(
        self,
        coupling_map: rx.PyGraph,
        circuit_generator: CircuitGenerator,
        initial_mapping: Optional[NDArray] = None,
        allow_bridge_gate: bool = True,
        noise_config: Optional[NoiseConfig] = None,
        training: bool = True,
        training_iterations: int = 1,
        use_decomposed_actions: bool = False,
        max_interaction_depth: int = 8,
    ):
        super().__init__(coupling_map, circuit_generator, initial_mapping, allow_bridge_gate, noise_config, training,
                         training_iterations, use_decomposed_actions)

        self.max_interaction_depth = max_interaction_depth

        self.qubit_interactions = {
            (j, i): -1 for i in range(self.num_qubits) for j in range(i)
        }

        self.observation_space = spaces.Dict(self._obs_spaces())

    def _obs_spaces(self) -> Dict[str, spaces.Space]:
        obs_spaces = super()._obs_spaces()

        obs_spaces['locked_nodes'] = spaces.Box(0, 4 if self.use_decomposed_actions else 1, shape=(self.num_qubits,),
                                                dtype=np.int32)
        obs_spaces['qubit_interactions'] = spaces.Box(-1, 1000, shape=(len(self.qubit_interactions),), dtype=np.int32)

        return obs_spaces

    def _current_obs(self) -> RoutingObsType:
        obs = super()._current_obs()

        obs['locked_nodes'] = np.array([self.scheduling_map.get(q, 0) for q in range(self.num_qubits)])
        obs['qubit_interactions'] = np.array(list(self.qubit_interactions.values()))

        return obs

    def _update_state(self):
        super()._update_state()

        self.qubit_interactions.update((k, -1) for k in self.qubit_interactions)

        for i, layer in enumerate(dag_layers(self.dag)[:self.max_interaction_depth]):
            for op_node in layer:
                if len(op_node.qargs) == 2:
                    indices = tuple(sorted(qubits_to_indices(self.circuit, op_node.qargs)))

                    if self.qubit_interactions[indices] == -1:
                        self.qubit_interactions[indices] = i

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
