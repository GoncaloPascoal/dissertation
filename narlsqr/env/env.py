
import copy
import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import e
from typing import Any, Literal, Optional, Self, SupportsFloat, TypeAlias

import gymnasium as gym
import numpy as np
import rustworkx as rx
from gymnasium import spaces
from nptyping import Int8, NDArray
from ordered_set import OrderedSet
from qiskit import QuantumCircuit
from qiskit.circuit import Operation, Qubit
from qiskit.circuit.library import CXGate, SwapGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import AnalysisPass, TranspilerError
from qiskit.transpiler.passes import CommutationAnalysis

from narlsqr.utils import dag_layers, indices_to_qubits, qubits_to_indices

RoutingObs: TypeAlias = dict[str, NDArray]
GateSchedulingList: TypeAlias = list[tuple[DAGOpNode, tuple[int, ...]]]
BridgeArgs: TypeAlias = tuple[DAGOpNode, tuple[int, int, int]]


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """
    Configuration values for controlling rewards in noise-aware routing environments.

    :ivar log_base: Base used to calculate log reliabilities from gate error rates.
    :ivar min_log_reliability: Greatest penalty that can be issued from scheduling a gate, to prevent infinite rewards.
        Cannot be positive.
    :ivar added_gate_reward: Flat value that will be added to the reward associated with each two-qubit gate from the
        original circuit. Cannot be negative.
    """
    log_base: float = field(default=e, kw_only=True)
    min_log_reliability: float = field(default=-100.0, kw_only=True)
    added_gate_reward: float = field(default=0.01, kw_only=True)

    def __post_init__(self):
        if self.log_base <= 1.0:
            raise ValueError(f'Logarithm base must be greater than 1, got {self.log_base}')
        if self.min_log_reliability > 0.0:
            raise ValueError(f'Minimum log reliability cannot be positive, got {self.min_log_reliability}')
        if self.added_gate_reward < 0.0:
            raise ValueError(f'Added gate reward cannot be negative, got {self.added_gate_reward}')

    def calculate_log_reliabilities(self, error_rates: NDArray) -> NDArray:
        if np.any((error_rates < 0.0) | (error_rates > 1.0)):
            raise ValueError('Got invalid values for error rates')

        return np.where(
            error_rates < 1.0,
            np.emath.logn(self.log_base, 1.0 - error_rates),
            -np.inf
        ).clip(self.min_log_reliability)


class RoutingEnv(gym.Env[RoutingObs, int], ABC):
    """
    Qubit routing environment.

    :param coupling_map: Graph representing the connectivity of the target device.
    :param circuit: Quantum circuit to compile. This argument is optional during environment construction but a valid
        circuit must be assigned before the :py:meth:`reset` method is called.
    :param initial_mapping: Initial mapping from physical nodes to logical qubits. If ``None``, a random initial mapping
        will be used for each training iteration.
    :param layout_pass: Analysis pass used to select the initial mapping during evaluation.
    :param allow_bridge_gate: Allow the use of BRIDGE gates when routing.
    :param commutation_analysis: Use commutation rules to schedule additional gates at each time step.
    :param restrict_swaps_to_front_layer: Only allow SWAP operations that involve qubits from the first layer of gates.
    :param force_swap_distance_reduction: Only allow SWAP operations that reduce the total distance between interacting
        qubits in the first layer of gates. Only active if :py:attr:`restrict_swaps_to_front_layer` is also True.
    :param error_rates: Array of two-qubit gate error rates.
    :param noise_aware: Whether the environment should be noise-aware.
    :param noise_config: Configuration for rewards calculated from log reliabilities.
    :param obs_modules: Observation modules that define the key-value pairs in observations.
    :param log_metrics: Log additional metrics when training, such as the number of SWAP and BRIDGE gates used in each
        episode.
    :param name: Name of the routing environment. Useful for organizing training logs.

    :ivar node_to_qubit: Current mapping from physical nodes to logical qubits.
    :ivar qubit_to_node: Current mapping from logical qubits to physical nodes.
    :ivar routed_gates: List of already routed gates, including additional SWAP and BRIDGE operations.
    :ivar log_reliabilities: Array of logarithms of two-qubit gate reliabilities (``reliability = 1 - error_rate``)
    :ivar added_gate_reward: Reward for scheduling a two-qubit gate from the original circuit. Used only in
        noise-aware environments.
    :ivar edge_to_log_reliability: Mapping of edges to their corresponding log reliability values.
    :ivar edge_to_reliability: Mapping of edges to their corresponding reliability values.
    :ivar pair_to_bridge_args: Mapping of qubit pairs to arguments required to perform a BRIDGE operation.
    """

    observation_space: spaces.Dict
    action_space: spaces.Discrete

    initial_mapping: NDArray
    error_rates: NDArray
    obs_modules: list['ObsModule']
    edge_list: list[tuple[int, int]]

    node_to_qubit: NDArray
    qubit_to_node: NDArray

    routed_gates: list[tuple[Operation, tuple[int, ...]]]
    routed_op_nodes: set[DAGOpNode]

    log_reliabilities: NDArray
    added_gate_reward: float
    edge_to_log_reliability: dict[tuple[int, int], float]
    edge_to_reliability: dict[tuple[int, int], float]

    pair_to_bridge_args: dict[tuple[int, int], BridgeArgs]

    _scheduled_2q_gates: list[tuple[int, int]]
    _blocked_swaps: set[int]

    def __init__(
        self,
        coupling_map: rx.PyGraph,
        circuit: Optional[QuantumCircuit] = None,
        initial_mapping: Optional[Iterable[int]] = None,
        layout_pass: Optional[AnalysisPass] = None,
        allow_bridge_gate: bool = True,
        commutation_analysis: bool = True,
        restrict_swaps_to_front_layer: bool = True,
        force_swap_distance_reduction: bool = False,
        error_rates: Optional[Iterable[float]] = None,
        noise_aware: bool = True,
        noise_config: Optional[NoiseConfig] = None,
        obs_modules: Optional[list['ObsModule']] = None,
        log_metrics: bool = False,
        name: str = 'routing_env',
    ):
        num_qubits = coupling_map.num_nodes()
        num_edges = coupling_map.num_edges()

        if circuit is None:
            circuit = QuantumCircuit(num_qubits)

        initial_mapping = np.arange(num_qubits) if initial_mapping is None else np.array(initial_mapping, copy=False)
        error_rates = np.zeros(num_edges) if error_rates is None else np.array(error_rates, copy=False)

        if initial_mapping.shape != (num_qubits,):
            raise ValueError('Initial mapping has invalid shape for the provided coupling map')
        if error_rates.shape != (num_edges,):
            raise ValueError('Error rates have invalid shape for the provided coupling map')

        self.coupling_map = coupling_map
        self.initial_mapping = initial_mapping
        self.layout_pass = layout_pass
        self.allow_bridge_gate = allow_bridge_gate
        self.commutation_analysis = commutation_analysis
        self.restrict_swaps_to_front_layer = restrict_swaps_to_front_layer
        self.force_swap_distance_reduction = force_swap_distance_reduction
        self.error_rates = error_rates
        self.noise_aware = noise_aware
        self.noise_config = NoiseConfig() if noise_config is None else noise_config
        self.obs_modules = [] if obs_modules is None else obs_modules
        self.log_metrics = log_metrics
        self.name = name

        # Computations using coupling map
        self.num_qubits = num_qubits
        self.num_edges = num_edges
        self.edge_list = list(coupling_map.edge_list())  # type: ignore

        self.shortest_paths = rx.graph_all_pairs_dijkstra_shortest_paths(coupling_map, lambda _: 1.0)
        self.distance_matrix = rx.graph_distance_matrix(coupling_map).astype(np.int32)

        if self.allow_bridge_gate:
            self.bridge_pairs = sorted(
                (j, i) for i in range(self.num_qubits) for j in range(i) if self.distance_matrix[j, i] == 2
            )
        else:
            self.bridge_pairs = []

        self.commutation_pass = CommutationAnalysis()
        self.circuit = circuit

        # State information
        self.node_to_qubit = initial_mapping.copy()
        self.qubit_to_node = np.zeros_like(self.node_to_qubit)
        for node, qubit in enumerate(self.node_to_qubit):
            self.qubit_to_node[qubit] = node

        self.dag = circuit_to_dag(self.circuit)
        self.routed_gates = []
        self.routed_op_nodes = set()

        # Noise-related setup
        if self.noise_aware:
            self.obs_modules.append(LogReliabilities())
        self.calibrate(self.error_rates)

        bridge_circuit = QuantumCircuit(3)
        for _ in range(2):
            bridge_circuit.cx(1, 2)
            bridge_circuit.cx(0, 1)

        self.bridge_gate = bridge_circuit.to_gate(label='bridge')
        self.bridge_gate.name = 'bridge'
        self.swap_gate = SwapGate()

        self.pair_to_bridge_args = {}

        self._scheduled_2q_gates = []
        self._scheduling_reward = 0.0

        self._blocked_swaps = set()

        # Action / observation space
        num_actions = self.num_edges + len(self.bridge_pairs)
        self.action_space = spaces.Discrete(num_actions)

        self.observation_space = spaces.Dict(self._obs_spaces())

        # Custom metrics
        self.metrics = defaultdict(float)

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @circuit.setter
    def circuit(self, value: QuantumCircuit):
        self._circuit = value

        if self.layout_pass is not None:
            try:
                self.layout_pass.run(circuit_to_dag(self.circuit))

                layout = self.layout_pass.property_set['layout'].get_physical_bits()
                qargs = tuple(layout[i] for i in range(self.num_qubits))
                self.initial_mapping = np.array(qubits_to_indices(self.circuit, qargs))
            except TranspilerError:
                print('Exception occurred during layout pass, defaulting to trivial initial mapping')
                self.initial_mapping = np.arange(self.num_qubits)

    @property
    def terminated(self) -> bool:
        return self.dag.size() == 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[RoutingObs, dict[str, Any]]:
        if self.log_metrics:
            self.metrics.clear()
            self.metrics['reliability'] = 1.0

        self.dag = circuit_to_dag(self.circuit)
        self.routed_gates = []
        self.routed_op_nodes = set()

        if self.commutation_analysis:
            self.commutation_pass.run(self.dag)

        self.node_to_qubit = self.initial_mapping.copy()
        for node, qubit in enumerate(self.node_to_qubit):
            self.qubit_to_node[qubit] = node

        self._blocked_swaps.clear()
        self._schedule_gates()

        return self.current_obs(), {}

    def step(self, action: int) -> tuple[RoutingObs, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.terminated:
            # Environment terminated immediately and does not require routing
            return self.current_obs(), 0.0, True, False, {}

        if action < self.num_edges:
            # SWAP action
            edge = self.edge_list[action]
            self._swap(edge)
            reward = self._swap_reward(edge)

            self._remove_blocked_swaps(set(edge))
            self._blocked_swaps.add(action)
        else:
            # BRIDGE action
            pair = self.bridge_pairs[action - self.num_edges]
            args = self.pair_to_bridge_args.get(pair)

            if args is not None:
                op_node, nodes = args

                self._remove_op_node(op_node)
                self._bridge(*nodes)
                reward = self._bridge_reward(*nodes)

                self._remove_blocked_swaps({nodes[0], nodes[2]})
            else:
                print('Invalid BRIDGE action was selected')
                reward = 0.0

        self._schedule_gates()
        reward += self._scheduling_reward
        self._remove_blocked_swaps(set(itertools.chain(*self._scheduled_2q_gates)))

        return self.current_obs(), reward, self.terminated, False, {}

    def action_mask(self) -> NDArray[Literal['*'], Int8]:
        mask = np.ones(self.action_space.n, dtype=Int8)
        front_layer = self.dag.front_layer()

        # Swap actions
        if self.restrict_swaps_to_front_layer:
            front_layer_nodes = set()
            front_layer_pairs = []

            for op_node in front_layer:
                indices = qubits_to_indices(self.circuit, op_node.qargs)
                nodes = tuple(self.qubit_to_node[i] for i in indices)

                front_layer_nodes.update(nodes)
                front_layer_pairs.append(nodes)

            sum_distances_before = sum(self.distance_matrix[node_a][node_b] for node_a, node_b in front_layer_pairs)

            for i, edge in enumerate(self.edge_list):
                if front_layer_nodes.intersection(edge):
                    if self.force_swap_distance_reduction:
                        after_map = dict(zip(edge, edge[::-1]))
                        sum_distances_after = sum(
                            self.distance_matrix[after_map.get(node_a, node_a)][after_map.get(node_b, node_b)]
                            for node_a, node_b in front_layer_pairs
                        )
                        mask[i] = sum_distances_after <= sum_distances_before
                else:
                    mask[i] = 0

        if self.allow_bridge_gate:
            # Compute bridge args
            pair_to_bridge_args = {}

            for op_node in front_layer:
                if isinstance(op_node.op, CXGate):
                    indices = qubits_to_indices(self.circuit, op_node.qargs)
                    nodes = tuple(self.qubit_to_node[i] for i in indices)

                    control, target = nodes
                    shortest_path = self.shortest_paths[control][target]

                    if len(shortest_path) == 3:
                        pair = tuple(sorted(nodes))
                        pair_to_bridge_args[pair] = (op_node, tuple(shortest_path))

            self.pair_to_bridge_args = pair_to_bridge_args    # type: ignore

            # Bridge actions
            invalid_bridge_actions = [
                self.num_edges + i for i, pair in enumerate(self.bridge_pairs)
                if pair not in self.pair_to_bridge_args
            ]
            mask[invalid_bridge_actions] = 0

        # Disallow redundant consecutive SWAPs, as long as there is still a
        # valid action
        new_mask = mask.copy()
        new_mask[list(self._blocked_swaps)] = 0
        if new_mask.any():
            mask = new_mask

        return mask

    def calibrate(self, error_rates: NDArray):
        """
        Given an array of edge error rates, calculates the log reliabilities according to the noise
        configuration, which can, in turn, be used to obtain agent rewards.

        :param error_rates: Array of two-qubit gate error rates.
        """
        self.error_rates = error_rates.copy()

        self.log_reliabilities = self.noise_config.calculate_log_reliabilities(error_rates)
        self.added_gate_reward = abs(np.min(self.log_reliabilities)) + self.noise_config.added_gate_reward

        self.edge_to_reliability, self.edge_to_log_reliability = {}, {}
        for edge, reliability, log_reliability in zip(self.edge_list, 1.0 - error_rates, self.log_reliabilities):
            self.edge_to_reliability[edge] = reliability
            self.edge_to_reliability[edge[::-1]] = reliability

            self.edge_to_log_reliability[edge] = log_reliability
            self.edge_to_log_reliability[edge[::-1]] = log_reliability

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

        env.routed_gates = self.routed_gates.copy()

        return env

    def current_obs(self) -> RoutingObs:
        return {
            'action_mask': self.action_mask(),
            'true_obs': {module.key(): module.obs(self) for module in self.obs_modules},
        }

    def routed_circuit(self) -> QuantumCircuit:
        routed_dag = self.dag.copy_empty_like()
        for op, nodes in self.routed_gates:
            routed_dag.apply_operation_back(op, indices_to_qubits(self.circuit, nodes))
        return dag_to_circuit(routed_dag)

    def _obs_spaces(self) -> dict[str, spaces.Space]:
        return {
            'action_mask': spaces.Box(0, 1, shape=(self.action_space.n,), dtype=np.int8),
            'true_obs': spaces.Dict({module.key(): module.space(self) for module in self.obs_modules}),
        }

    def _remove_blocked_swaps(self, qubits: set[int]):
        to_remove = {i for i in self._blocked_swaps if qubits.intersection(self.edge_list[i])}
        self._blocked_swaps.difference_update(to_remove)

    def _remove_op_node(self, op_node: DAGOpNode):
        self.dag.remove_op_node(op_node)
        self.routed_op_nodes.add(op_node)

    def _schedule_gate(self, op_node: DAGOpNode, nodes: tuple[int, ...]):
        self.routed_gates.append((op_node.op, nodes))
        self._remove_op_node(op_node)

        if len(nodes) == 2:
            self._scheduled_2q_gates.append(nodes)  # type: ignore
            self._scheduling_reward += self._gate_reward(nodes)  # type: ignore

            if self.log_metrics:
                self.metrics['reliability'] *= self.edge_to_reliability[nodes]  # type: ignore

    def _commuting_op_nodes(self, op_node: DAGOpNode, qubit: Qubit) -> OrderedSet[DAGOpNode]:
        commutation_info = self.commutation_pass.property_set['commutation_set']
        idx = commutation_info[(op_node, qubit)]
        return OrderedSet(commutation_info[qubit][idx])

    def _schedule_gates(self, only_front_layer: bool = False):
        self._scheduled_2q_gates = []
        self._scheduling_reward = 0.0

        op_nodes = self.dag.front_layer() if only_front_layer else self.dag.op_nodes()
        locked_nodes: set[int] = set()
        to_schedule = set()

        for op_node in op_nodes:
            qargs = op_node.qargs
            indices = qubits_to_indices(self.circuit, qargs)
            nodes = tuple(self.qubit_to_node[i] for i in indices)

            if op_node in to_schedule:
                self._schedule_gate(op_node, nodes)
            elif len(locked_nodes) < self.num_qubits:
                if self._is_schedulable(nodes, locked_nodes):
                    self._schedule_gate(op_node, nodes)
                else:
                    if self.commutation_analysis:
                        commutation_sets: dict[Qubit, OrderedSet[DAGOpNode]] = {
                            q: self._commuting_op_nodes(op_node, q)
                            for q in qargs
                            if self.circuit.find_bit(q)[0] not in locked_nodes
                        }

                        all_commuting_nodes: set[DAGOpNode] = set().union(*commutation_sets.values())

                        for qubit, commuting_nodes in commutation_sets.items():
                            commuting_nodes.difference_update({op_node}, self.routed_op_nodes)

                            for commuting_node in commuting_nodes:
                                commutes = True
                                cmt_qargs = commuting_node.qargs

                                if len(cmt_qargs) == 2:
                                    other_qubit = cmt_qargs[0] if cmt_qargs[1] == qubit else cmt_qargs[1]
                                    other_commuting_nodes = self._commuting_op_nodes(commuting_node, other_qubit)

                                    for wire_node in self.dag.nodes_on_wire(other_qubit, only_ops=True):
                                        if wire_node == commuting_node:
                                            break

                                        # If the current node in this wire does not share qubits with the original
                                        # op_node, it must commute with the commuting_op_node. Otherwise, it must
                                        # commute with the original op_node.

                                        must_commute_with = (
                                            all_commuting_nodes if set(wire_node.qargs).intersection(qargs)
                                            else other_commuting_nodes
                                        )

                                        if wire_node not in must_commute_with:
                                            commutes = False
                                            break

                                if commutes:
                                    cmt_indices = qubits_to_indices(self.circuit, cmt_qargs)
                                    cmt_nodes = tuple(self.qubit_to_node[i] for i in cmt_indices)

                                    if self._is_schedulable(cmt_nodes, locked_nodes):
                                        to_schedule.add(commuting_node)

                    locked_nodes.update(nodes)

    def _is_schedulable(self, nodes: tuple[int, ...], locked_nodes: set[int]) -> bool:
        valid_under_current_mapping = len(nodes) != 2 or self.coupling_map.has_edge(nodes[0], nodes[1])
        not_blocked = not locked_nodes.intersection(nodes)

        return valid_under_current_mapping and not_blocked

    def _bridge(self, control: int, middle: int, target: int):
        if self.log_metrics:
            self.metrics['added_cnot_count'] += 3
            self.metrics['bridge_count'] += 1
            self.metrics['reliability'] *= (
                self.edge_to_reliability[(middle, target)] * self.edge_to_reliability[(control, middle)]
            ) ** 2

        self.routed_gates.append((self.bridge_gate, (control, middle, target)))

    def _swap(self, edge: tuple[int, int]):
        if self.log_metrics:
            self.metrics['added_cnot_count'] += 3
            self.metrics['swap_count'] += 1
            self.metrics['reliability'] *= self.edge_to_reliability[edge] ** 3

        self.routed_gates.append((self.swap_gate, edge))

        # Update mappings
        node_a, node_b = edge
        qubit_a, qubit_b = (self.node_to_qubit[n] for n in edge)

        self.node_to_qubit[node_a], self.node_to_qubit[node_b] = qubit_b, qubit_a
        self.qubit_to_node[qubit_a], self.qubit_to_node[qubit_b] = node_b, node_a

    def _gate_reward(self, edge: tuple[int, int]) -> float:
        return self.edge_to_log_reliability[edge] + self.added_gate_reward

    def _swap_reward(self, edge: tuple[int, int]) -> float:
        return 3.0 * self.edge_to_log_reliability[edge]

    def _bridge_reward(self, control: int, middle: int, target: int) -> float:
        return 2.0 * (
            self.edge_to_log_reliability[(middle, target)] +
            self.edge_to_log_reliability[(control, middle)]
        ) + self.added_gate_reward


class ObsModule(ABC):
    @staticmethod
    @abstractmethod
    def key() -> str:
        raise NotImplementedError

    @abstractmethod
    def space(self, env: RoutingEnv) -> spaces.Box:
        raise NotImplementedError

    @abstractmethod
    def obs(self, env: RoutingEnv) -> NDArray:
        raise NotImplementedError


class LogReliabilities(ObsModule):
    @staticmethod
    def key() -> str:
        return 'log_reliabilities'

    def space(self, env: RoutingEnv) -> spaces.Box:
        return spaces.Box(env.noise_config.min_log_reliability, 0.0, shape=env.log_reliabilities.shape)

    def obs(self, env: RoutingEnv) -> NDArray:
        return env.log_reliabilities.copy()


class CircuitMatrix(ObsModule):
    def __init__(self, depth: int = 8):
        if depth <= 0:
            raise ValueError(f'Depth must be positive, got {depth}')

        self.depth = depth

    @staticmethod
    def key() -> str:
        return 'circuit_matrix'

    def space(self, env: RoutingEnv) -> spaces.Box:
        return spaces.Box(0, env.num_qubits, (env.num_qubits, self.depth), dtype=np.int32)

    def obs(self, env: RoutingEnv) -> NDArray:
        space = self.space(env)
        circuit = np.zeros(space.shape, dtype=space.dtype)

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

            circuit[idx_a, layer_idx] = env.qubit_to_node[idx_b] + 1
            circuit[idx_b, layer_idx] = env.qubit_to_node[idx_a] + 1

        mapped_circuit = np.zeros_like(circuit)
        for q in range(env.num_qubits):
            mapped_circuit[q] = circuit[env.node_to_qubit[q]]

        return mapped_circuit


class QubitInteractions(ObsModule):
    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth

    @staticmethod
    def key() -> str:
        return 'qubit_interactions'

    def space(self, env: RoutingEnv) -> spaces.Box:
        return spaces.Box(-1, self.max_depth, shape=(env.num_qubits * (env.num_qubits - 1) // 2,), dtype=np.int32)

    def obs(self, env: RoutingEnv) -> NDArray:
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
