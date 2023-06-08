
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Set, List, SupportsFloat

import gymnasium as gym
import numpy as np
import rustworkx as rx
from gymnasium import spaces
from gymnasium.core import ObsType, ActType
from nptyping import NDArray, Int32
from qiskit import QuantumCircuit
from qiskit.circuit import Clbit
from qiskit.circuit.library import SwapGate, CXGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

from routing.circuit_gen import CircuitGenerator
from utils import qubits_to_indices, indices_to_qubits


class RoutingEnv(gym.Env[ObsType, ActType], ABC):
    initial_mapping: Optional[NDArray]
    protected_nodes: Set[int]
    gates_to_schedule: List[Tuple[DAGOpNode, Tuple[int, ...]]]

    def __init__(
        self,
        coupling_map: rx.PyGraph,
        circuit_generator: CircuitGenerator,
        gate_reward: float = 1.0,
        initial_mapping: Optional[NDArray] = None,
        allow_bridge_gate: bool = True,
        training: bool = True,
        training_iterations: int = 1,
    ):
        if initial_mapping is not None and initial_mapping.shape != (coupling_map.num_nodes(),):
            raise ValueError('Initial mapping has invalid shape for the provided coupling map')

        if training_iterations < 1:
            raise ValueError(f'Number of training iterations must be greater than zero, got {training_iterations}')

        self.coupling_map = coupling_map
        self.circuit_generator = circuit_generator
        self.gate_reward = gate_reward
        self.initial_mapping = initial_mapping
        self.allow_bridge_gate = allow_bridge_gate
        self.training = training
        self.training_iterations = training_iterations

        self.num_qubits = coupling_map.num_nodes()
        self.num_edges = coupling_map.num_edges()
        self.shortest_paths = rx.all_pairs_dijkstra_shortest_paths(coupling_map, lambda _: 1.0)
        self.distance_matrix = rx.distance_matrix(coupling_map).astype(np.int32)

        if self.allow_bridge_gate:
            self.bridge_pairs = sorted(
                (j, i) for i in range(self.num_qubits) for j in range(i) if self.distance_matrix[j, i] == 2
            )
        else:
            self.bridge_pairs = []

        # State information
        self.node_to_qubit = np.zeros(self.num_qubits, dtype=np.int32)
        self.qubit_to_node = np.zeros(self.num_qubits, dtype=np.int32)
        self.protected_nodes = set()
        self.gates_to_schedule = []

        self.circuit = QuantumCircuit(self.num_qubits)
        self.dag = circuit_to_dag(self.circuit)
        self.routed_dag = self.dag.copy_empty_like()

        bridge_qc = QuantumCircuit(3)
        for _ in range(2):
            bridge_qc.cx(1, 2)
            bridge_qc.cx(0, 1)

        self.bridge_gate = bridge_qc.to_gate(label='BRIDGE')
        self.bridge_gate.name = 'bridge'
        self.swap_gate = SwapGate()

        self.iter = 0
        self.rng = np.random.default_rng()

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        reward = self._schedule_bridges(action) + self._schedule_gates() + self._schedule_swaps(action)
        return self._current_obs(), reward, self._terminated(), False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if self.training and self.iter == 0:
            if self.iter == 0:
                self.circuit = self.circuit_generator.generate()
            self.iter = (self.iter + 1) % self.training_iterations

        self._reset_dag()

        return self._current_obs(), {}

    @abstractmethod
    def action_masks(self) -> NDArray:
        raise NotImplementedError

    @abstractmethod
    def _current_obs(self) -> ObsType:
        raise NotImplementedError

    @abstractmethod
    def _schedule_bridges(self, action: ActType) -> float:
        raise NotImplementedError

    @abstractmethod
    def _schedule_swaps(self, action: ActType) -> float:
        raise NotImplementedError

    def _terminated(self) -> bool:
        return not self.dag.op_nodes(include_directives=False)

    def _schedule_gates(self) -> float:
        reward = 0.0

        for op_node, nodes in self.gates_to_schedule:
            qargs = indices_to_qubits(self.circuit, nodes)

            self.routed_dag.apply_operation_back(op_node.op, qargs)
            self.dag.remove_op_node(op_node)

            if len(nodes) == 2:
                reward += self.gate_reward

        return reward

    def _bridge(self, control: int, middle: int, target: int):
        indices = (control, middle, target)
        qargs = indices_to_qubits(self.circuit, indices)

        self.routed_dag.apply_operation_back(self.bridge_gate, qargs)
        self.protected_nodes.update(indices)

    def _swap(self, edge: Tuple[int, int]):
        qargs = indices_to_qubits(self.circuit, edge)

        self.routed_dag.apply_operation_back(self.swap_gate, qargs)
        self.protected_nodes.update(edge)

        # Update mappings
        node_a, node_b = edge
        qubit_a, qubit_b = [self.node_to_qubit[q] for q in edge]

        self.node_to_qubit[node_a], self.node_to_qubit[node_b] = qubit_b, qubit_a
        self.qubit_to_node[qubit_a], self.qubit_to_node[qubit_b] = node_b, node_a

    def _reset_dag(self):
        self.dag = circuit_to_dag(self.circuit)
        self.routed_dag = self.dag.copy_empty_like()

        self._reset_state()

    def _reset_state(self):
        if self.initial_mapping is None:
            self.node_to_qubit = np.arange(self.num_qubits)
            np.random.shuffle(self.node_to_qubit)
        else:
            self.node_to_qubit = self.initial_mapping.copy()

        for node, qubit in enumerate(self.node_to_qubit):
            self.qubit_to_node[qubit] = node

        self._update_state()

    def _update_state(self):
        self.protected_nodes = set()
        self.gates_to_schedule = []

        for op_node in self.dag.front_layer():
            indices = qubits_to_indices(self.circuit, op_node.qargs)
            nodes = tuple(self.qubit_to_node[i] for i in indices)

            if len(nodes) != 2 or self.coupling_map.has_edge(nodes[0], nodes[1]):
                self.protected_nodes.update(nodes)
                self.gates_to_schedule.append((op_node, nodes))


@dataclass
class NoiseConfig:
    mean: float
    std: float
    recalibration_interval: int = 16


QcpObsType = Dict[str, NDArray]


class QcpRoutingEnv(RoutingEnv[QcpObsType, int]):
    observation_space: spaces.Dict
    action_space: spaces.Discrete

    def __init__(
        self,
        coupling_map: rx.PyGraph,
        circuit_generator: CircuitGenerator,
        depth: int,
        gate_reward: float = 1.0,
        termination_reward: float = 10.0,
        swap_penalty: float = -3.0,
        non_execution_penalty: float = -1.0,
        initial_mapping: Optional[NDArray] = None,
        allow_bridge_gate: bool = True,
        training: bool = True,
        training_iterations: int = 1,
        allow_idle_action: bool = True,
        noise_config: Optional[NoiseConfig] = None,
    ):
        super().__init__(coupling_map, circuit_generator, gate_reward, initial_mapping, allow_bridge_gate, training,
                         training_iterations)

        if depth <= 0:
            raise ValueError(f'Depth must be positive, got {depth}')

        self.termination_reward = termination_reward
        self.swap_penalty = swap_penalty
        self.non_execution_penalty = non_execution_penalty
        self.allow_idle_action = allow_idle_action
        self.noise_config = noise_config

        self.circuit_iter = 0
        self.log_reliabilities = np.zeros(self.num_edges)

        obs_spaces = {
            'circuit': spaces.Box(-1, self.num_qubits, (self.num_qubits, depth), dtype=Int32),
        }

        if self.noise_config:
            obs_spaces['log_reliabilities'] = spaces.Box(0.0, np.inf, shape=self.log_reliabilities.shape)

        num_actions = 1 + self.num_edges + len(self.bridge_pairs)

        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Discrete(num_actions)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[QcpObsType, Dict[str, Any]]:
        result = super().reset(seed=seed, options=options)

        if self.noise_config and self.training:
            if self.circuit_iter == 0:
                self._recalibrate()

            self.circuit_iter = (self.circuit_iter + 1) % self.noise_config.recalibration_interval

        return result

    def step(self, action: int) -> Tuple[QcpObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        next_obs, reward, terminated, truncated, info = super().step(action)

        if terminated:
            reward += self.termination_reward

        return next_obs, reward, terminated, truncated, info

    def action_masks(self) -> NDArray:
        mask = np.ones(self.action_space.n, dtype=bool)

        # Swap actions
        for i, edge in enumerate(self.coupling_map.edge_list()):
            if self.protected_nodes.intersection(edge):
                mask[i + 1] = False

        # Bridge actions
        for i, pair in enumerate(self.bridge_pairs):
            bridge_args = self._bridge_args(pair)

            if bridge_args is None:
                is_valid = False
            else:
                control, target = bridge_args[1]
                is_valid = not bool(self.protected_nodes.intersection(self.shortest_paths[control][target]))

            mask[i + self.num_edges + 1] = is_valid

        if not self.allow_idle_action and np.any(mask[1:]):
            mask[0] = False

        return mask

    def _current_obs(self) -> QcpObsType:
        space_circuit = self.observation_space.spaces['circuit']
        circuit = np.full(space_circuit.shape, -1, dtype=space_circuit.dtype)

        layer_idx = 0
        layer_qubits = set()

        for op_node in self.dag.op_nodes(include_directives=False):
            if len(op_node.qargs) == 2:
                indices = qubits_to_indices(self.circuit, op_node.qargs)

                if layer_qubits.intersection(indices):
                    layer_qubits = set(indices)
                    layer_idx += 1

                    if layer_idx == circuit.shape[1]:
                        break
                else:
                    layer_qubits.update(indices)

                idx_a, idx_b = indices

                circuit[idx_a, layer_idx] = idx_b
                circuit[idx_b, layer_idx] = idx_a

        mapped_circuit = np.zeros(shape=circuit.shape, dtype=circuit.dtype)

        for q in range(self.num_qubits):
            mapped_circuit[q] = circuit[self.node_to_qubit[q]]

        obs = {'circuit': mapped_circuit}
        if self.noise_config:
            obs['log_reliabilities'] = self.log_reliabilities.copy()

        return obs

    def _recalibrate(self):
        error_rates = np.clip(
            self.rng.normal(self.noise_config.mean, self.noise_config.std, self.num_edges), 0.0, 1.0
        )

        self.log_reliabilities = np.clip(np.where(error_rates > 0.0, -np.log(error_rates), np.inf), None, 300.0)

    def _bridge_args(self, pair: Tuple[int, int]) -> Optional[Tuple[DAGOpNode, Tuple[int, ...]]]:
        for op_node in self.dag.front_layer():
            if isinstance(op_node.op, CXGate):
                indices = qubits_to_indices(self.circuit, op_node.qargs)
                nodes = tuple(sorted(self.qubit_to_node[i] for i in indices))

                if nodes == pair:
                    return op_node, nodes

        return None

    def _schedule_bridges(self, action: int) -> float:
        reward = 0.0

        if action > self.num_edges:
            pair = self.bridge_pairs[action - self.num_edges - 1]
            bridge_args = self._bridge_args(pair)

            if bridge_args is not None:
                op_node, (control, target) = bridge_args
                self._bridge(*self.shortest_paths[control][target])
                self.dag.remove_op_node(op_node)
                reward += self.gate_reward + self.swap_penalty

        return reward

    def _schedule_swaps(self, action: int) -> float:
        reward = 0.0
        action_is_swap = 1 <= action <= self.num_edges

        if action_is_swap:
            edge = self.coupling_map.edge_list()[action - 1]
            if self.protected_nodes.intersection(edge):
                raise ValueError(f'Invalid action: cannot perform SWAP on edge {edge}')

            reward += self.swap_penalty
            self._swap(edge)

        self._update_state()

        if action_is_swap and not self.gates_to_schedule:
            reward += self.non_execution_penalty

        return reward


class LayeredRoutingEnv(RoutingEnv[NDArray, NDArray]):
    observation_space: spaces.MultiDiscrete
    action_space: spaces.MultiBinary

    def __init__(
        self,
        coupling_map: rx.PyGraph,
        circuit_generator: CircuitGenerator,
        gate_reward: float = 1.0,
        distance_reduction_reward: float = 0.1,
        initial_mapping: Optional[NDArray] = None,
        allow_bridge_gate: bool = True,
        training: bool = True,
        training_iterations: int = 1,
    ):
        super().__init__(coupling_map, circuit_generator, gate_reward, initial_mapping, allow_bridge_gate, training,
                         training_iterations)

        self.distance_reduction_reward = distance_reduction_reward

        self.diameter = np.max(self.distance_matrix)
        self.max_degree = max(coupling_map.degree(idx) for idx in coupling_map.node_indices())

        # State information
        self.qubit_targets = np.full(self.num_qubits, -1, dtype=np.int32)
        self.circuit_progress = np.zeros(self.num_qubits, dtype=np.int32)

        self.observation_space = spaces.MultiDiscrete(
            np.full(self.diameter + self.max_degree + 2, self.num_qubits + 1, dtype=np.int32)
        )
        self.action_space = spaces.MultiBinary(self.num_edges)

    def action_masks(self) -> NDArray:
        mask = np.ones(shape=(self.action_space.n, 2), dtype=bool)

        for i, edge in enumerate(self.coupling_map.edge_list()):
            if self.protected_nodes.intersection(edge):
                mask[i, 1] = False

        return mask

    def _current_obs(self) -> NDArray:
        obs = np.zeros(shape=self.observation_space.shape, dtype=self.observation_space.dtype)
        edge_vector_idx = self.diameter + 1

        for qubit in range(self.num_qubits):
            target = self.qubit_targets[qubit]
            if target != -1:
                # Distance vector
                qubit_node = self.qubit_to_node[qubit]
                target_node = self.qubit_to_node[target]

                distance = self.distance_matrix[qubit_node, target_node]
                obs[distance] += 1

                # Edge vector
                num_edges = 0
                for _, child_node, _ in self.coupling_map.out_edges(qubit):
                    child_target = self.qubit_targets[target]
                    if (
                        child_target not in {-1, qubit} and
                        self.shortest_paths[qubit][child_target][1] == child_node and
                        not self.protected_nodes.intersection({qubit, child_node})
                    ):
                        num_edges += 1
                obs[edge_vector_idx + num_edges] += 1

        return obs

    def _schedule_bridges(self, action: NDArray) -> float:
        # TODO
        return 0.0

    def _schedule_swaps(self, action: NDArray) -> float:
        reward = 0.0

        pre_swap_distances = self._qubit_distances()

        edge_list = self.coupling_map.edge_list()
        to_swap = [edge_list[i] for i in np.flatnonzero(action)]

        for edge in to_swap:
            if not self.protected_nodes.intersection(edge):
                self._swap(edge)

        self._update_state()

        post_swap_distances = self._qubit_distances()
        reward += np.sum(np.sign(pre_swap_distances - post_swap_distances)) * self.distance_reduction_reward

        return reward

    def _terminated(self) -> bool:
        return np.all(self.qubit_targets == -1)

    def _reset_state(self):
        super()._reset_state()

        self.circuit_progress.fill(0)

    def _update_state(self):
        super()._update_state()

        self.qubit_targets.fill(-1)

        for i, wire in enumerate(self.dag.wires):
            if isinstance(wire, Clbit):
                break

            for op_node in self.dag.nodes_on_wire(wire, only_ops=True):
                if isinstance(op_node.op, CXGate):
                    other = [q for q in op_node.qargs if q != wire][0]
                    other_idx = self.circuit.find_bit(other).index
                    self.qubit_targets[i] = other_idx

                    break

    def _qubit_distances(self) -> NDArray:
        distances = np.zeros(self.num_qubits)

        for qubit, target in enumerate(self.qubit_targets):
            if target != -1:
                qubit_node = self.qubit_to_node[qubit]
                target_node = self.qubit_to_node[target]

                distances[qubit] = self.distance_matrix[qubit_node, target_node]

        return distances
