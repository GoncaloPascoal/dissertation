
from typing import Dict, Any, Tuple, SupportsFloat, Optional, Set, List

import gymnasium as gym
import numpy as np
import rustworkx as rx
from gymnasium import spaces
from nptyping import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import Clbit
from qiskit.circuit.library import CXGate, SwapGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

from routing.circuit_gen import CircuitGenerator
from utils import qubits_to_indices, indices_to_qubits


class QubitRoutingEnv(gym.Env[NDArray, NDArray]):
    observation_space: spaces.MultiDiscrete
    action_space: spaces.MultiBinary

    protected_nodes: Set[int]
    cnots_to_schedule: List[Tuple[DAGOpNode, List[int]]]

    def __init__(
        self,
        coupling_map: rx.PyGraph,
        circuit_generator: CircuitGenerator,
        cnot_reward: float = 1.0,
        distance_reduction_reward: float = 0.1,
    ):
        self.coupling_map = coupling_map
        self.circuit_generator = circuit_generator

        self.cnot_reward = cnot_reward
        self.distance_reduction_reward = distance_reduction_reward

        self.shortest_paths = rx.all_pairs_dijkstra_shortest_paths(coupling_map, lambda e: 1.0)
        self.distance_matrix = rx.distance_matrix(coupling_map)

        num_qubits = coupling_map.num_nodes()
        diameter = int(np.max(self.distance_matrix))
        max_degree = max(coupling_map.degree(idx) for idx in coupling_map.node_indices())

        self.num_qubits = num_qubits
        self.diameter = diameter
        self.max_degree = max_degree

        self.observation_space = spaces.MultiDiscrete(np.full(diameter + max_degree + 2, num_qubits, dtype=np.int64))
        self.action_space = spaces.MultiBinary(coupling_map.num_edges())

        # State information
        self.node_to_qubit = np.zeros(num_qubits, dtype=np.int64)
        self.qubit_to_node = np.zeros(num_qubits, dtype=np.int64)
        self.qubit_targets = np.full(num_qubits, -1, dtype=np.int64)
        self.circuit_progress = np.zeros(num_qubits, dtype=np.int64)
        self.protected_nodes = set()
        self.cnots_to_schedule = []

        self.circuit = QuantumCircuit(num_qubits)
        self.dag = circuit_to_dag(self.circuit)
        self.routed_dag = self.dag.copy_empty_like()

    def step(self, action: NDArray) -> Tuple[NDArray, SupportsFloat, bool, bool, Dict[str, Any]]:
        reward = 0.0

        for op_node, nodes in self.cnots_to_schedule:
            qargs = indices_to_qubits(self.circuit, nodes)

            self.routed_dag.apply_operation_back(op_node.op, qargs)
            self.dag.remove_op_node(op_node)

            reward += self.cnot_reward

        pre_swap_distances = self._qubit_distances()

        edge_list = self.coupling_map.edge_list()
        to_swap = [edge_list[i] for i in np.flatnonzero(action)]

        for edge in to_swap:
            if not self.protected_nodes.intersection(edge):
                self._swap_nodes(edge)

        self._update_qubit_targets()

        post_swap_distances = self._qubit_distances()
        reward += np.sum(post_swap_distances - pre_swap_distances < 0) * self.distance_reduction_reward

        terminated = np.all(self.qubit_targets == -1)

        return self._current_obs(), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray, Dict[str, Any]]:
        self._generate_circuit()

        return self._current_obs(), {}

    def action_masks(self) -> NDArray:
        mask = np.ones(shape=(self.action_space.n, 2), dtype=bool)

        for i, edge in enumerate(self.coupling_map.edge_list()):
            if self.protected_nodes.intersection(edge):
                mask[i, 1] = False

        return mask

    def _generate_circuit(self):
        self.circuit = self.circuit_generator.generate()
        self.dag = circuit_to_dag(self.circuit)
        self.routed_dag = self.dag.copy_empty_like()

        self._reset_state()

    def _reset_state(self):
        self.node_to_qubit = np.arange(self.num_qubits)
        np.random.shuffle(self.node_to_qubit)

        for node, qubit in enumerate(self.node_to_qubit):
            self.qubit_to_node[qubit] = node

        self._update_qubit_targets()
        self.circuit_progress.fill(0)

    def _update_qubit_targets(self):
        self.protected_nodes = set()
        self.cnots_to_schedule = []

        for op_node in self.dag.front_layer():
            if isinstance(op_node.op, CXGate):
                indices = qubits_to_indices(self.circuit, op_node.qargs)
                nodes = [self.qubit_to_node[i] for i in indices]

                if self.distance_matrix[nodes[0]][nodes[1]] == 1:
                    self.protected_nodes.update(nodes)
                    self.cnots_to_schedule.append((op_node, nodes))

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

    def _swap_nodes(self, edge: Tuple[int, int]):
        qargs = indices_to_qubits(self.circuit, edge)

        self.routed_dag.apply_operation_back(SwapGate(), qargs)
        self.protected_nodes.update(edge)

        # Update mappings
        node_a, node_b = edge
        qubit_a, qubit_b = [self.node_to_qubit[q] for q in edge]

        self.node_to_qubit[node_a], self.node_to_qubit[node_b] = qubit_b, qubit_a
        self.qubit_to_node[qubit_a], self.qubit_to_node[qubit_b] = node_b, node_a

    def _current_obs(self) -> NDArray:
        obs = np.zeros(shape=self.observation_space.shape)
        edge_vector_idx = self.diameter + 1

        for qubit in range(self.num_qubits):
            target = self.qubit_targets[qubit]
            if target != -1:
                # Distance vector
                qubit_node = self.qubit_to_node[qubit]
                target_node = self.qubit_to_node[target]

                distance = int(self.distance_matrix[qubit_node, target_node])
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

    def _qubit_distances(self) -> NDArray:
        distances = np.zeros(self.num_qubits)

        for qubit, target in enumerate(self.qubit_targets):
            if target != -1:
                qubit_node = self.qubit_to_node[qubit]
                target_node = self.qubit_to_node[target]

                distances[qubit] = int(self.distance_matrix[qubit_node, target_node])

        return distances
