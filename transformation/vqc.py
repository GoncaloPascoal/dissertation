
from typing import Literal, List

import numpy as np
from nptyping import NDArray, Int8
from qiskit import QuantumCircuit
from qiskit.converters import dag_to_circuit

from dag_utils import op_node_at, dag_layers
from gate_class import GateClass
from vqc.optimization.continuous.gradient_based import gradient_based_hst_weighted
from parameter_generator import ParameterGenerator
from transformation.env import TransformationCircuitEnv, TransformationRule
from utils import ContinuousOptimizationFunction


class VqcTransformationCircuitEnv(TransformationCircuitEnv):
    ObsType = NDArray[Literal['*, *, *'], Int8]

    def __init__(
        self,
        max_depth: int,
        num_qubits: int,
        gate_classes: List[GateClass],
        transformation_rules: List[TransformationRule],
        *,
        continuous_optimization: ContinuousOptimizationFunction = gradient_based_hst_weighted,
        exponent_depth: float = 0.4,
        exponent_gate_count: float = 0.8,
        exponent_cost: float = 5.0,
        incremental_weight: float = 1.0e-3,
        **kwargs,
    ):
        super().__init__(max_depth, num_qubits, gate_classes, transformation_rules, **kwargs)

        self.continuous_optimization = continuous_optimization
        self.exponent_depth = exponent_depth
        self.exponent_gate_count = exponent_gate_count
        self.exponent_cost = exponent_cost
        self.incremental_weight = incremental_weight

        self.param_gen = ParameterGenerator()
        self.current_cost = 0.0
        self.next_cost = 0.0

        self.best_reward = 0.0
        self.best_circuit = self.target_circuit.copy()

    def _calculate_reward(self, depth_diff: float, gate_count_diff: float, cost_diff: float) -> float:
        reward_depth = np.sign(depth_diff) * abs(depth_diff) ** self.exponent_depth
        reward_gate_count = np.sign(gate_count_diff) * abs(gate_count_diff) ** self.exponent_gate_count
        reward_cost = cost_diff ** self.exponent_cost

        return (reward_depth + 0.2 * reward_gate_count) * reward_cost

    def reward(self) -> float:
        depth_next = self.next_circuit.depth()
        depth_current = self.current_circuit.depth()
        depth_target = self.target_circuit.depth()

        gate_count_next = self.next_circuit.size()
        gate_count_current = self.current_circuit.size()
        gate_count_target = self.target_circuit.size()

        incremental_depth_diff = (depth_current - depth_next) / depth_target
        incremental_gate_count_diff = (gate_count_current - gate_count_next) / depth_target
        incremental_cost_diff = self.current_cost - self.next_cost

        incremental_reward = self.incremental_weight * self._calculate_reward(
            incremental_depth_diff,
            incremental_gate_count_diff,
            incremental_cost_diff,
        )

        depth_diff = 1.0 - depth_next / depth_target
        gate_count_diff = 1.0 - gate_count_next / gate_count_target

        if depth_diff > 0.0 or gate_count_diff > 0.0:
            cost_diff = 1.0 - self.next_cost
            discovery_reward = self._calculate_reward(depth_diff, gate_count_diff, cost_diff)
        else:
            discovery_reward = 0.0

        if not self.training and discovery_reward > self.best_reward:
            self.best_reward, self.best_circuit = discovery_reward, self.next_circuit.copy()

        return incremental_reward + discovery_reward

    def current_obs(self) -> ObsType:
        return np.concatenate(
            (self.circuit_to_obs(self.target_circuit), self.circuit_to_obs(self.current_circuit)),
            axis=1,
        )

    def build_next_circuit(self, decoded_action: TransformationCircuitEnv.DecodedAction) -> QuantumCircuit:
        rule = self.transformation_rules[decoded_action.rule]

        qc = rule.apply(self, decoded_action.layer, decoded_action.qubit)
        qc = self.param_gen.parameterize_circuit(qc)

        params, cost = self.continuous_optimization
        self.next_cost = cost

        return qc.bind_parameters(params)

    def reset_circuits(self):
        super().reset_circuits()

        if not self.training:
            self.best_reward = 0.0
            self.best_circuit = self.target_circuit.copy()


class SwapGates(TransformationRule):
    def is_valid(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> bool:
        dag = env.current_dag
        op_node_a, op_node_b = op_node_at(dag, layer, qubit), op_node_at(dag, layer + 1, qubit)

        if op_node_a is None or op_node_b is None:
            return False

        if op_node_a.op == op_node_b.op and op_node_a.qargs == op_node_b.qargs:
            return False

        qubits_a = set(dag.qubits.index(q) for q in op_node_a.qargs)
        qubits_b = set(dag.qubits.index(q) for q in op_node_b.qargs)

        if qubits_a != qubits_b:
            depth_increase = 0

            if any(op_node_at(dag, layer + 1, q) is not None for q in qubits_a - qubits_b):
                depth_increase += 1
            if any(op_node_at(dag, layer, q) is not None for q in qubits_b - qubits_a):
                depth_increase += 1

            if env.current_circuit.depth() + depth_increase > env.max_depth:
                # Disallow transformation if it would cause the circuit to exceed the maximum depth
                return False

        return True

    def apply(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> QuantumCircuit:
        dag = env.current_dag
        op_node_a, op_node_b = op_node_at(dag, layer, qubit), op_node_at(dag, layer + 1, qubit)

        dag.swap_nodes(op_node_a, op_node_b)

        return dag_to_circuit(dag, copy_operations=False)


class ShiftQubits(TransformationRule):
    def __init__(self, down: bool):
        self.offset = 1 if down else -1

    def is_valid(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> bool:
        qc = env.current_circuit
        dag = env.current_dag
        op_node = op_node_at(dag, layer, qubit)

        if op_node is None:
            return False

        first_qubit, _ = env.indices_from_op_node(qc, op_node)

        if first_qubit != qubit:
            return False

        qubits_dst = tuple(qc.find_bit(q)[0] + self.offset for q in op_node.qargs)

        if any(q not in range(env.num_qubits) for q in qubits_dst):
            # Destination qubits are not valid
            return False

        for q in qubits_dst:
            op_node_dst = op_node_at(dag, layer, q)
            if op_node_dst is not None and op_node_dst != op_node:
                # Already a gate at the target location
                return False

        return True

    def apply(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> QuantumCircuit:
        qc = env.current_circuit
        dag = env.current_dag

        op_node = op_node_at(dag, layer, qubit)
        qubits_dst = tuple(qc.find_bit(q)[0] + self.offset for q in op_node.qargs)
        qargs_dst = tuple(qc.qubits[q] for q in qubits_dst)

        op_node.qargs = qargs_dst

        return dag_to_circuit(dag, copy_operations=False)


class ChangeGateClass(TransformationRule):
    def __init__(self, next_class: bool):
        self.offset = 1 if next_class else -1

    def is_valid(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> bool:
        dag = env.current_dag
        op_node = op_node_at(dag, layer, qubit)

        if op_node is None:
            return False

        first_qubit, class_idx = env.indices_from_op_node(env.current_circuit, op_node)

        if first_qubit != qubit:
            return False

        class_idx = (class_idx + self.offset) % len(env.gate_classes)
        new_class = env.gate_classes[class_idx]

        qubits = new_class.qubits(qubit)
        for q in qubits:
            op_node_dst = op_node_at(dag, layer, q)
            if op_node_dst is not None and op_node_dst != op_node:
                # Already a gate at the target location
                return False

        return True

    def apply(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> QuantumCircuit:
        dag = env.current_dag
        op_node_target = op_node_at(dag, layer, qubit)

        new_dag = dag.copy_empty_like()

        _, class_idx = env.indices_from_op_node(env.current_circuit, op_node_target)

        class_idx = (class_idx + self.offset) % len(env.gate_classes)
        new_class = env.gate_classes[class_idx]
        qubits = new_class.qubits(qubit)
        qargs = [dag.qubits[q] for q in qubits]

        for op_node in dag.op_nodes(include_directives=False):
            if op_node == op_node_target:
                new_dag.apply_operation_back(new_class.gate, qargs)
            else:
                new_dag.apply_operation_back(op_node.op, op_node.qargs, op_node.cargs)

        return dag_to_circuit(dag, copy_operations=False)


class RemoveGate(TransformationRule):
    def is_valid(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> bool:
        dag = env.current_dag
        op_node = op_node_at(dag, layer, qubit)

        if op_node is None:
            return False

        first_qubit, _ = env.indices_from_op_node(env.current_circuit, op_node)

        if first_qubit != qubit:
            return False

        return True

    def apply(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> QuantumCircuit:
        dag = env.current_dag
        op_node = op_node_at(dag, layer, qubit)

        dag.remove_op_node(op_node)

        return dag_to_circuit(dag, copy_operations=False)


class AddGate(TransformationRule):
    def __init__(self, gate_class: GateClass):
        self.gate_class = gate_class

    def is_valid(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> bool:
        dag = env.current_dag
        depth = dag.depth()

        if layer > depth:
            return False

        qubits = self.gate_class.qubits(qubit)

        if any(q >= env.num_qubits for q in qubits):
            # Instruction qubits must be within bounds
            return False

        if layer == depth and depth < env.max_depth:
            # Can insert instruction the end of the circuit without exceeding maximum depth
            return True

        if any(op_node_at(dag, layer, q) is not None for q in qubits):
            return False

        if layer > 0:
            if all(op_node_at(dag, layer - 1, q) is None for q in qubits):
                return False

        return True

    def apply(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> QuantumCircuit:
        dag = env.current_dag
        layers = dag_layers(dag)

        new_dag = dag.copy_empty_like()
        gate = self.gate_class.gate
        qubits = self.gate_class.qubits(qubit)

        for i, dag_layer in enumerate(layers):
            for op_node in dag_layer:
                new_dag.apply_operation_back(op_node.op, op_node.qargs, op_node.cargs)

            if i == layer:
                new_dag.apply_operation_back(gate, qubits)

        return dag_to_circuit(new_dag, copy_operations=False)
