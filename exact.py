from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import CXGate, HGate, RXGate, RZGate, SXGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from tce import TransformationCircuitEnv, TransformationRule
from dag_utils import dag_layers, op_node_at


class InvertCnot(TransformationRule):
    def is_valid(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> bool:
        qc = env.current_circuit
        dag = circuit_to_dag(qc)
        op_node = op_node_at(dag, layer, qubit)

        if op_node is None or not isinstance(op_node.op, CXGate):
            return False

        first_qubit, _ = env.indices_from_op_node(qc, op_node)

        if first_qubit != qubit:
            return False

        return True

    def apply(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> QuantumCircuit:
        qc = env.current_circuit
        dag = circuit_to_dag(qc)
        cx_op_node = op_node_at(dag, layer, qubit)

        new_dag = dag.copy_empty_like()
        hadamard = HGate()

        for i, dag_layer in enumerate(dag_layers(dag)):
            if i == layer:
                dag_layer = [n for n in dag_layer if n != cx_op_node]

            for op_node in dag_layer:
                new_dag.apply_operation_back(op_node.op, op_node.qargs, op_node.cargs)

            if i == layer:
                new_dag.apply_operation_back(hadamard, cx_op_node.qargs[:1])
                new_dag.apply_operation_back(hadamard, cx_op_node.qargs[1:])

                new_dag.apply_operation_back(cx_op_node.op, cx_op_node.qargs[::-1])

                new_dag.apply_operation_back(hadamard, cx_op_node.qargs[:1])
                new_dag.apply_operation_back(hadamard, cx_op_node.qargs[1:])

        return dag_to_circuit(new_dag)


class CommuteGates(TransformationRule):
    def is_valid(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> bool:
        qc = env.current_circuit
        dag = circuit_to_dag(qc)
        op_node_a, op_node_b = op_node_at(dag, layer, qubit), op_node_at(dag, layer + 1, qubit)

        if op_node_a is None or op_node_b is None:
            # Gates must exist at the given location
            return False

        if (
            op_node_a.name == op_node_b.name and
            op_node_a.qargs == op_node_b.qargs and
            op_node_a.op.params == op_node_b.op.params
        ):
            # Commuting two identical gates is valid but redundant
            return False

        qubits_a = [dag.qubits.index(q) for q in op_node_a.qargs]
        qubits_b = [dag.qubits.index(q) for q in op_node_b.qargs]

        qubits_a_set = set(qubits_a)
        qubits_b_set = set(qubits_b)

        if qubits_a_set != qubits_b_set:
            depth_increase = 0

            if any(op_node_at(dag, layer + 1, q) is not None for q in qubits_a_set - qubits_b_set):
                depth_increase += 1
            if any(op_node_at(dag, layer, q) is not None for q in qubits_b_set - qubits_a_set):
                depth_increase += 1

            if env.current_circuit.depth() + depth_increase > env.max_depth:
                # Disallow transformation if it would cause the circuit to exceed the maximum depth
                return False

        op_a: Gate = op_node_a.op
        op_b: Gate = op_node_b.op

        if isinstance(op_node_a.op, CXGate) != isinstance(op_node_b.op, CXGate):
            # Exactly one of the two gates is a CNOT
            op_node_cx, op_node_other = op_node_a, op_node_b
            if isinstance(op_b, CXGate):
                op_node_cx, op_node_other = op_node_other, op_node_cx

            cx_control, cx_target = [dag.qubits.index(q) for q in op_node_cx.qargs]

            if isinstance(op_node_other.op, (SXGate, RXGate)):
                # X rotations (such as Sqrt-X) commute with CNOTs at the target qubit
                return qubit == cx_target
            elif isinstance(op_node_other.op, RZGate):
                # Z rotations commute with CNOTs at the control qubit
                return qubit == cx_control

        if isinstance(op_a, RZGate) and isinstance(op_b, RZGate):
            # Rz gates with different parameters commute
            return True

        if isinstance(op_a, CXGate) and isinstance(op_b, CXGate):
            # CNOT gates commute if they share a control or target qubit
            return qubits_a[0] == qubits_b[0] or qubits_a[1] == qubits_b[1]

        return False

    def apply(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> QuantumCircuit:
        qc = env.current_circuit
        dag = circuit_to_dag(qc)
        op_node_a, op_node_b = op_node_at(dag, layer, qubit), op_node_at(dag, layer + 1, qubit)

        new_dag = dag.copy_empty_like()

        for i, dag_layer in enumerate(dag_layers(dag)):
            if i == layer:
                dag_layer = [n for n in dag_layer if n != op_node_a]
                dag_layer.append(op_node_b)
            elif i == layer + 1:
                dag_layer = [n for n in dag_layer if n != op_node_b]
                dag_layer.insert(0, op_node_a)

            for op_node in dag_layer:
                new_dag.apply_operation_back(op_node.op, op_node.qargs, op_node.cargs)

        return dag_to_circuit(new_dag)


class CommuteRzBetweenCnots(TransformationRule):
    @staticmethod
    def _assign_op_nodes(
        op_nodes: List[DAGOpNode],
        reverse: bool,
    ) -> Tuple[DAGOpNode, DAGOpNode, DAGOpNode, DAGOpNode]:
        if reverse:
            cx_c, cx_a, rz, cx_b = op_nodes
        else:
            cx_a, rz, cx_b, cx_c = op_nodes

        return cx_a, rz, cx_b, cx_c

    @staticmethod
    def _assign_layer_indices(
        layer: int,
        reverse: bool,
    ) -> Tuple[int, int, int, int]:
        if reverse:
            return layer + 1, layer + 2, layer + 3, layer
        else:
            return tuple(range(layer, layer + 4))

    @staticmethod
    def _check_gates(dag: DAGCircuit, op_nodes: List[DAGOpNode], qubit: int, reverse: bool) -> bool:
        cx_a, rz, cx_b, cx_c = CommuteRzBetweenCnots._assign_op_nodes(op_nodes, reverse)

        cx_a_qubits, cx_b_qubits, cx_c_qubits = (
            tuple(dag.qubits.index(q) for q in n.qargs)
            for n in (cx_a, cx_b, cx_c)
        )

        if not (
            isinstance(cx_a.op, CXGate) and isinstance(cx_b.op, CXGate) and
            isinstance(rz.op, RZGate) and isinstance(cx_c.op, CXGate)
        ):
            return False

        ctrl_a, target_a = cx_a_qubits
        ctrl_c, target_c = cx_c_qubits

        return target_a == qubit and cx_b_qubits == cx_a_qubits and ctrl_c == target_a and target_c != ctrl_a

    @staticmethod
    def _check_layers(
        env: TransformationCircuitEnv,
        dag: DAGCircuit,
        layer: int,
        qubit: int,
        reverse: bool
    ) -> bool:
        layer_cx_a, layer_rz, layer_cx_b, layer_cx_c = CommuteRzBetweenCnots._assign_layer_indices(layer, reverse)

        cx_a, cx_c = (op_node_at(dag, l, qubit) for l in (layer_cx_a, layer_cx_c))
        cx_a_qubits, cx_c_qubits = (
            tuple(dag.qubits.index(q) for q in n.qargs)
            for n in (cx_a, cx_c)
        )

        ctrl_a = cx_a_qubits[0]
        target_c = cx_c_qubits[1]

        op_node = op_node_at(dag, layer_rz, ctrl_a)
        if not (op_node is None or isinstance(op_node.op, RZGate)):
            return False

        depth_increase = 0

        for l in (layer_cx_a, layer_rz, layer_cx_b):
            if op_node_at(dag, l, target_c) is not None:
                depth_increase += 1

        if op_node_at(dag, layer_cx_c, ctrl_a) is not None:
            depth_increase += 1

        return env.current_circuit.depth() + depth_increase <= env.max_depth

    @staticmethod
    def _is_reverse(dag: DAGCircuit, layer: int, qubit: int) -> bool:
        return isinstance(op_node_at(dag, layer + 1, qubit), CXGate)

    def is_valid(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> bool:
        qc = env.current_circuit
        dag = circuit_to_dag(qc)
        op_nodes = [op_node_at(dag, l, qubit) for l in range(layer, layer + 4)]

        if any(n is None for n in op_nodes):
            return False

        reverse = self._is_reverse(dag, layer, qubit)

        return (
            CommuteRzBetweenCnots._check_gates(dag, op_nodes, qubit, reverse) and
            CommuteRzBetweenCnots._check_layers(env, dag, layer, qubit, reverse)
        )

    def apply(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> QuantumCircuit:
        qc = env.current_circuit
        dag = circuit_to_dag(qc)
        op_nodes = [op_node_at(dag, l, qubit) for l in range(layer, layer + 4)]

        reverse = self._is_reverse(dag, layer, qubit)
        cx_a, rz, cx_b, cx_c = CommuteRzBetweenCnots._assign_op_nodes(op_nodes, reverse)
        layer_cx_a, layer_rz, layer_cx_b, layer_cx_c = CommuteRzBetweenCnots._assign_layer_indices(layer, reverse)

        new_dag = dag.copy_empty_like()

        for i, dag_layer in enumerate(dag_layers(dag)):
            if i < layer or i >= layer + 4:
                for op_node in dag_layer:
                    new_dag.apply_operation_back(op_node.op, op_node.qargs, op_node.cargs)
            else:
                if i in {layer_cx_a, layer_rz, layer_cx_b}:
                    for op_node in dag_layer:
                        if op_node not in {cx_a, rz, cx_b}:
                            new_dag.apply_operation_back(op_node.op, op_node.qargs, op_node.cargs)

                    if i == layer_cx_b:
                        ops = [cx_a, rz, cx_b] if reverse else [cx_c]
                        for op_node in ops:
                            new_dag.apply_operation_back(op_node.op, op_node.qargs, op_node.cargs)
                else:
                    ops = [cx_c] if reverse else [cx_a, rz, cx_b]
                    for op_node in ops:
                        new_dag.apply_operation_back(op_node.op, op_node.qargs, op_node.cargs)

                    for op_node in dag_layer:
                        if op_node != cx_c:
                            new_dag.apply_operation_back(op_node.op, op_node.qargs, op_node.cargs)

        return dag_to_circuit(new_dag)


class CollapseFourAlternatingCnots(TransformationRule):
    def is_valid(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> bool:
        qc = env.current_circuit
        dag = circuit_to_dag(qc)
        op_nodes = [op_node_at(dag, layer + i, qubit) for i in range(4)]

        if any(n is None or not isinstance(n.op, CXGate) for n in op_nodes):
            return False

        qargs = [n.qargs for n in op_nodes]

        return all(q_b == q_a[::-1] for q_a, q_b in zip(qargs, qargs[1:]))

    def apply(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> QuantumCircuit:
        qc = env.current_circuit
        dag = circuit_to_dag(qc)

        to_remove = [op_node_at(dag, layer, qubit), op_node_at(dag, layer + 3, qubit)]
        for op_node in to_remove:
            dag.remove_op_node(op_node)

        return dag_to_circuit(dag)
