
from typing import List, Tuple, Literal

from gymnasium import spaces
from nptyping import NDArray, Int8

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import CXGate, HGate, RXGate, RZGate, SXGate
from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from gate_class import GateClass
from transformation.env import TransformationCircuitEnv, TransformationRule
from dag_utils import op_node_at


class ExactTransformationCircuitEnv(TransformationCircuitEnv):
    ObsType = NDArray[Literal['*, *, *'], Int8]

    def __init__(
        self,
        max_depth: int,
        num_qubits: int,
        gate_classes: List[GateClass],
        transformation_rules: List[TransformationRule],
        *,
        weight_depth: float = 1.0,
        weight_gate_count: float = 0.2,
        **kwargs,
    ):
        super().__init__(max_depth, num_qubits, gate_classes, transformation_rules, **kwargs)

        self.weight_depth = weight_depth
        self.weight_gate_count = weight_gate_count

        self.observation_space: spaces.Box = spaces.Box(
            0, 1, (len(gate_classes), num_qubits, max_depth), dtype=Int8,
        )

    def reward(self) -> float:
        depth_diff = self.current_circuit.depth() - self.next_circuit.depth()
        gate_count_diff = self.current_circuit.size() - self.next_circuit.size()

        return self.weight_depth * depth_diff + self.weight_gate_count * gate_count_diff

    def current_obs(self) -> ObsType:
        return self.circuit_to_obs(self.current_circuit)

    def build_next_circuit(self, decoded_action: TransformationCircuitEnv.DecodedAction) -> QuantumCircuit:
        rule = self.transformation_rules[decoded_action.rule]
        return transpile(
            rule.apply(self, decoded_action.layer, decoded_action.qubit),
            approximation_degree=0.0,
            basis_gates=self.basis_gates,
        )


class InvertCnot(TransformationRule):
    def is_valid(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> bool:
        qc = env.current_circuit
        dag = env.current_dag
        op_node = op_node_at(dag, layer, qubit)

        if op_node is None or not isinstance(op_node.op, CXGate):
            return False

        first_qubit, _ = env.indices_from_op_node(qc, op_node)

        if first_qubit != qubit:
            return False

        return qc.depth() + 6 <= env.max_depth

    def apply(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> QuantumCircuit:
        dag = env.current_dag
        cx_op_node = op_node_at(dag, layer, qubit)

        new_dag: DAGCircuit = dag.copy_empty_like()
        hadamard = HGate()

        new_dag.apply_operation_back(hadamard, cx_op_node.qargs[:1])
        new_dag.apply_operation_back(hadamard, cx_op_node.qargs[1:])

        new_dag.apply_operation_back(cx_op_node.op, cx_op_node.qargs[::-1])

        new_dag.apply_operation_back(hadamard, cx_op_node.qargs[:1])
        new_dag.apply_operation_back(hadamard, cx_op_node.qargs[1:])

        dag.substitute_node_with_dag(cx_op_node, new_dag, list(cx_op_node.qargs))

        return dag_to_circuit(dag, copy_operations=False)


class CommuteGates(TransformationRule):
    """
    Commutes two adjacent gates. The resulting unitary must be equivalent up to a global phase. The
    following commutation relations are implemented:

    **1. Rz and CNOT**
        Z-rotations commute with CNOT gates in the control qubit.
        ::
            ┌───────┐               ┌───────┐
            ┤ Rz(π) ├──■──     ──■──┤ Rz(π) ├
            └───────┘┌─┴─┐  =  ┌─┴─┐└───────┘
            ─────────┤ X ├     ┤ X ├─────────
                     └───┘     └───┘

    **2. Sx and CNOT**
        X-rotations commute with CNOT gates in the target qubit.
        ::
            ────────■──     ──■────────
            ┌────┐┌─┴─┐  =  ┌─┴─┐┌────┐
            ┤ √X ├┤ X ├     ┤ X ├┤ √X ├
            └────┘└───┘     └───┘└────┘

    **3. Between CNOTs**
        CNOT gates commute if they share a target or control qubit.
    """

    def is_valid(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> bool:
        dag = env.current_dag
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

        if isinstance(op_a, CXGate) != isinstance(op_b, CXGate):
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

        if isinstance(op_a, CXGate) and isinstance(op_b, CXGate):
            # CNOT gates commute if they share a control or target qubit
            return qubits_a[0] == qubits_b[0] or qubits_a[1] == qubits_b[1]

        return False

    def apply(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> QuantumCircuit:
        dag = env.current_dag
        op_node_a, op_node_b = op_node_at(dag, layer, qubit), op_node_at(dag, layer + 1, qubit)

        dag.swap_nodes(op_node_a, op_node_b)

        return dag_to_circuit(dag, copy_operations=False)


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
        dag = env.current_dag
        op_nodes = [op_node_at(dag, l, qubit) for l in range(layer, layer + 4)]

        if any(n is None for n in op_nodes):
            return False

        reverse = self._is_reverse(dag, layer, qubit)

        return (
            CommuteRzBetweenCnots._check_gates(dag, op_nodes, qubit, reverse) and
            CommuteRzBetweenCnots._check_layers(env, dag, layer, qubit, reverse)
        )

    def apply(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> QuantumCircuit:
        dag = env.current_dag
        op_nodes = [op_node_at(dag, l, qubit) for l in range(layer, layer + 4)]

        reverse = self._is_reverse(dag, layer, qubit)
        cx_a, rz, cx_b, cx_c = CommuteRzBetweenCnots._assign_op_nodes(op_nodes, reverse)

        nodes = (cx_b, rz, cx_a)
        if reverse:
            nodes = reversed(nodes)

        for node in nodes:
            dag.swap_nodes(node, cx_c)

        return dag_to_circuit(dag, copy_operations=False)


class CollapseFourAlternatingCnots(TransformationRule):
    def is_valid(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> bool:
        dag = env.current_dag
        op_nodes = [op_node_at(dag, layer + i, qubit) for i in range(4)]

        if any(n is None or not isinstance(n.op, CXGate) for n in op_nodes):
            return False

        qargs = [n.qargs for n in op_nodes]

        return all(q_b == q_a[::-1] for q_a, q_b in zip(qargs, qargs[1:]))

    def apply(self, env: TransformationCircuitEnv, layer: int, qubit: int) -> QuantumCircuit:
        dag = env.current_dag

        to_remove = [op_node_at(dag, layer, qubit), op_node_at(dag, layer + 3, qubit)]
        for op_node in to_remove:
            dag.remove_op_node(op_node)

        return dag_to_circuit(dag, copy_operations=False)
