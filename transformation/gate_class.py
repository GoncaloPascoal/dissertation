
from dataclasses import dataclass
from math import isclose
from typing import ClassVar, List, Optional, Tuple

import rustworkx as rx
from qiskit.circuit import Gate, ParameterExpression
from qiskit.circuit.library import CXGate


@dataclass(frozen=True, slots=True)
class GateClass:
    gate: Gate

    PARAM_EPSILON: ClassVar[float] = 1.0e-4

    def qubits(self, qubit: int) -> Tuple[int, ...]:
        return qubit,

    def equals_gate(self, qubit: int, other: Gate, other_qubits: Tuple[int, ...]) -> bool:
        if (
            self.gate.name != other.name or
            self.qubits(qubit) != other_qubits or
            len(self.gate.params) != len(other.params)
        ):
            return False

        for param, other_param in zip(self.gate.params, other.params):
            if not (
                isinstance(param, ParameterExpression) or
                isclose(param, other_param, abs_tol=GateClass.PARAM_EPSILON)
            ):
                return False

        return True


@dataclass(frozen=True, slots=True)
class _TwoQubitMapGateClass(GateClass):
    qubit_map: List[int]

    def qubits(self, qubit: int) -> Tuple[int, ...]:
        return qubit, self.qubit_map[qubit]


def generate_two_qubit_gate_classes_from_coupling_map(
    coupling_map: List[Tuple[int, int]],
    gate: Optional[Gate] = None,
) -> List[GateClass]:
    if gate is None:
        gate = CXGate()

    graph = rx.PyGraph()
    graph.extend_from_edge_list(coupling_map)

    gate_classes = []
    max_degree = max(graph.degree(n) for n in graph.node_indices())
    for i in range(max_degree):
        qubit_map = []
        for q in graph.node_indices():
            neighbors = sorted(graph.neighbors(q))
            qubit_map.append(neighbors[i] if i < len(neighbors) else -1)

        gate_classes.append(_TwoQubitMapGateClass(gate, qubit_map))

    return gate_classes
