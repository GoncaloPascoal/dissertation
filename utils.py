
import functools
import operator
from typing import Iterable

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit, Gate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode


def qubits_to_indices(qc: QuantumCircuit, qubits: Iterable[Qubit]) -> tuple[int, ...]:
    return tuple(qc.find_bit(q).index for q in qubits)  # type: ignore


def indices_to_qubits(qc: QuantumCircuit, indices: Iterable[int]) -> tuple[Qubit, ...]:
    return tuple(qc.qubits[i] for i in indices)


def reliability(circuit: QuantumCircuit, reliability_map: dict[tuple[int, ...], float]) -> float:
    return functools.reduce(operator.mul, [
        reliability_map[qubits_to_indices(circuit, instruction.qubits)]
        for instruction in circuit.get_instructions('cx')
    ])


def dag_layers(dag: DAGCircuit) -> list[list[DAGOpNode]]:
    graph_layers = dag.multigraph_layers()
    try:
        next(graph_layers)  # Remove input nodes
    except StopIteration:
        return []

    layers = []
    for graph_layer in graph_layers:
        layer = [node for node in graph_layer if isinstance(node, DAGOpNode) and isinstance(node.op, Gate)]

        if not layer:
            return layers

        layers.append(layer)

    return layers
