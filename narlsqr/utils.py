
import functools
import operator
import random
from collections.abc import Callable
from typing import Final, Iterable, TypeAlias, TypeVar

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Qubit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

_T = TypeVar('_T')
Factory: TypeAlias = Callable[[], _T]


IBM_BASIS_GATES: Final = ['cx', 'id', 'rz', 'sx', 'x']


def qubits_to_indices(qc: QuantumCircuit, qubits: Iterable[Qubit]) -> tuple[int, ...]:
    return tuple(qc.find_bit(q).index for q in qubits)  # type: ignore

def indices_to_qubits(qc: QuantumCircuit, indices: Iterable[int]) -> tuple[Qubit, ...]:
    return tuple(qc.qubits[i] for i in indices)


def reliability(circuit: QuantumCircuit, reliability_map: dict[tuple[int, ...], float]) -> float:
    return functools.reduce(
        operator.mul,
        [
            reliability_map[qubits_to_indices(circuit, instruction.qubits)]
            for instruction in circuit.get_instructions('cx')
        ],
        1.0,
    )


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


def seed_default_generators(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
