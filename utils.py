
import functools
import operator
from typing import Iterable

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit


def qubits_to_indices(qc: QuantumCircuit, qubits: Iterable[Qubit]) -> tuple[int, ...]:
    return tuple(qc.find_bit(q).index for q in qubits)  # type: ignore


def indices_to_qubits(qc: QuantumCircuit, indices: Iterable[int]) -> tuple[Qubit, ...]:
    return tuple(qc.qubits[i] for i in indices)


def reliability(circuit: QuantumCircuit, reliability_map: dict[tuple[int, ...], float]) -> float:
    return functools.reduce(operator.mul, [
        reliability_map[qubits_to_indices(circuit, instruction.qubits)]
        for instruction in circuit.get_instructions('cx')
    ])
