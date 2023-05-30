
from qiskit import QuantumCircuit
from qiskit.circuit.quantumcircuit import QubitSpecifier


def apply_bridge_gate(
    qc: QuantumCircuit,
    control_qubit: QubitSpecifier,
    middle_qubit: QubitSpecifier,
    target_qubit: QubitSpecifier,
):
    for _ in range(2):
        qc.cx(middle_qubit, target_qubit)
        qc.cx(control_qubit, middle_qubit)
