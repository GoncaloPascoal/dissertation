
from typing import Dict
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

def hilbert_schmidt_test(u: QuantumCircuit, v_adj: QuantumCircuit) -> QuantumCircuit:
    assert u.num_qubits == v_adj.num_qubits

    n = u.num_qubits

    sys_a = QuantumRegister(n, 'SysA')
    sys_b = QuantumRegister(n, 'SysB')
    classical = ClassicalRegister(2 * n, 'C')

    qc = QuantumCircuit(sys_a, sys_b, classical)

    qc.h(sys_a)
    for i in range(n):
        qc.cx(i, i + n)

    qc.compose(u.to_gate(label='U'), sys_a, inplace=True)
    qc.compose(v_adj.to_gate(label='U'), sys_a, inplace=True)

    for i in range(n):
        qc.cx(i, i + n)
    qc.h(sys_a)

    qc.measure_all(add_bits=False)

    return qc

def fidelity_hst(counts: Dict[str, float]) -> float:
    assert counts

    return counts['0' * len(next(iter(counts)))]

def fidelity_lhst(counts: Dict[str, float]) -> float:
    assert counts

    total = 0.0
    n = len(next(iter(counts))) // 2

    for i in range(n):
        for k, v in counts.items():
            if k[i] == k[i + n] == '0':
                total += v

    return total / n

def cost_hst(fidelity: float, n: int) -> float:
    d = 2 ** n
    return (d + 1) * (1 - fidelity) / d

def cost_lhst(fidelity: float) -> float:
    return 1 - fidelity
