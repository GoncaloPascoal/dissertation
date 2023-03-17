
from typing import Dict, List, Tuple
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

def _hst_base(n: int) -> Tuple[QuantumCircuit, QuantumRegister, QuantumRegister, ClassicalRegister]:
    sys_a = QuantumRegister(n, 'SysA')
    sys_b = QuantumRegister(n, 'SysB')
    classical = ClassicalRegister(2 * n, 'C')

    return QuantumCircuit(sys_a, sys_b, classical), sys_a, sys_b, classical

def _hst_entangle(qc: QuantumCircuit):
    n = qc.num_qubits // 2

    qc.h(range(n))
    for i in range(n):
        qc.cx(i, i + n)

def lhst(u: QuantumCircuit, v_adj: QuantumCircuit, i: int) -> QuantumCircuit:
    assert u.num_qubits == v_adj.num_qubits

    n = u.num_qubits
    assert i < n
    qc, sys_a, sys_b, classical = _hst_base(n)

    _hst_entangle(qc)

    qc.compose(u.to_gate(label='U'), sys_a, inplace=True)
    qc.compose(v_adj.to_gate(label='V*'), sys_b, inplace=True)

    qc.cx(i, i + n)
    qc.h(i)

    pair = [i, i + n]
    qc.measure(pair, pair)

    return qc

def hst(u: QuantumCircuit, v_adj: QuantumCircuit) -> QuantumCircuit:
    assert u.num_qubits == v_adj.num_qubits

    n = u.num_qubits
    qc, sys_a, sys_b, classical = _hst_base(n)

    _hst_entangle(qc)

    qc.compose(u.to_gate(label='U'), sys_a, inplace=True)
    qc.compose(v_adj.to_gate(label='V*'), sys_b, inplace=True)

    for i in range(n):
        qc.cx(i, i + n)
    qc.h(sys_a)

    qc.measure_all(add_bits=False)

    return qc

def fidelity_hst(counts: Dict[str, float]) -> float:
    assert counts

    return counts['0' * len(next(iter(counts)))]

def fidelity_lhst(counts_list: List[Dict[str, float]]) -> float:
    assert counts_list

    total = 0.0
    n = len(counts_list)

    for i, counts in enumerate(counts_list):
        for k, v in counts.items():
            k = k[::-1]
            if k[i] == k[i + n] == '0':
                total += v

    return total / n

def cost_hst(fidelity: float, n: int) -> float:
    d = 2 ** n
    return (d + 1) * (1 - fidelity) / d

def cost_lhst(fidelity: float) -> float:
    return 1 - fidelity
