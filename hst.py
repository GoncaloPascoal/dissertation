
from typing import Dict, List, Tuple
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from utils import counts_to_ratios

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


def _let_base(n: int) -> Tuple[QuantumCircuit, QuantumRegister, ClassicalRegister]:
    quantum = QuantumRegister(n, name='Q')
    classical = ClassicalRegister(n, name='C')

    return QuantumCircuit(quantum, classical), quantum, classical

def let(u: QuantumCircuit, v_adj: QuantumCircuit) -> QuantumCircuit:
    assert u.num_qubits == v_adj.num_qubits

    n = u.num_qubits
    qc, quantum, classical = _let_base(n)

    qc.compose(u.to_gate(label='U'), quantum, inplace=True)
    qc.compose(v_adj.to_gate(label='V*'), quantum, inplace=True)

    qc.measure_all(add_bits=False)

    return qc

def llet(u: QuantumCircuit, v_adj: QuantumCircuit, i: int) -> QuantumCircuit:
    assert u.num_qubits == v_adj.num_qubits

    n = u.num_qubits
    qc, quantum, classical = _let_base(n)

    qc.compose(u.to_gate(label='U'), quantum, inplace=True)
    qc.compose(v_adj.to_gate(label='V*'), quantum, inplace=True)

    qc.measure(i, i)

    return qc


def fidelity_global(counts: Dict[str, int]) -> float:
    assert counts

    l = len(next(iter(counts)))
    return counts_to_ratios(counts).get('0' * l, 0.0)

def fidelity_lhst(counts_list: List[Dict[str, int]]) -> float:
    assert counts_list

    total = 0.0
    n = len(counts_list)

    for i, counts in enumerate(counts_list):
        ratios = counts_to_ratios(marginal_counts(counts, [i, i + n]))
        total += ratios.get('00', 0.0)

    return total / n

def fidelity_llet(counts_list: List[Dict[str, int]]) -> float:
    assert counts_list

    total = 0.0
    n = len(counts_list)

    for i, counts in enumerate(counts_list):
        ratios = counts_to_ratios(marginal_counts(counts, [i]))
        total += ratios.get('0', 0.0)

    return total / n

def cost_hst(counts: Dict[str, int]) -> float:
    """
    Calculates the cost function according to the Hilbert-Schmidt test.
    
    .. math::
        C_\\text{HST} = \\frac{d + 1}{d}(1 - \\bar{F}(U, V))
    """
    assert counts

    n = len(next(iter(counts))) // 2
    d = 2 ** n
    return (d + 1) * (1 - fidelity_global(counts)) / d

def cost_lhst(counts_list: List[Dict[str, int]]) -> float:
    return 1 - fidelity_lhst(counts_list)

def cost_let(counts: Dict[str, int]) -> float:
    return 1 - fidelity_global(counts)

def cost_llet(counts_list: List[Dict[str, int]]) -> float:
    return 1 - fidelity_llet(counts_list)

def cost_hst_weighted(counts: Dict[str, int], counts_list: List[Dict[str, int]], q: float) -> float:
    return q * cost_hst(counts) + (1 - q) * cost_lhst(counts_list)

def cost_let_weighted(counts: Dict[str, int], counts_list: List[Dict[str, int]], q: float) -> float:
    return q * cost_let(counts) + (1 - q) * cost_llet(counts_list)

