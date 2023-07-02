
from collections.abc import Collection
from typing import Dict, Optional

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.result import marginal_counts

from utils import counts_to_ratios


class HilbertSchmidt(QuantumCircuit):
    def __init__(self, u: QuantumCircuit, v: QuantumCircuit, measure_qubit: Optional[int] = None, name: str = 'HST'):
        if u.num_qubits != v.num_qubits:
            raise ValueError('Unitaries U and V do not have the same number of qubits.')

        n = u.num_qubits

        sys_a = QuantumRegister(n, 'A')
        sys_b = QuantumRegister(n, 'B')
        classical = ClassicalRegister(2 * n, 'C')

        qc = QuantumCircuit(sys_a, sys_b, classical)

        qc.h(range(n))
        for i in range(n):
            qc.cx(i, i + n)

        qc.compose(u.to_gate(label='U'), sys_a, inplace=True)
        qc.compose(v.inverse().to_gate(label='V*'), sys_a, inplace=True)

        qubits = range(n) if measure_qubit is None else [measure_qubit]
        for i in reversed(qubits):
            qc.cx(i, i + n)
            qc.h(i)

        if measure_qubit is None:
            qc.measure_all(add_bits=False)
        else:
            pair = [measure_qubit, measure_qubit + n]
            qc.measure(pair, pair)

        super().__init__(sys_a, sys_b, classical, name=name)
        self.compose(qc, qubits=self.qubits, inplace=True)

class LocalHilbertSchmidt(HilbertSchmidt):
    def __init__(self, u: QuantumCircuit, v: QuantumCircuit, measure_qubit: int, name: str = 'LHST'):
        super().__init__(u, v, measure_qubit, name)


class LoschmidtEcho(QuantumCircuit):
    def __init__(self, u: QuantumCircuit, v: QuantumCircuit, measure_qubit: Optional[int] = None, name: str = 'LET'):
        if u.num_qubits != v.num_qubits:
            raise ValueError('Unitaries U and V do not have the same number of qubits.')

        n = u.num_qubits

        quantum = QuantumRegister(n, name='Q')
        classical = ClassicalRegister(n, name='C')

        qc = QuantumCircuit(quantum, classical)

        qc.compose(u.to_gate(label='U'), quantum, inplace=True)
        qc.compose(v.inverse().to_gate(label='V*'), quantum, inplace=True)

        if measure_qubit is None:
            qc.measure_all(add_bits=False)
        else:
            qc.measure(measure_qubit, measure_qubit)

        super().__init__(quantum, classical, name=name)
        self.compose(qc, qubits=self.qubits, inplace=True)

class LocalLoschmidtEcho(LoschmidtEcho):
    def __init__(self, u: QuantumCircuit, v: QuantumCircuit, measure_qubit: int, name: str = 'LLET'):
        super().__init__(u, v, measure_qubit, name)


def fidelity_global(counts: Dict[str, int]) -> float:
    assert counts

    l = len(next(iter(counts)))
    return counts_to_ratios(counts).get('0' * l, 0.0)

def fidelity_lhst(counts_list: Collection[Dict[str, int]]) -> float:
    assert counts_list

    total = 0.0
    n = len(counts_list)

    for i, counts in enumerate(counts_list):
        ratios = counts_to_ratios(marginal_counts(counts, [i, i + n]))
        total += ratios.get('00', 0.0)

    return total / n

def fidelity_llet(counts_list: Collection[Dict[str, int]]) -> float:
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
        C_\\text{HST} = 1 - F_\\text{HST}(U, V)

    :param counts: dictionary containing the number of times each value was measured.
    """
    return 1 - fidelity_global(counts)

def cost_lhst(counts_list: Collection[Dict[str, int]]) -> float:
    return 1 - fidelity_lhst(counts_list)

def cost_let(counts: Dict[str, int]) -> float:
    return 1 - fidelity_global(counts)

def cost_llet(counts_list: Collection[Dict[str, int]]) -> float:
    return 1 - fidelity_llet(counts_list)

def cost_hst_weighted(counts: Dict[str, int], counts_list: Collection[Dict[str, int]], q: float) -> float:
    if q == 0.0:
        return cost_lhst(counts_list)
    elif q == 1.0:
        return cost_hst(counts)
    else:
        return q * cost_hst(counts) + (1 - q) * cost_lhst(counts_list)

def cost_let_weighted(counts: Dict[str, int], counts_list: Collection[Dict[str, int]], q: float) -> float:
    if q == 0.0:
        return cost_llet(counts_list)
    elif q == 1.0:
        return cost_let(counts)
    else:
        return q * cost_let(counts) + (1 - q) * cost_llet(counts_list)
