
import random

from typing import Callable, Sequence, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, CircuitInstruction, Instruction
from qiskit.circuit.library import CXGate, SXGate, RZGate

from qiskit_aer import AerSimulator

from rich import print

from rl import CircuitAction
from gradient_based import gradient_based_hst_weighted

def _change_instruction(qc: QuantumCircuit, idx: int, instruction: Instruction):
    old = qc.data[idx]
    qc.data[idx] = CircuitInstruction(instruction, old.qubits, old.clbits)

def _change_qubits(qc: QuantumCircuit, idx: int, indices: Sequence[int]):
    qubits = tuple(qc.qubits[i] for i in indices)
    old = qc.data[idx]
    qc.data[idx] = CircuitInstruction(old.operation, qubits, old.clbits)

def _remove_instruction(qc: QuantumCircuit, idx: int):
    qc.data.pop(idx)

def _add_instruction(qc: QuantumCircuit, idx: int, instruction: Instruction, qubit_indices: Sequence[int]):
    qubits = tuple(qc.qubits[i] for i in qubit_indices)
    qc.data.insert(idx, CircuitInstruction(instruction, qubit_indices))

def _swap_instructions(qc: QuantumCircuit, idx_a: int, idx_b: int):
    qc.data[idx_a], qc.data[idx_b] = qc.data[idx_b], qc.data[idx_a]

class SimulatedAnnealing:
    ContinuousOptimizationFunction = Callable[
        [QuantumCircuit, QuantumCircuit],
        Tuple[Sequence[float], float]
    ]

    def _is_valid_instruction(self, instruction: CircuitAction) -> bool:
        return (instruction[0].name not in {'delay', 'id', 'reset'} and
            instruction[0].num_clbits == 0 and
            instruction[0].num_qubits <= self.u.num_qubits)

    def _generate_random_circuit(self, num_qubits: int, num_instructions: int) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)

        for _ in range(num_instructions):
            instruction, qubits = random.choice(self.native_instructions)
            qc.append(instruction, qubits)

        return qc

    # TODO: Temperature and annealing schedule. Replace CircuitAction in signature
    def __init__(
        self,
        u: QuantumCircuit,
        native_instructions: Sequence[CircuitAction],
        continuous_optimization: ContinuousOptimizationFunction,
        max_iterations: int,
    ):
        assert max_iterations > 0

        self.u = u
        self.native_instructions = native_instructions
        self.continuous_optimization = lambda v: continuous_optimization(self.u, v)
        self.max_iterations = max_iterations

        self.v = self._generate_random_circuit(u.num_qubits, 3)
        self.best_v = self.v
        self.best_params, self.best_cost = self.continuous_optimization(self.v)

    def _generate_neighbor(self) -> QuantumCircuit:
        neighbor = self.v.copy()
        # TODO: generate neighbor
        return neighbor

    def run(self) -> Tuple[QuantumCircuit, Sequence[float], float]:
        for _ in range(self.max_iterations):
            # TODO: simulated annealing algorithm
            pass

        return self.best_v, self.best_params, self.best_cost

if __name__ == '__main__':
    u = QuantumCircuit(2)
    u.h(0)

    sx = SXGate()
    rz = RZGate(Parameter('a'))
    cx = CXGate()

    native_instructions = [
        (sx, (0,)),
        (sx, (1,)),
        (rz, (0,)),
        (rz, (1,)),
        (cx, (0, 1)),
        (cx, (1, 0)),
    ]

    def continuous_optimization(
        u: QuantumCircuit,
        v: QuantumCircuit,
    ) -> Tuple[Sequence[float], float]:
        return gradient_based_hst_weighted(u, v)

    algo = SimulatedAnnealing(u, native_instructions, continuous_optimization, 50)
