
import itertools

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Instruction


class ParameterGenerator:
    def __init__(self, prefix: str = 'π'):
        self.prefix = prefix
        self.count = itertools.count()

    def generate(self) -> Parameter:
        return Parameter(f'{self.prefix}{next(self.count)}')

    def parameterize(self, instruction: Instruction) -> Instruction:
        instruction = instruction.copy()
        instruction.params = [self.generate() for _ in instruction.params]
        return instruction

    def parameterize_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        new = qc.copy_empty_like()
        for circuit_instruction in qc.data:
            instruction = self.parameterize(circuit_instruction.operation)
            new.append(instruction, circuit_instruction.qubits, circuit_instruction.clbits)
        return new

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.prefix!r}, {self.count!r})'
