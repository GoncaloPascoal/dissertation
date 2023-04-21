
import itertools

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

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.prefix!r}, {self.count!r})'
