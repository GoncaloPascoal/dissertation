
import itertools

from qiskit.circuit import Parameter, Instruction


class ParameterGenerator:
    def __init__(self, prefix: str = 'π'):
        self.prefix = prefix
        self.count = itertools.count()

    def generate(self) -> Parameter:
        return Parameter(f'{self.prefix}{next(self.count)}')

    def parametrize(self, instruction: Instruction) -> Instruction:
        instruction = instruction.copy()
        instruction.params = [self.generate() for _ in instruction.params]
        return instruction


class AutoParameter(Parameter):
    param_num = itertools.count()

    def __new__(cls, uuid=None):
        return super().__new__(cls, None, uuid=uuid)

    def __getnewargs__(self):
        return self._uuid,

    def __init__(self):
        super().__init__(f'π{next(AutoParameter.param_num)}')

    def __getstate__(self):
        return {
            'name': self._name,
            'uuid': self._uuid,
        }

    def __setstate__(self, state):
        self._name = state['name']
        self._uuid = state['uuid']
        super().__init__(self._name)


def auto_parametrize(instruction: Instruction) -> Instruction:
    instruction = instruction.copy()
    instruction.params = [AutoParameter() for _ in instruction.params]
    return instruction
