
import itertools

from qiskit.circuit import Parameter, Instruction


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
