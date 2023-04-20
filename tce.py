
from collections.abc import Callable
import functools
import operator
from typing import Any, Dict, List, Literal, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from nptyping import Int8, NDArray
from qiskit import QuantumCircuit

from rich import print

from auto_parameter import AutoParameter

InstructionApplicationMap = List[Callable[[QuantumCircuit, Tuple[int, ...]], None]]


class TransformationCircuitEnv(gym.Env):
    Observation = NDArray[Literal['*, *, *'], Int8]
    Action = Observation

    def __init__(self, context: Dict[str, Any]):
        self.u: QuantumCircuit = context['u']
        self.native_instructions: InstructionApplicationMap = context['native_instructions']
        self.max_depth: int = context['max_depth']

        self.observation_space = spaces.MultiBinary(
            (self.max_depth, self.u.num_qubits, len(self.native_instructions)),
        )
        self.action_space = spaces.Discrete(functools.reduce(operator.mul, self.observation_space.shape))

        self.current_observation = np.zeros(self.observation_space.shape)

    @classmethod
    def from_args(
        cls,
        u: QuantumCircuit,
        native_instructions: InstructionApplicationMap,
        max_depth: int,
    ) -> 'TransformationCircuitEnv':
        return cls({
            'u': u,
            'native_instructions': native_instructions,
            'max_depth': max_depth,
        })

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        return self.current_observation, 0.0, False, False, {}


def main():
    u = QuantumCircuit(2)
    u.ch(0, 1)

    def rz(qc: QuantumCircuit, qubits: Tuple[int, ...]):
        qc.rz(AutoParameter(), *qubits)

    def sx(qc: QuantumCircuit, qubits: Tuple[int, ...]):
        qc.sx(*qubits)

    def cx(qc: QuantumCircuit, qubits: Tuple[int, ...]):
        qc.cx(*qubits)

    def rcx(qc: QuantumCircuit, qubits: Tuple[int, ...]):
        qc.cx(*qubits[::-1])

    native_instructions = [rz, sx, cx, rcx]
    env = TransformationCircuitEnv.from_args(u, native_instructions, 6)

    print(env.observation_space)
    print(env.action_space)


if __name__ == '__main__':
    main()
