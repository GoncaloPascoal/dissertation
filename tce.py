
from collections.abc import Callable, Sequence
import functools
import operator
from typing import Any, Dict, Literal, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from nptyping import Int8, NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.converters import circuit_to_dag

from rich import print

from parameter_generator import ParameterGenerator


class TransformationCircuitEnv(gym.Env):
    Observation = NDArray[Literal['*, *, *'], Int8]
    Action = int

    QubitCallback = Callable[[int], Tuple[int, ...]]

    def __init__(self, context: Dict[str, Any]):
        self.u: QuantumCircuit = context['u']
        self.native_instructions: Sequence[Instruction] = context['native_instructions']
        self.qubit_callbacks: Dict[int, TransformationCircuitEnv.QubitCallback] = context['qubit_callbacks']
        self.max_depth: int = context['max_depth']

        self.instruction_map = {
            instruction.name: i for i, instruction in enumerate(self.native_instructions)
        }
        self.param_gen = ParameterGenerator()

        self.observation_space = spaces.MultiBinary(
            (self.max_depth, self.u.num_qubits, len(self.native_instructions)),
        )
        self.action_space = spaces.Discrete(functools.reduce(operator.mul, self.observation_space.shape))

        self.current_observation = np.zeros(self.observation_space.shape)

    @classmethod
    def from_args(
        cls,
        u: QuantumCircuit,
        native_instructions: Sequence[Instruction],
        qubit_callbacks: Dict[int, QubitCallback],
        max_depth: int,
    ) -> 'TransformationCircuitEnv':
        return cls({
            'u': u,
            'native_instructions': native_instructions,
            'qubit_callbacks': qubit_callbacks,
            'max_depth': max_depth,
        })

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        return self.current_observation, 0.0, False, False, {}

    def reset(self, *, seed=None, options=None) -> Tuple[Observation, Dict[str, Any]]:
        return self.current_observation, {}

    @staticmethod
    def _default_qubit_callback(q: int) -> Tuple[int, ...]:
        return (q,)

    def _circuit_to_obs(self, qc: QuantumCircuit) -> Observation:
        if qc.depth() > self.max_depth:
            raise ValueError(f'Circuit depth must not exceed {self.max_depth}')

        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        dag = circuit_to_dag(qc)

        for layer_idx, layer in enumerate(dag.layers()):
            for op_node in layer['graph'].op_nodes(include_directives=False):
                first_qubit = qc.find_bit(op_node.qargs[0]).index
                instruction_type = self.instruction_map[op_node.name]

                obs[layer_idx, first_qubit, instruction_type] = 1

        return obs

    def _obs_to_circuit(self, obs: Observation) -> QuantumCircuit:
        qc = QuantumCircuit(self.u.num_qubits)

        for layer in obs:
            for first_qubit, instructions in enumerate(layer):
                indices = np.flatnonzero(instructions)
                if indices.size > 0:
                    idx = indices[0]

                    instruction = self.native_instructions[idx]
                    qubits = self.qubit_callbacks.get(idx, self._default_qubit_callback)(first_qubit)

                    if instruction.is_parameterized():
                        instruction = self.param_gen.parameterize(instruction)

                    qc.append(instruction, qubits)

        return qc


def main():
    u = QuantumCircuit(2)
    u.ch(0, 1)

    from qiskit.circuit.library import CXGate, SXGate, RZGate
    from qiskit.circuit import Parameter

    sx = SXGate()
    rz = RZGate(Parameter('x'))
    cx = CXGate()

    native_instructions = [sx, rz, cx, cx]
    qubit_callbacks = {
        2: lambda q: (q, q + 1),
        3: lambda q: (q + 1, q),
    }

    env = TransformationCircuitEnv.from_args(u, native_instructions, qubit_callbacks, 6)


if __name__ == '__main__':
    main()
