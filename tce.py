
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from nptyping import Int8, NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.converters import circuit_to_dag

from rich import print

from parameter_generator import ParameterGenerator
from utils import index_with_key


def default_qubit_callback(qubit: int) -> Tuple[int, ...]:
    return qubit,

def nn_qubit_callback(qubit: int) -> Tuple[int, ...]:
    return qubit, qubit + 1

def nn_qubit_callback_reversed(qubit: int) -> Tuple[int, ...]:
    return qubit + 1, qubit


@dataclass
class InstructionCallback:
    QubitCallback = Callable[[int], Tuple[int, ...]]

    instruction: Instruction
    qubit_callback: QubitCallback = default_qubit_callback

    def equals_instruction(
        self,
        other: Instruction,
        other_qubits: Tuple[int, ...],
        qubit: int,
    ) -> bool:
        if (
            self.instruction.name != other.name or
            self.qubit_callback(qubit) != other_qubits or
            len(self.instruction.params) != len(other.params)
        ):
            return False

        for param, other_param in zip(self.instruction.params, other.params):
            if not isinstance(param, ParameterExpression) and param != other_param:
                return False

        return True

    def __call__(self, param_gen: ParameterGenerator, qc: QuantumCircuit, qubit: int):
        instruction = self.instruction
        if self.instruction.is_parameterized():
            instruction = param_gen.parameterize(instruction)

        qc.append(instruction, self.qubit_callback(qubit))


class TransformationCircuitEnv(gym.Env):
    Observation = NDArray[Literal['*, *, *'], Int8]
    Action = int

    TransformationCallback = Callable[[ParameterGenerator, QuantumCircuit, int, int], None]

    def __init__(self, context: Dict[str, Any]):
        self.u: QuantumCircuit = context['u']
        self.max_depth: int = context['max_depth']
        self.instruction_callbacks: List[InstructionCallback] = context['instruction_callbacks']
        self.transformation_callbacks = context['transformation_callbacks']

        self.param_gen = ParameterGenerator()

        self.observation_space = spaces.MultiBinary(
            (self.max_depth, self.u.num_qubits, len(self.instruction_callbacks)),
        )
        self.action_space = spaces.Discrete(self.max_depth * self.u.num_qubits * len(self.transformation_callbacks))

        self.current_observation = np.zeros(self.observation_space.shape)

    @classmethod
    def from_args(
        cls,
        u: QuantumCircuit,
        max_depth: int,
        instruction_callbacks: List[InstructionCallback],
        transformation_callbacks: List[TransformationCallback],
    ) -> 'TransformationCircuitEnv':
        return cls({
            'u': u,
            'max_depth': max_depth,
            'instruction_callbacks': instruction_callbacks,
            'transformation_callbacks': transformation_callbacks,
        })

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        return self.current_observation, 0.0, False, False, {}

    def reset(self, *, seed=None, options=None) -> Tuple[Observation, Dict[str, Any]]:
        return self.current_observation, {}

    def _circuit_to_obs(self, qc: QuantumCircuit) -> Observation:
        if qc.depth() > self.max_depth:
            raise ValueError(f'Circuit depth must not exceed {self.max_depth}')

        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        dag = circuit_to_dag(qc)

        for layer_idx, layer in enumerate(dag.layers()):
            for op_node in layer['graph'].op_nodes(include_directives=False):
                qubit = qc.find_bit(op_node.qargs[0]).index
                qubits = tuple(qc.find_bit(q).index for q in op_node.qargs)

                try:
                    instruction_idx = index_with_key(
                        self.instruction_callbacks,
                        lambda c: c.equals_instruction(op_node.op, qubits, qubit)
                    )
                except ValueError as ex:
                    raise ValueError('Circuit is incompatible with target instruction set') from ex

                obs[layer_idx, qubit, instruction_idx] = 1

        return obs

    def _obs_to_circuit(self, obs: Observation) -> QuantumCircuit:
        qc = QuantumCircuit(self.u.num_qubits)

        for layer in obs:
            for first_qubit, instructions in enumerate(layer):
                indices = np.flatnonzero(instructions)
                if indices.size > 0:
                    self.instruction_callbacks[indices[0]](self.param_gen, qc, first_qubit)

        return qc


def main():
    u = QuantumCircuit(2)
    u.ch(0, 1)

    from qiskit.circuit.library import CXGate, SXGate, RZGate
    from qiskit.circuit import Parameter

    sx = SXGate()
    rz = RZGate(Parameter('x'))
    cx = CXGate()

    instruction_callbacks = [
        InstructionCallback(sx),
        InstructionCallback(rz),
        InstructionCallback(cx, nn_qubit_callback),
        InstructionCallback(cx, nn_qubit_callback_reversed),
    ]

    # TODO: transformation callbacks
    env = TransformationCircuitEnv.from_args(u, 6, instruction_callbacks, [])


if __name__ == '__main__':
    main()
