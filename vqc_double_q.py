
from collections.abc import Iterable
from typing import Tuple, Optional, List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import SXGate, RZGate, CXGate
from ray.rllib.algorithms.dqn import DQN

from rl import dqn, build_algorithm
from utils import NativeInstruction


def build_native_instructions(
    num_qubits: int,
    qubit_topology: Optional[Iterable[Tuple[int, int]]] = None,
    directed: bool = False,
) -> List[NativeInstruction]:
    sx = SXGate()
    rz = RZGate(Parameter('x'))
    cx = CXGate()

    native_instructions = []

    for q in range(num_qubits):
        native_instructions.append(NativeInstruction(sx, (q,)))
        native_instructions.append(NativeInstruction(rz, (q,)))

    if qubit_topology is None:
        qubit_topology = ((q, q + 1) for q in range(num_qubits - 1))

    for qubits in qubit_topology:
        if not directed:
            native_instructions.append(NativeInstruction(cx, qubits[::-1]))
        native_instructions.append(NativeInstruction(cx, qubits))

    return native_instructions


def main():
    u = QuantumCircuit(2)
    u.ch(0, 1)

    actions = build_native_instructions(2)

    epsilon_greedy_episodes = [(1.0, 1500), (0.9, 100), (0.8, 100), (0.7, 100), (0.6, 150),
                               (0.5, 150),  (0.4, 150), (0.3, 150), (0.2, 150), (0.1, 150)]

    config = dqn(algo_class=DQN)
    algorithm = build_algorithm(config, u, actions, 6, epsilon_greedy_episodes)

    for _ in range(30):
        algorithm.train()


if __name__ == '__main__':
    main()
