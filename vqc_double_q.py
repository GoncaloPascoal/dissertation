
import random

from collections.abc import Iterable, Sequence
from typing import Tuple, Optional, List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import SXGate, RZGate, CXGate
from ray.rllib.algorithms.dqn import DQN

from rl import dqn, build_algorithm, CircuitEnv
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


def vqc_double_q_learning(
    u: QuantumCircuit,
    actions: Sequence[NativeInstruction],
    epsilon_greedy_episodes: Sequence[Tuple[float, int]],
    length: int,
    cx_penalty_weight: float = 0.0,
    learning_rate: float = 0.02,
    discount_factor: float = 0.9,
    batch_size: int = 128,
):
    replay_buffer = []
    env = CircuitEnv.from_args(u, actions, length, cx_penalty_weight)

    best_v = QuantumCircuit(u.num_qubits)
    best_params = []
    best_reward = 0.0

    for epsilon, num_episodes in epsilon_greedy_episodes:
        for _ in range(num_episodes):
            env.reset()
            seq = []
            reward = 0.0

            for _ in range(length):
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = 0

                seq.append(action)
                reward = env.step(action)[1]

            replay_buffer.append((seq, reward))

    return best_v, best_params, best_reward


def main():
    u = QuantumCircuit(2)
    u.ch(0, 1)

    actions = build_native_instructions(2)

    epsilon_greedy_episodes = [(1.0, 1500), (0.9, 100), (0.8, 100), (0.7, 100), (0.6, 150),
                               (0.5, 150),  (0.4, 150), (0.3, 150), (0.2, 150), (0.1, 150)]

    vqc_double_q_learning(u, actions, epsilon_greedy_episodes, 6)


if __name__ == '__main__':
    main()
