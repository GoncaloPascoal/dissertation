
import random
from collections.abc import Iterable, Sequence
from typing import Tuple, Optional, List, Dict

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import SXGate, RZGate, CXGate
from tqdm import tqdm

from rl import CircuitEnv
from utils import NativeInstruction

Observation = Tuple[int, int, int]
Action = int


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
) -> Tuple[QuantumCircuit, float]:
    q_a, q_b = {}, {}

    replay_buffer = []
    env = CircuitEnv.from_args(u, actions, length, cx_penalty_weight)
    env_actions = list(range(env.action_space.n))

    def max_q(q: Dict[Tuple[Observation, Action], float], obs: Observation) -> float:
        max_action = max(env_actions, key=lambda a: q.get((obs, a), 0.0))
        return q.get((obs, max_action), 0.0)

    best_v = QuantumCircuit(u.num_qubits)
    best_reward = 0.0

    for epsilon, num_episodes in epsilon_greedy_episodes:
        for _ in tqdm(range(num_episodes), desc=f'{epsilon = }'):
            obs, _ = env.reset()
            seq = []
            reward = 0.0

            for _ in range(length):
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = max(env_actions, key=lambda a: q_a.get((obs, a), 0.0) + q_b.get((obs, a), 0.0))

                seq.append((obs, action))
                obs, reward, *_ = env.step(action)
            seq.append((obs, None))

            replay_buffer.append((seq, reward))

            if reward > best_reward:
                best_v, best_reward = env.v.copy(), reward

            for sample, sample_reward in random.sample(replay_buffer, min(batch_size, len(replay_buffer))):
                y = random.random()
                intermediate_reward = sample_reward / length

                for (obs_t, action), (obs_tp1, _) in zip(sample, sample[1:]):
                    if y < 0.5:
                        q_a[(obs_t, action)] = (
                            (1 - learning_rate) * q_a.get((obs_t, action), 0.0) +
                            learning_rate * (intermediate_reward + discount_factor) + max_q(q_b, obs_tp1)
                        )
                    else:
                        q_a[(obs_t, action)] = (
                            (1 - learning_rate) * q_b.get((obs_t, action), 0.0) +
                            learning_rate * (intermediate_reward + discount_factor) + max_q(q_a, obs_tp1)
                        )

    return best_v, best_reward


def main():
    from rich import print

    u = QuantumCircuit(2)
    u.ch(0, 1)

    actions = build_native_instructions(2)

    epsilon_greedy_episodes = [(1.0, 1500), (0.9, 100), (0.8, 100), (0.7, 100), (0.6, 150),
                               (0.5, 150),  (0.4, 150), (0.3, 150), (0.2, 150), (0.1, 150)]

    best_v, best_reward = vqc_double_q_learning(u, actions, epsilon_greedy_episodes, 6)
    print(best_v, best_reward)


if __name__ == '__main__':
    main()
