
from typing import List, Tuple

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction

from gradient_based import gradient_based_hst_weighted

CircuitAction = Tuple[Instruction, Tuple[int, ...]]

class CircuitEnv(gym.Env):
    @staticmethod
    def _is_valid_action(action: CircuitAction) -> bool:
        return (action[0].name not in {'delay', 'id', 'reset'} and
            action[0].num_clbits == 0)

    def __init__(
        self,
        u: QuantumCircuit,
        actions: List[CircuitAction],
        max_depth: int,
    ):
        self.u = u
        self.v = QuantumCircuit(u.num_qubits)
        self.depth = 0
        self.max_depth = max_depth
        self.actions = list(filter(CircuitEnv._is_valid_action, actions))
        num_actions = len(actions)

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(num_actions + 1), spaces.Discrete(max_depth + 1))
        )

    def step(self, action: int):
        instruction, qubits = self.actions[action]
        self.v.append(instruction, qubits)

        reward = 0.0
        terminated = self.depth == self.max_depth
        if terminated:
            # Optimize continuous parameters
            params, cost = gradient_based_hst_weighted(self.u, self.v)
            reward = 1.0 - cost

        return (action + 1, self.depth), reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.u = QuantumCircuit()
        self.v = QuantumCircuit()
        self.depth = 0

        return (0, self.depth), {}

def dqn_train(
    u: QuantumCircuit,
    max_depth: int,
    e_greedy_circuits: List[Tuple[float, int]],
    actions: List[CircuitAction],
    learning_rate: float,
    discount_factor: float,
    batch_size: int,
) -> DQN:
    model = DQN(
        'MlpPolicy',
        CircuitEnv(u, actions, max_depth),
        learning_rate=learning_rate,
        gamma=discount_factor,
        batch_size=batch_size,
    )

    for epsilon, num_episodes in e_greedy_circuits:
        pass

    return model
