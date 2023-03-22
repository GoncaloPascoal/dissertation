
from typing import List, Tuple

import gymnasium as gym
from stable_baselines3 import DQN

from qiskit.circuit import Instruction
from qiskit.transpiler import CouplingMap

class CircuitEnv(gym.Env):
    CircuitAction = Tuple[Instruction, Tuple[int, ...]]

    @staticmethod
    def _is_valid_action(action: CircuitAction) -> bool:
        return (action[0].name not in {'delay', 'id', 'reset'} and
            action[0].num_clbits == 0)

    def __init__(self, actions: List[CircuitAction]):
        self.actions = list(filter(CircuitEnv._is_valid_action, actions))
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def step(self, action):
        pass

