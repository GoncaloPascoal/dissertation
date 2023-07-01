
from typing import List, Optional

import rustworkx as rx
import torch

from routing.circuit_gen import CircuitGenerator
from routing.env import LayeredRoutingEnv


class MctsState:
    children: List[Optional['MctsState']]

    def __init__(self, env: LayeredRoutingEnv):
        self.env = env

        num_actions = env.action_space.n
        self.q_values = torch.zeros(num_actions)
        self.num_visits = torch.zeros(num_actions)
        self.children = [None] * num_actions

    @property
    def is_terminal(self) -> bool:
        return self.env.terminated


class Mcts:
    def __init__(self, coupling_map: rx.PyGraph, circuit_generator: CircuitGenerator):
        self.root = MctsState(LayeredRoutingEnv(coupling_map, circuit_generator))

    def search(self):
        state = self.root
        depth = 0

        while True:
            depth += 1
            action = 1

            if state.children[action] is not None:
                state = state.children[action]
            elif state.is_terminal:
                break
            else:
                env = state.env.copy()
                obs, reward, terminated, *_ = env.step(action)
                next_state = MctsState(env)
                state.children[action] = next_state
