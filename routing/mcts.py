
import random
from math import sqrt
from typing import List, Optional

import numpy as np
import rustworkx as rx
import torch

from routing.circuit_gen import CircuitGenerator
from routing.env import LayeredRoutingEnv


class MctsState:
    parent: Optional['MctsState']
    children: List[Optional['MctsState']]

    def __init__(
        self,
        env: LayeredRoutingEnv,
        parent: Optional['MctsState'] = None,
        parent_action: Optional[int] = None,
        reward: float = 0.0,
    ):
        self.env = env
        self.parent = parent
        self.parent_action = parent_action
        self.reward = reward

        num_actions = env.action_space.n
        self.q_values = torch.zeros(num_actions)
        self.num_visits = torch.zeros(num_actions)
        self.children = [None] * num_actions
        self.action_masks = self.env.action_masks()

        self.priors = torch.tensor(np.bitwise_not(self.action_masks) * -1e8)

    @property
    def is_terminal(self) -> bool:
        return self.env.terminated

    def select(self) -> int:
        total_visits = torch.sum(self.num_visits).item()
        uct: torch.Tensor = self.q_values + self.priors * sqrt(total_visits + 1e-3) / (self.num_visits + 1e-3)
        max_indices, = torch.where(uct == uct.max())
        return random.choice(max_indices).item()

    def select_best(self) -> int:
        q_masked: torch.Tensor = self.q_values + torch.tensor(np.bitwise_not(self.action_masks) * -1e8)
        max_indices, = torch.where(q_masked == q_masked.max())
        return random.choice(max_indices).item()

    def update_q_value(self, action: int, reward: float):
        self.q_values[action] = (
            (self.q_values[action] * self.num_visits[action] + reward) /
            (self.num_visits[action] + 1)
        )
        self.num_visits[action] += 1


class Mcts:
    def __init__(
        self,
        coupling_map: rx.PyGraph,
        circuit_generator: CircuitGenerator,
        discount_factor: float = 0.95,
        search_depth: int = 128,
    ):
        self.root = MctsState(LayeredRoutingEnv(coupling_map, circuit_generator))
        self.discount_factor = discount_factor
        self.search_depth = search_depth

        self.env.reset()

    @property
    def env(self) -> LayeredRoutingEnv:
        return self.root.env

    def search(self):
        for _ in range(self.search_depth):
            state = self.root
            depth = 0

            while True:
                depth += 1
                action = state.select()

                if state.children[action] is not None:
                    state = state.children[action]
                elif state.is_terminal:
                    break
                else:
                    env = state.env.copy()
                    obs, reward, terminated, *_ = env.step(action)
                    next_state = MctsState(env, state, action, reward)
                    state.children[action] = next_state
                    state = next_state
                    break

            total_reward = 0.0
            while state.parent is not None:
                total_reward = state.reward + self.discount_factor * total_reward
                state.parent.update_q_value(state.parent_action, total_reward)
                state = state.parent

    def act(self):
        while True:
            self.search()
            action = self.root.select_best()

            if action == self.env.action_space.n - 1:
                self.root = self.root.children[action]
                return
            else:
                self.root = self.root.children[action]
