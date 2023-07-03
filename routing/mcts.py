
import random
from math import sqrt
from typing import List, Optional, Tuple

import numpy as np
import rustworkx as rx
import torch
from torch import nn
from torch_geometric.nn import EdgeConv

from routing.circuit_gen import CircuitGenerator
from routing.env import LayeredRoutingEnv, RoutingObsType


class MctsGnn(nn.Module):
    def __init__(self, env: LayeredRoutingEnv):
        super().__init__()
        coupling_map = env.coupling_map

        mlp = nn.Sequential(
            nn.Linear(2 * coupling_map.num_nodes(), 50),
            nn.SiLU(),
            nn.Linear(50, 10),
            nn.SiLU(),
            nn.Linear(10, 4),
            nn.SiLU(),
        )
        self.edge_conv = EdgeConv(mlp)

        self.value_net = nn.Sequential(
            nn.Linear(4 * coupling_map.num_nodes() + coupling_map.num_nodes() + coupling_map.num_edges(), 64),
            nn.SiLU(),
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        self.policy_net = nn.Sequential(
            nn.Linear(4 * coupling_map.num_nodes() + coupling_map.num_edges(), env.action_space.n),
            nn.Softmax(dim=-1),
        )

        self.edges = torch.tensor(coupling_map.edge_list()).transpose(1, 0)

    def forward(self, obs: RoutingObsType) -> Tuple[float, torch.Tensor]:
        qubit_interactions = obs['qubit_interactions']
        locked_edges = obs['locked_edges']

        qubit_interactions = self.edge_conv(qubit_interactions, self.edges).view(-1)
        value_input = torch.cat([qubit_interactions, locked_edges])
        policy_input = torch.cat([qubit_interactions, locked_edges])

        value = self.value_net(value_input).item()
        policy = self.policy_net(policy_input)

        return value, policy


class MctsState:
    parent: Optional['MctsState']
    children: List[Optional['MctsState']]

    def __init__(
        self,
        env: LayeredRoutingEnv,
        model: MctsGnn,
        parent: Optional['MctsState'] = None,
        parent_action: Optional[int] = None,
        reward: float = 0.0,
        dirichlet_alpha: float = 0.2,
        prior_fraction: float = 0.25,
    ):
        self.env = env
        self.model = model
        self.parent = parent
        self.parent_action = parent_action
        self.reward = reward

        self.obs = env.current_obs()
        self.estimated_value = 0.0  # TODO

        num_actions = env.action_space.n
        self.q_values = torch.zeros(num_actions)
        self.num_visits = torch.zeros(num_actions)
        self.children = [None] * num_actions
        self.action_mask = self.env.action_masks()

        self.model.eval()
        with torch.no_grad():
            _, self.prior_probabilities = self.model(self.obs)
            self.prior_probabilities += torch.tensor(np.bitwise_not(self.action_mask) * -1e8)
            self.prior_probabilities = torch.flatten(self.prior_probabilities)

        noise = torch.tensor(np.random.dirichlet([dirichlet_alpha] * len(self.prior_probabilities)) * self.action_mask)
        self.prior_probabilities = prior_fraction * self.prior_probabilities + (1 - prior_fraction) * noise

    @property
    def is_terminal(self) -> bool:
        return self.env.terminated

    def select_uct(self) -> int:
        total_visits = torch.sum(self.num_visits).item()
        uct: torch.Tensor = (
            self.q_values +
            self.prior_probabilities * sqrt(total_visits + 1e-3) / (self.num_visits + 1e-3)
        )
        max_indices, = torch.where(uct == uct.max())
        return random.choice(max_indices).item()

    def select_q(self) -> int:
        q_masked: torch.Tensor = self.q_values + torch.tensor(np.bitwise_not(self.action_mask) * -1e8)
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
        max_iters: int = 128,
    ):
        self.discount_factor = discount_factor
        self.max_iters = max_iters

        env = LayeredRoutingEnv(coupling_map, circuit_generator)
        env.reset()

        self.model = MctsGnn(env)
        self.root = MctsState(env, self.model)

    @property
    def env(self) -> LayeredRoutingEnv:
        return self.root.env

    def search(self):
        for _ in range(self.max_iters):
            state = self.root
            depth = 0

            while True:
                depth += 1
                action = state.select_uct()

                # Select move with greatest UCT value until a leaf node is reached
                if state.children[action] is not None:
                    state = state.children[action]
                elif state.is_terminal:
                    break
                else:
                    env = state.env.copy()
                    obs, reward, terminated, *_ = env.step(action)

                    next_state = MctsState(env, self.model, state, action, reward)
                    state.children[action] = next_state
                    state = next_state
                    break

            # Backup
            total_reward = state.estimated_value
            while state.parent is not None:
                total_reward = state.reward + self.discount_factor * total_reward
                state.parent.update_q_value(state.parent_action, total_reward)
                state = state.parent

    def act(self):
        while True:
            self.search()
            action = self.root.select_q()

            if action == self.env.action_space.n - 1:
                self.root = self.root.children[action]
                return
            else:
                self.root = self.root.children[action]

    def replay(self):
        # TODO
        self.model.train()
