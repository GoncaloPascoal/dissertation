
from typing import Dict, List, Tuple

import gymnasium as gym
from gymnasium import spaces

from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.utils.schedules import PiecewiseSchedule

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction

from gradient_based import gradient_based_hst_weighted

from qiskit.circuit import Parameter
from qiskit.circuit.library import SXGate, RZGate, CXGate

from rich import print

CircuitAction = Tuple[Instruction, Tuple[int, ...]]

class CircuitEnv(gym.Env):
    @staticmethod
    def _is_valid_action(action: CircuitAction) -> bool:
        return (action[0].name not in {'delay', 'id', 'reset'} and
            action[0].num_clbits == 0)

    def __init__(self, context: Dict):
        self.u = context['u']
        self.v = QuantumCircuit(self.u.num_qubits)
        self.num_params = 0
        self.depth = 0
        self.max_depth = context['max_depth']
        self.cx_penalty_weight = context.get('cx_penalty_weight', 0.0)

        self.actions: List[CircuitAction] = list(filter(
            CircuitEnv._is_valid_action,
            context['actions']
        ))
        num_actions = len(self.actions)

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(num_actions + 1, start=-1), spaces.Discrete(self.max_depth + 1))
        )

    def step(self, action: int):
        instruction, qubits = self.actions[action]

        # Give unique name to each instruction parameter
        if instruction.params:
            instruction = instruction.copy()
            new_params = []
            for _ in instruction.params:
                new_params.append(Parameter(f'p{self.num_params}'))
                self.num_params += 1
            instruction.params = new_params

        self.v.append(instruction, qubits)
        self.depth += 1

        reward = 0.0
        terminated = self.depth == self.max_depth
        if terminated:
            # Optimize continuous parameters using gradient descent
            params, cost = gradient_based_hst_weighted(self.u, self.v)
            reward = self._reward(cost)

        return (action, self.depth), reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.v = QuantumCircuit(self.u.num_qubits)
        self.num_params = 0
        self.depth = 0

        return (-1, self.depth), {}

    def _reward(self, cost: float) -> float:
        instructions = self.v.data
        n_total = len(instructions)
        n_cx = len([i for i in instructions if i.operation.name == 'cx'])
        cx_penalty = self.cx_penalty_weight * n_cx / n_total

        return 1 - cost - cx_penalty

def double_dqn(
    u: QuantumCircuit,
    actions: List[CircuitAction],
    max_depth: int,
    epsilon_greedy_episodes: List[Tuple[float, int]],
    learning_rate: float = 0.02,
    discount_factor: float = 0.9,
    batch_size: int = 32,
    cx_penalty_weight: float = 0.0,
) -> DQN:

    # Construct epsilon greedy exploration schedule as a step function
    endpoints = []
    total_episodes = 0
    for epsilon, num_episodes in epsilon_greedy_episodes:
        endpoints.append((total_episodes * max_depth, epsilon))
        total_episodes += num_episodes
        endpoints.append((total_episodes * max_depth, epsilon))

    config = (DQNConfig()
        .training(
            lr=learning_rate,
            gamma=discount_factor,
            train_batch_size=batch_size,
            double_q=True,
        )
        .rollouts(num_rollout_workers=6)
        .resources(num_gpus=0)
        .framework('torch')
        .debugging(log_level='INFO')
        .environment(
            env=CircuitEnv,
            env_config={
                'u': u,
                'actions': actions,
                'max_depth': max_depth,
                'cx_penalty_weight': cx_penalty_weight,
            },
        )
        .exploration(
            exploration_config={
                'type': 'EpsilonGreedy',
                # TODO: Fix problem with epsilon schedule
                # 'epsilon_schedule': PiecewiseSchedule(endpoints),
            }
        )
    )

    return ApexDQN(config)


if __name__ == '__main__':
    u = QuantumCircuit(2)
    u.h(0)

    sx = SXGate()
    rz = RZGate(Parameter('a'))
    cx = CXGate()

    actions = [
        (sx, (0,)),
        (sx, (1,)),
        (rz, (0,)),
        (rz, (1,)),
        (cx, (0, 1)),
        (cx, (1, 0)),
    ]

    model = double_dqn(
        u,
        actions,
        3,
        [(1.0, 100), (0.5, 50), (0.1, 50)],
    )
