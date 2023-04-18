
import itertools
from typing import Any, Dict, List, Sequence, Tuple, Type

import gymnasium as gym
from gymnasium import spaces

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from ray.rllib.algorithms import AlgorithmConfig, Algorithm
from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.utils.schedules import PiecewiseSchedule

from gradient_based import gradient_based_hst_weighted
from utils import NativeInstruction


class CircuitEnv(gym.Env):
    @staticmethod
    def _is_valid_action(action: NativeInstruction) -> bool:
        return (action.instruction.name not in {'delay', 'id', 'reset'} and
                action.instruction.num_clbits == 0)

    def __init__(self, context: Dict[str, Any]):
        self.u = context['u']
        self.max_depth = context['max_depth']
        self.cx_penalty_weight = context.get('cx_penalty_weight', 0.0)
        self.circuit_actions: List[NativeInstruction] = [
            a for a in context['actions'] if CircuitEnv._is_valid_action(a)
        ]

        def grouper(action: NativeInstruction):
            return action.instruction.name
        groups = itertools.groupby(sorted(context['actions'], key=grouper), key=grouper)

        qubit_set = set()
        action_observation_map = {}

        for i, (key, group) in enumerate(groups):
            qubit_set.update(action.qubits for action in group)
            action_observation_map[key] = i

        self.action_observation_map = action_observation_map
        self.qubit_observation_map = {qubits: i for i, qubits in enumerate(qubit_set)}

        self.action_space = spaces.Discrete(len(self.circuit_actions))
        self.observation_space = spaces.Tuple((
            spaces.Discrete(len(self.action_observation_map) + 1),
            spaces.Discrete(len(self.qubit_observation_map) + 1),
            spaces.Discrete(self.max_depth + 1)
        ))
        self.reward_range = (-self.cx_penalty_weight, 1.0)

        self.v = QuantumCircuit(self.u.num_qubits)
        self.num_params = 0
        self.current_observation = (0, 0, 0)

    @classmethod
    def from_args(
        cls,
        u: QuantumCircuit,
        actions: Sequence[NativeInstruction],
        max_depth: int,
        cx_penalty_weight: float = 0.0,
    ) -> 'CircuitEnv':
        return cls({
            'u': u,
            'actions': actions,
            'max_depth': max_depth,
            'cx_penalty_weight': cx_penalty_weight,
        })

    def step(self, action: int):
        instruction, qubits = self.circuit_actions[action - 1]

        # Give unique name to each instruction parameter
        if instruction.params:
            instruction = instruction.copy()
            new_params = []
            for _ in instruction.params:
                new_params.append(Parameter(f'p{self.num_params}'))
                self.num_params += 1
            instruction.params = new_params

        self.v.append(instruction, qubits)
        self.current_observation = (
            self.action_observation_map[instruction.name],
            self.qubit_observation_map[qubits],
            self.depth
        )

        reward = 0.0
        terminated = self.depth == self.max_depth
        if terminated:
            # Optimize continuous parameters using gradient descent
            params, cost = gradient_based_hst_weighted(self.u, self.v)
            reward = self._reward(cost)

        return self.current_observation, reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.v = QuantumCircuit(self.u.num_qubits)
        self.num_params = 0
        self.current_observation = (len(self.action_observation_map), len(self.qubit_observation_map), self.depth)

        return self.current_observation, {}

    @property
    def depth(self):
        return len(self.v)

    def _reward(self, cost: float) -> float:
        instructions = self.v.data
        n_total = len(instructions)
        n_cx = len([i for i in instructions if i.operation.name == 'cx'])
        cx_penalty = 0.0 if n_total == 0 else self.cx_penalty_weight * n_cx / n_total

        return 1 - cost - cx_penalty


def dqn(
    algo_class: Type[DQN] = ApexDQN,
    learning_rate: float = 0.02,
    discount_factor: float = 0.9,
    batch_size: int = 128,
    double_q: bool = True,
) -> DQNConfig:
    return DQNConfig(algo_class).training(
        lr=learning_rate,
        gamma=discount_factor,
        train_batch_size=batch_size,
        double_q=double_q,
    )


def sac(
    learning_rate: float = 0.02,
    discount_factor: float = 0.9,
    batch_size: int = 32,
) -> SACConfig:
    return SACConfig().training(
        lr=learning_rate,
        gamma=discount_factor,
        train_batch_size=batch_size,
    )


def build_algorithm(
    config: AlgorithmConfig,
    u: QuantumCircuit,
    actions: Sequence[NativeInstruction],
    max_depth: int,
    epsilon_greedy_episodes: Sequence[Tuple[float, int]],
    cx_penalty_weight: float = 0.0,
    num_gpus: int = 0,
) -> Algorithm:
    # Construct epsilon greedy exploration schedule as a step function
    endpoints = []
    total_episodes = 0
    for epsilon, num_episodes in epsilon_greedy_episodes:
        endpoints.append((total_episodes * max_depth, epsilon))
        total_episodes += num_episodes
        endpoints.append((total_episodes * max_depth, epsilon))
    epsilon_schedule = PiecewiseSchedule(endpoints, outside_value=epsilon_greedy_episodes[-1][0])

    (
        config.rollouts(
            batch_mode='complete_episodes',
            num_rollout_workers=6,
            ignore_worker_failures=True,
            recreate_failed_workers=True,
            restart_failed_sub_environments=True,
        )
        .resources(num_gpus=num_gpus)
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
                'epsilon_schedule': epsilon_schedule,
            }
        )
    )

    return config.build()


def main():
    from vqc_double_q import build_native_instructions

    u = QuantumCircuit(2)
    u.ch(0, 1)

    actions = build_native_instructions(2)

    algo = build_algorithm(dqn(), u, actions, 6, [(1.0, 400), (0.5, 200), (0.1, 200)])

    for _ in range(10):
        algo.train()

    env = CircuitEnv.from_args(u, actions, 6)
    observation, _ = env.reset()

    while True:
        action = algo.compute_single_action(observation, explore=False)
        observation, reward, terminated, _, _ = env.step(action)

        if terminated:
            print(reward)
            break

    print(env.v.draw())


if __name__ == '__main__':
    main()
