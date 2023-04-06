
import itertools
import collections
from typing import Any, Dict, List, Sequence, Tuple, Type

import gymnasium as gym
from gymnasium import spaces

from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.utils.schedules import PiecewiseSchedule

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Parameter

from gradient_based import gradient_based_hst_weighted

from rich import print


# Use collections.namedtuple instead of typing.NamedTuple to avoid exception cased by cloudpickle (Ray)
CircuitAction: Type[Tuple[Instruction, Tuple[int, ...]]] = collections.namedtuple(
    'CircuitAction',
    ['instruction', 'qubits']
)

class CircuitEnv(gym.Env):
    @staticmethod
    def _is_valid_action(action: CircuitAction) -> bool:
        return (action.instruction.name not in {'delay', 'id', 'reset'} and
                action.instruction.num_clbits == 0)

    def __init__(self, context: Dict[str, Any]):
        self.u = context['u']
        self.max_depth = context['max_depth']
        self.cx_penalty_weight = context.get('cx_penalty_weight', 0.0)
        self.circuit_actions: List[CircuitAction] = [
            a for a in context['actions'] if CircuitEnv._is_valid_action(a)
        ]
        num_circuit_actions = len(self.circuit_actions)

        def grouper(action: CircuitAction):
            return action.instruction.name
        groups = itertools.groupby(sorted(context['actions'], key=grouper), key=grouper)

        qubit_set = set()
        action_observation_map = {}

        for i, (key, group) in enumerate(groups):
            qubit_set.update(action.qubits for action in group)
            action_observation_map[key] = i

        self.action_observation_map = action_observation_map
        self.qubit_observation_map = {qubits: i for i, qubits in enumerate(qubit_set)}

        self.action_space = spaces.Discrete(num_circuit_actions + 1)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(len(self.action_observation_map) + 1),
            spaces.Discrete(len(self.qubit_observation_map) + 1),
            spaces.Discrete(self.max_depth + 1)
        ))
        self.reward_range = (-self.cx_penalty_weight, 1.0)

        self.v = QuantumCircuit(self.u.num_qubits)
        self.num_params = 0

    def step(self, action: int):
        circuit_finished = action == 0

        if not circuit_finished:
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
        terminated = self.depth == self.max_depth or circuit_finished
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


def double_dqn(
    u: QuantumCircuit,
    actions: Sequence[CircuitAction],
    max_depth: int,
    epsilon_greedy_episodes: Sequence[Tuple[float, int]],
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
        .rollouts(
            num_rollout_workers=6,
            ignore_worker_failures=True,
            recreate_failed_workers=True,
            restart_failed_sub_environments=True,
        )
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
    from qiskit.circuit.library import SXGate, RZGate, CXGate, QFT

    u = QuantumCircuit(2)
    u.swap(0, 1)

    sx = SXGate()
    rz = RZGate(Parameter('a'))
    cx = CXGate()

    actions = [
        CircuitAction(sx, (0,)),
        CircuitAction(sx, (1,)),
        CircuitAction(rz, (0,)),
        CircuitAction(rz, (1,)),
        CircuitAction(cx, (0, 1)),
        CircuitAction(cx, (1, 0)),
    ]

    algo = double_dqn(
        u,
        actions,
        3,
        [(1.0, 100), (0.5, 50), (0.1, 50)],
    )

    for _ in range(10):
        algo.train()
