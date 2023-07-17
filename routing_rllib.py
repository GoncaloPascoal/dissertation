
from typing import Any

import gymnasium as gym
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune import register_env

from action_mask_model import ActionMaskModel
from routing.circuit_gen import RandomCircuitGenerator
from routing.env import TrainingWrapper, QcpRoutingEnv, NoiseGenerationConfig, NoiseConfig
from routing_sb3 import t_topology


def env_creator(env_config: dict[str, Any]) -> gym.Env:
    return TrainingWrapper(
        QcpRoutingEnv(env_config['coupling_map'], 8, noise_config=NoiseConfig(), rllib=True),
        env_config['circuit_generator'],
        noise_generation_config=env_config.get('noise_generation_config'),
    )


def main():
    register_env('QcpRoutingEnv', env_creator)

    g = t_topology()
    circuit_generator = RandomCircuitGenerator(g.num_nodes(), 64)
    noise_generation_config = NoiseGenerationConfig(1e-2, 3e-3)

    config = (
        DQNConfig().training(
            train_batch_size=128,
            model={
                'custom_model': ActionMaskModel,
                'fcnet_hiddens': [64, 64, 96],
                'fcnet_activation': 'silu',
            },
            dueling=False,
            hiddens=[],
        )
        .resources(num_gpus=0)
        .reporting(
            min_train_timesteps_per_iteration=2000,
        )
        .rollouts(
            batch_mode='complete_episodes',
            num_rollout_workers=3,
            num_envs_per_worker=4,
        )
        .fault_tolerance(
            recreate_failed_workers=True,
            restart_failed_sub_environments=True,
        )
        .framework('torch')
        .environment(
            env='QcpRoutingEnv',
            env_config={
                'circuit_generator': circuit_generator,
                'coupling_map': g,
                'noise_generation_config': noise_generation_config,
            }
        )
    )

    algorithm = config.build()

    for _ in range(500):
        algorithm.train()

    algorithm.save('models/rllib')


if __name__ == '__main__':
    main()
