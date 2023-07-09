from typing import Any

import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env

from action_mask_model import ActionMaskModel
from routing.circuit_gen import RandomCircuitGenerator
from routing.env import TrainingWrapper, QcpRoutingEnv, NoiseConfig
from routing_sb3 import t_topology


def env_creator(env_config: dict[str, Any]) -> gym.Env:
    return TrainingWrapper(
        env_config['circuit_generator'],
        env_config['coupling_map'],
        QcpRoutingEnv,
        8,
        noise_config=env_config.get('noise_config'),
        rllib=True,
    )


def main():
    register_env('QcpRoutingEnv', env_creator)

    g = t_topology()
    circuit_generator = RandomCircuitGenerator(g.num_nodes(), 64)
    noise_config = NoiseConfig(1e-2, 3e-3)

    config = (
        PPOConfig().training(
            train_batch_size=8192,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
            model={
                'custom_model': ActionMaskModel,
                'fcnet_hiddens': [64, 64, 96],
                'fcnet_activation': 'silu',
            },
        )
        .rollouts(
            batch_mode='complete_episodes',
            num_rollout_workers=4,
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
                'noise_config': noise_config,
            }
        )
    )

    algorithm = config.build()

    for _ in range(128):
        algorithm.train()

    algorithm.save('models/rllib')


if __name__ == '__main__':
    main()
