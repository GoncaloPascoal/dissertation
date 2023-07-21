
from argparse import ArgumentParser
from typing import Any

import gymnasium as gym
import rustworkx as rx
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env

from action_mask_model import ActionMaskModel
from routing.circuit_gen import RandomCircuitGenerator
from routing.env import TrainingWrapper, QcpRoutingEnv, NoiseGenerationConfig, NoiseConfig


def t_topology() -> rx.PyGraph:
    g = rx.PyGraph()
    g.extend_from_edge_list([(0, 1), (1, 2), (1, 3), (3, 4)])
    return g


def env_creator(env_config: dict[str, Any]) -> gym.Env:
    return TrainingWrapper(
        QcpRoutingEnv(env_config['coupling_map'], 8, noise_config=NoiseConfig(), rllib=True),
        env_config['circuit_generator'],
        noise_generation_config=env_config.get('noise_generation_config'),
    )


def main():
    parser = ArgumentParser('routing_rllib')

    parser.add_argument('-g', '--num-gpus', metavar='G', type=int, default=0,
                        help='number of GPUs used for training (PPO has multi-GPU support)')

    register_env('QcpRoutingEnv', env_creator)

    g = t_topology()
    circuit_generator = RandomCircuitGenerator(g.num_nodes(), 64)
    noise_generation_config = NoiseGenerationConfig(1e-2, 3e-3)

    config = (
        PPOConfig().training(
            clip_param=0.2,
            lr=1e-4,
            lambda_=0.95,
            model={
                'custom_model': ActionMaskModel,
                'fcnet_hiddens': [64, 64, 96],
                'fcnet_activation': 'silu',
            },
            num_sgd_iter=10,
            sgd_minibatch_size=64,
            train_batch_size=12 * 512,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            _enable_learner_api=False,
        )
        .resources(num_gpus=0)
        .rl_module(
            _enable_rl_module_api=False,
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
                'coupling_map': g,
                'circuit_generator': circuit_generator,
                'noise_generation_config': noise_generation_config,
            }
        )
    )

    algorithm = config.build()

    for _ in range(150):
        algorithm.train()

    algorithm.save('models/rllib')


if __name__ == '__main__':
    main()
