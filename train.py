
from argparse import ArgumentParser
from typing import Any

import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env

from action_mask_model import ActionMaskModel
from routing.circuit_gen import RandomCircuitGenerator
from routing.env import CircuitMatrixRoutingEnv
from routing.env_wrapper import TrainingWrapper
from routing.noise import NoiseConfig, UniformNoiseGenerator
from routing.topology import t_topology


def env_creator(env_config: dict[str, Any]) -> gym.Env:
    return TrainingWrapper(
        CircuitMatrixRoutingEnv(env_config['coupling_map'], depth=env_config['depth'], noise_config=NoiseConfig()),
        env_config['circuit_generator'],
        noise_generator=env_config.get('noise_generator'),
        training_iters=env_config['training_iters'],
    )


def main():
    parser = ArgumentParser('train', description='Noise-Resilient Reinforcement Learning Strategies for Quantum '
                                                 'Compiling (model training script)')

    parser.add_argument('-m', '--model', metavar='M', help='model name', required=True)
    parser.add_argument('-d', '--depth', metavar='N', type=int, default=8, help='depth of circuit observations')
    parser.add_argument('-g', '--num-gpus', metavar='N', type=float, default=0.0,
                        help='number of GPUs used for training (PPO has multi-GPU support), can be fractional')
    parser.add_argument('-w', '--workers', metavar='N', type=int, default=2, help='number of rollout workers')
    parser.add_argument('-e', '--envs-per-worker', metavar='N', type=int, default=4,
                        help='number of environments per rollout worker')
    parser.add_argument('-i', '--iters', metavar='N', type=int, default=100, help='training iterations')
    parser.add_argument('--circuit-size', metavar='N', type=int, default=64, help='random circuit gate count')
    parser.add_argument('--training-episodes', metavar='N', type=int, default=1, help='training episodes per circuit')
    parser.add_argument('--batch-size', metavar='N', type=int, default=8192, help='training batch size')
    parser.add_argument('--lr', metavar='N', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--minibatch-size', metavar='N', type=int, default=128,
                        help='stochastic gradient descent minibatch size')
    parser.add_argument('--net-arch', metavar='N', nargs='+', type=int, default=[64, 64, 96],
                        help='neural network architecture (number of nodes in each hidden FC layer)')
    parser.add_argument('--sgd-iters', metavar='N', type=int, default=128,
                        help='stochastic gradient descent iterations per batch')

    args = parser.parse_args()

    register_env('CircuitMatrixRoutingEnv', env_creator)

    g = t_topology()
    circuit_generator = RandomCircuitGenerator(g.num_nodes(), args.circuit_size)
    noise_generator = UniformNoiseGenerator(1e-2, 3e-3)

    config = (
        PPOConfig().training(
            clip_param=0.2,
            lr=args.lr,
            lambda_=0.95,
            model={
                'custom_model': ActionMaskModel,
                'fcnet_hiddens': args.net_arch,
                'fcnet_activation': 'silu',
            },
            num_sgd_iter=args.sgd_iters,
            sgd_minibatch_size=args.minibatch_size,
            train_batch_size=args.batch_size,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            _enable_learner_api=False,
        )
        .resources(
            num_gpus=args.num_gpus,
        )
        .rl_module(
            _enable_rl_module_api=False,
        )
        .rollouts(
            batch_mode='complete_episodes',
            num_rollout_workers=args.workers,
            num_envs_per_worker=args.envs_per_worker,
        )
        .fault_tolerance(
            recreate_failed_workers=True,
            restart_failed_sub_environments=True,
        )
        .framework('torch')
        .environment(
            env='CircuitMatrixRoutingEnv',
            env_config={
                'coupling_map': g,
                'depth': args.depth,
                'circuit_generator': circuit_generator,
                'noise_generator': noise_generator,
                'training_iters': args.training_episodes,
            }
        )
    )

    algorithm = config.build()

    for _ in range(args.iters):
        algorithm.train()

    algorithm.save(f'models/{args.model}')


if __name__ == '__main__':
    main()
