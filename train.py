
import argparse
import logging
from argparse import ArgumentParser
from typing import Any

import gymnasium as gym
import ray

from parsing import parse_train_config
from routing.env import CircuitMatrixRoutingEnv
from routing.env_wrapper import TrainingWrapper
from routing.noise import NoiseConfig


def env_creator(env_config: dict[str, Any]) -> gym.Env:
    return TrainingWrapper(
        CircuitMatrixRoutingEnv(env_config['coupling_map'], depth=env_config['depth'], noise_config=NoiseConfig()),
        env_config['circuit_generator'],
        noise_generator=env_config.get('noise_generator'),
        episodes_per_circuit=env_config['episodes_per_circuit'],
    )


def main():
    parser = ArgumentParser('train', description='Noise-Resilient Reinforcement Learning Strategies for Quantum '
                                                 'Compiling (model training script)')

    parser.add_argument('env_config', help='environment configuration file')
    parser.add_argument('train_config', help='training configuration file')

    parser.add_argument('-m', '--model-dir', metavar='M', help='directory to save the model', required=True)
    parser.add_argument('-i', '--iters', metavar='N', type=int, help='training iterations', required=True)
    parser.add_argument('-g', '--num-gpus', metavar='N', type=float, default=argparse.SUPPRESS,
                        help='number of GPUs used for training (PPO has multi-GPU support), can be fractional')
    parser.add_argument('-w', '--num-workers', metavar='N', type=int, default=argparse.SUPPRESS,
                        help='number of rollout workers')
    parser.add_argument('-e', '--envs-per-worker', metavar='N', type=int, default=argparse.SUPPRESS,
                        help='number of environments per rollout worker')

    args = vars(parser.parse_args())

    ray.init(logging_level=logging.ERROR)

    env_config = args.pop('env_config')
    train_config = args.pop('train_config')
    iters = args.pop('iters')
    model_dir = args.pop('model_dir')

    orchestrator = parse_train_config(env_config, train_config, override_args=args)

    orchestrator.train(iters)
    orchestrator.save(model_dir)


if __name__ == '__main__':
    main()
