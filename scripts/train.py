
import argparse
import logging
import os
import pathlib
from argparse import ArgumentParser

import ray

from narlsqr.orchestration import get_latest_checkpoint_dir, is_checkpoint
from narlsqr.parsing import parse_train_config

def main():
    parser = ArgumentParser(
        'train',
        description='Noise-Adaptive Reinforcement Learning Strategies for Qubit Routing (model training script)',
    )

    parser.add_argument('env_config', help='environment configuration file')
    parser.add_argument('train_config', help='training configuration file')

    parser.add_argument(
        '-m', '--model-dir', metavar='P',
        help='directory to save or load the model (can be a path to a checkpoint or the root folder, in which case '
             'the latest checkpoint will be loaded)',
        required=True,
    )
    parser.add_argument('-i', '--iters', metavar='N', type=int, help='training iterations', required=True)
    parser.add_argument('-g', '--num-gpus', metavar='N', type=float, default=argparse.SUPPRESS,
                        help='number of GPUs used for training (PPO has multi-GPU support), can be fractional')
    parser.add_argument('-w', '--num-workers', metavar='N', type=int, default=argparse.SUPPRESS,
                        help='number of rollout workers')
    parser.add_argument('-e', '--envs-per-worker', metavar='N', type=int, default=argparse.SUPPRESS,
                        help='number of environments per rollout worker')
    parser.add_argument('-s', '--seed', type=int, default=argparse.SUPPRESS, help='seed for random number generators')

    args = vars(parser.parse_args())

    ray.init(logging_level=logging.ERROR)

    env_config: str = args.pop('env_config')
    train_config: str = args.pop('train_config')
    iters: int = args.pop('iters')
    model_dir: str = args.pop('model_dir')

    if os.path.exists(model_dir):
        if is_checkpoint(model_dir):
            checkpoint_dir = model_dir
            model_dir = str(pathlib.Path(model_dir).parent)
        else:
            checkpoint_dir = str(get_latest_checkpoint_dir(model_dir))
    else:
        checkpoint_dir = None

    orchestrator = parse_train_config(env_config, train_config, checkpoint_dir=checkpoint_dir, override_args=args)

    orchestrator.train(iters)
    orchestrator.save(model_dir)


if __name__ == '__main__':
    main()
