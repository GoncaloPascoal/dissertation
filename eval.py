
import argparse
import logging
from argparse import ArgumentParser

import ray

from parsing import parse_eval_config


def main():
    parser = ArgumentParser('eval', description='Noise-Resilient Reinforcement Learning Strategies for Quantum '
                                                'Compiling (model evaluation script)')

    parser.add_argument('env_config', help='environment configuration file')
    parser.add_argument('eval_config', help='evaluation configuration file')

    parser.add_argument('-c', '--checkpoint-path', metavar='P', default=argparse.SUPPRESS,
                        help='path to model checkpoint')
    parser.add_argument('-i', '--evaluation-iters', metavar='N', type=int, default=argparse.SUPPRESS,
                        help='evaluation iterations per circuit')
    parser.add_argument('-r', '--routing-methods', nargs='+', choices=['basic', 'stochastic', 'sabre'],
                        default=argparse.SUPPRESS, help='routing method(s) for Qiskit compiler')
    parser.add_argument('-s', '--seed', type=int, default=argparse.SUPPRESS, help='seed for random number generators')
    parser.add_argument('--num-circuits', metavar='N', type=int, default=argparse.SUPPRESS,
                        help='number of (random) evaluation circuits')
    parser.add_argument('--use-tqdm', action='store_const', const=True, default=argparse.SUPPRESS,
                        help='show a progress bar using tqdm')

    args = vars(parser.parse_args())

    ray.init(logging_level=logging.ERROR)

    env_config = args.pop('env_config')
    eval_config = args.pop('eval_config')

    orchestrator = parse_eval_config(env_config, eval_config, override_args=args)
    orchestrator.evaluate()


if __name__ == '__main__':
    main()

