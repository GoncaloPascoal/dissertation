
import argparse
import logging
from argparse import ArgumentParser

import ray

from narlsqr.parsing import parse_eval_config

def main():
    parser = ArgumentParser('eval', description='Noise-Adaptive Reinforcement Learning Strategies for Qubit '
                                                'Routing (model evaluation script)')

    parser.add_argument('env_config', help='environment configuration file')
    parser.add_argument('eval_config', help='evaluation configuration file')
    parser.add_argument('model_dir', help='path to trained model')
    parser.add_argument('save_to',help='path where metrics should be saved to')

    parser.add_argument('-e', '--evaluation-episodes', metavar='N', type=int, default=argparse.SUPPRESS,
                        help='evaluation episodes per circuit')
    parser.add_argument('-r', '--routing-methods', nargs='+', choices=['basic', 'stochastic', 'sabre'],
                        default=argparse.SUPPRESS, help='routing method(s) for Qiskit compiler')
    parser.add_argument('-s', '--seed', type=int, default=argparse.SUPPRESS, help='seed for random number generators')
    parser.add_argument('--num-circuits', metavar='N', type=int, default=argparse.SUPPRESS,
                        help='number of (random) evaluation circuits')
    parser.add_argument('--use-tqdm', action='store_const', const=True, default=argparse.SUPPRESS,
                        help='show a progress bar using tqdm')
    parser.add_argument('--stochastic', action='store_true', default=argparse.SUPPRESS,
                        help='Use stochastic policy')
    parser.add_argument('--deterministic', action='store_false', dest='stochastic', default=argparse.SUPPRESS,
                        help='Use deterministic policy (evaluation_iters will be set to 1)')

    args = vars(parser.parse_args())

    ray.init(logging_level=logging.ERROR)

    env_config = args.pop('env_config')
    eval_config = args.pop('eval_config')
    model_dir = args.pop('model_dir')
    save_to = args.pop('save_to')

    orchestrator = parse_eval_config(env_config, eval_config, model_dir, override_args=args)
    orchestrator.evaluate()
    orchestrator.metrics_analyzer.pickle(save_to)


if __name__ == '__main__':
    main()

