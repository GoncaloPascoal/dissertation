
from argparse import ArgumentParser

from ray.rllib.algorithms.ppo import PPOConfig


def main():
    parser = ArgumentParser()
    parser.add_argument('-g', '--num-gpus', metavar='N', type=float, default=0.0,
                        help='number of GPUs used for training (PPO has multi-GPU support), can be fractional')
    parser.add_argument('-w', '--workers', metavar='N', type=int, default=4, help='number of rollout workers')
    parser.add_argument('-e', '--envs-per-worker', metavar='N', type=int, default=1,
                        help='number of environments per rollout worker')
    parser.add_argument('-i', '--iters', metavar='N', type=int, default=50, help='training iterations')

    args = parser.parse_args()

    config = (
        PPOConfig()
        .training(
            _enable_learner_api=False,
        )
        .environment('CartPole-v1')
        .fault_tolerance(
            recreate_failed_workers=True,
            restart_failed_sub_environments=True,
        )
        .framework('torch')
        .resources(num_gpus=args.num_gpus)
        .rl_module(
            _enable_rl_module_api=False,
        )
        .rollouts(
            num_rollout_workers=args.workers,
            num_envs_per_worker=args.envs_per_worker,
        )
    )

    algo = config.build()

    for _ in range(args.iters):
        algo.train()


if __name__ == '__main__':
    main()
