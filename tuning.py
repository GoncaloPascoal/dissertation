
import logging

import numpy as np
import ray
from ray import air
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env import EnvContext
from ray.tune import register_env
from ray.tune.logger import TBXLoggerCallback
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from rich import print

from action_mask_model import ActionMaskModel
from parsing import parse_env_config
from routing.circuit_gen import RandomCircuitGenerator
from routing.env_wrapper import TrainingWrapper
from routing.noise import UniformNoiseGenerator
from routing.orchestration import ROUTING_ENV_NAME


def main():
    ray.init(logging_level=logging.ERROR)

    num_qubits = 5
    num_gates = 64
    base_seed = 0

    envs_per_worker = 8

    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(1e6, size=envs_per_worker)

    env_creator = parse_env_config('config/env.yaml')

    def create_circuit_generator(seed: int) -> RandomCircuitGenerator:
        return RandomCircuitGenerator(num_qubits, num_gates, seed=seed)

    def create_noise_generator(seed: int) -> UniformNoiseGenerator:
        return UniformNoiseGenerator(1e-2, 3e-3, seed=seed)

    def create_env(context: EnvContext) -> TrainingWrapper:
        return TrainingWrapper(
            env_creator(),
            create_circuit_generator(seeds[context.vector_index]),
            noise_generator=create_noise_generator(seeds[context.vector_index]),
            recalibration_interval=64,
        )

    register_env(ROUTING_ENV_NAME, create_env)

    resource_config = (
        PPOConfig()
        .training(_enable_learner_api=False)
        .resources(num_gpus=1, num_cpus_for_local_worker=0)
        .rl_module(_enable_rl_module_api=False)
        .rollouts(num_rollout_workers=9)
    )
    trainable = tune.with_resources(PPO, PPO.default_resource_request(resource_config))

    bohb_hyperband = HyperBandForBOHB(max_t=200)
    bohb = TuneBOHB()

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            metric='episode_reward_mean',
            mode='max',
            num_samples=4,
            scheduler=bohb_hyperband,
            search_alg=bohb,
        ),
        param_space=dict(
            # Training
            _enable_learner_api=False,
            train_batch_size=8192,
            grad_clip=0.5,
            model=dict(
                custom_model=ActionMaskModel,
                fcnet_hiddens=tune.sample_from(
                    lambda: [tune.choice([64, 96, 128, 192, 256]).sample()] * tune.randint(1, 3).sample()
                ),
                fcnet_activation=tune.choice(['silu', 'relu', 'tanh']),
            ),
            lr=tune.qloguniform(1e-5, 1e-3, 5e-6),
            sgd_minibatch_size=tune.choice([128, 256, 512]),
            num_sgd_iter=tune.randint(3, 15),
            lambda_=tune.quniform(0.9, 1.0, 0.01),
            clip_param=tune.quniform(0.1, 0.3, 0.01),
            entropy_coeff=tune.quniform(0.0, 1e-2, 2e-4),
            vf_loss_coeff=tune.quniform(0.5, 1.0, 0.02),
            # Debugging
            log_level='ERROR',
            # Environment
            env=ROUTING_ENV_NAME,
            # Fault Tolerance
            recreate_failed_workers=True,
            restart_failed_sub_environments=True,
            # Framework
            framework='torch',
            # Rollouts
            num_rollout_workers=9,
            num_envs_per_worker=envs_per_worker,
            # Resources
            num_gpus=1,
            num_cpus_for_local_worker=0,
            # RL Module
            _enable_rl_module_api=False,
        ),
        run_config=air.RunConfig(
            callbacks=[TBXLoggerCallback()],
        ),
    )

    results = tuner.fit()
    best_result = results.get_best_result()

    print(best_result.config)


if __name__ == '__main__':
    main()
