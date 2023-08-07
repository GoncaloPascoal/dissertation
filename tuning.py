
from ray import air
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.schedulers import PopulationBasedTraining
from rich import print

from action_mask_model import ActionMaskModel
from parsing import parse_env_config
from routing.circuit_gen import RandomCircuitGenerator
from routing.noise import UniformNoiseGenerator
from routing.orchestration import register_routing_env, ROUTING_ENV_NAME


def main():
    env_creator = parse_env_config('config/env.yaml')
    circuit_generator = RandomCircuitGenerator(5, 64)
    noise_generator = UniformNoiseGenerator(1e-2, 3e-3)

    register_routing_env(
        env_creator,
        circuit_generator,
        noise_generator=noise_generator,
    )

    hyperparam_mutations = dict(
        lr=tune.qloguniform(1e-5, 1e-3, 5e-6),
    )

    pbt = PopulationBasedTraining(
        hyperparam_mutations=hyperparam_mutations
    )

    tuner = tune.Tuner(
        PPO,
        tune_config=tune.TuneConfig(
            metric='episode_reward_mean',
            mode='max',
            scheduler=pbt,
        ),
        param_space=dict(
            # Training
            _enable_learner_api=False,
            lr=tune.qloguniform(1e-5, 1e-3, 5e-6),
            model=dict(
                custom_model=ActionMaskModel,
                fcnet_hiddens=[64, 64, 96],
                fcnet_activation='silu',
            ),
            train_batch_size=8192,
            lambda_=0.95,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
            vf_loss_coeff=0.5,
            clip_param=0.2,
            grad_clip=0.5,
            # Environment
            env=ROUTING_ENV_NAME,
            # Fault Tolerance
            recreate_failed_workers=True,
            restart_failed_sub_environments=True,
            # Framework
            framework='torch',
            # Rollouts
            num_rollout_workers=6,
            num_envs_per_worker=8,
            # Resources
            num_gpus=0.5,
            num_cpus_for_local_worker=0,
            # RL Module
            _enable_rl_module_api=False,
        ),
        run_config=air.RunConfig(
            stop=dict(
                training_iteration=100,
            ),
        ),
    )

    results = tuner.fit()
    best_result = results.get_best_result()

    print({k: v for k, v in best_result.config.items() if k in hyperparam_mutations})


if __name__ == '__main__':
    main()
