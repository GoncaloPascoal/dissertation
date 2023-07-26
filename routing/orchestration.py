
from collections.abc import Sequence, Collection
from typing import Any, ClassVar, Optional, Self

from qiskit.transpiler import CouplingMap
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune import register_env
from tqdm.rich import tqdm

from action_mask_model import ActionMaskModel
from routing.circuit_gen import CircuitGenerator, DatasetCircuitGenerator
from routing.env import RoutingEnvCreator, RoutingEnv
from routing.env_wrapper import TrainingWrapper, EvaluationWrapper
from routing.noise import NoiseGenerator


class TrainingOrchestrator:
    ENV_NAME: ClassVar[str] = 'RoutingEnv'

    def __init__(
        self,
        env_creator: RoutingEnvCreator,
        circuit_generator: CircuitGenerator,
        *,
        noise_generator: Optional[NoiseGenerator] = None,
        training_iters: int = 1,
        lr: float = 1e-4,
        hidden_layers: Optional[Sequence[int]] = None,
        activation_fn: str = 'silu',
        batch_size: int = 8192,
        minibatch_size: int = 128,
        sgd_iters: int = 10,
        num_gpus: float = 0.0,
        num_workers: int = 2,
        envs_per_worker: int = 4,
    ):
        if hidden_layers is None:
            hidden_layers = [64, 64]

        # TODO: maybe refactor
        def create_env(_config: dict[str, Any]) -> TrainingWrapper:
            return TrainingWrapper(env_creator.create(), circuit_generator, noise_generator, training_iters)

        register_env(TrainingOrchestrator.ENV_NAME, create_env)

        config = (
            PPOConfig()
            .training(
                lr=lr,
                model=dict(
                    custom_model=ActionMaskModel,
                    fcnet_hiddens=hidden_layers,
                    fcnet_activation=activation_fn,
                ),
                train_batch_size=batch_size,
                lambda_=0.95,
                sgd_minibatch_size=minibatch_size,
                num_sgd_iter=sgd_iters,
                vf_loss_coeff=0.5,
                clip_param=0.2,
                grad_clip=0.5,
                _enable_learner_api=False,
            )
            .environment(env=TrainingOrchestrator.ENV_NAME)
            .fault_tolerance(
                recreate_failed_workers=True,
                restart_failed_sub_environments=True,
            )
            .framework('torch')
            .resources(num_gpus=num_gpus)
            .rl_module(_enable_rl_module_api=False)
            .rollouts(
                batch_mode='complete_episodes',
                num_rollout_workers=num_workers,
                num_envs_per_worker=envs_per_worker,
            )
        )

        self.algorithm = config.build()

    @classmethod
    def from_checkpoint(cls, path: str) -> Self:
        obj = cls.__new__(cls)
        super(TrainingOrchestrator, obj).__init__()
        obj.algorithm = PPO.from_checkpoint(path)
        return obj

    def train(self, iters: int):
        for _ in range(iters):
            self.algorithm.train()

    def save(self, path: str) -> str:
        return self.algorithm.save(path)


class EvaluationOrchestrator:
    reliability_map: dict[tuple[int, int], float]

    def __init__(
        self,
        algorithm: PPO,
        env: RoutingEnv,
        circuit_generator: CircuitGenerator,
        *,
        noise_generator: Optional[NoiseGenerator] = None,
        evaluation_iters: int = 20,
        num_episodes: Optional[int] = None,
        routing_methods: str | Collection[str] = 'sabre',
        use_tqdm: bool = False,
    ):
        if num_episodes is None:
            if isinstance(circuit_generator, DatasetCircuitGenerator):
                num_episodes = len(circuit_generator.dataset)
            else:
                num_episodes = 100

        self.algorithm = algorithm
        self.eval_env = EvaluationWrapper(
            env,
            circuit_generator,
            noise_generator=noise_generator,
            evaluation_iters=evaluation_iters,
        )

        self.num_episodes = num_episodes
        self.routing_methods = [routing_methods] if isinstance(routing_methods, str) else list(routing_methods)
        self.use_tqdm = use_tqdm

        self.env = self.eval_env.env
        self.initial_layout = env.qubit_to_node.tolist()
        self.qiskit_coupling_map = CouplingMap(env.coupling_map.to_directed().edge_list())

        self.reliability_map = {}
        for edge, error_rate in zip(env.coupling_map, env.error_rates):  # type: ignore
            reliability = 1.0 - error_rate
            self.reliability_map[edge] = reliability
            self.reliability_map[edge[::-1]] = reliability

        self.reset_metrics()

    def reset_metrics(self):
        raise NotImplementedError

    def evaluate(self):
        iterable = range(self.num_episodes)
        if self.use_tqdm:
            iterable = tqdm(iterable)

        for _ in iterable:
            pass
