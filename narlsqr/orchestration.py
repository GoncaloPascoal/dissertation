
import copy
import ctypes
import os.path
import time
from collections.abc import Collection, Set
from dataclasses import dataclass, field
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Final, Optional, Self, cast

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.models import BackendProperties
from qiskit.transpiler import CouplingMap
from ray.rllib import Policy
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env import EnvContext
from ray.tune import register_env
from ray.tune.logger import UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR
from tqdm.rich import tqdm

from narlsqr.analysis import MetricsAnalyzer
from narlsqr.env import RoutingEnv
from narlsqr.env.wrappers import EvaluationWrapper, TrainingWrapper
from narlsqr.generators.circuit import CircuitGenerator, DatasetCircuitGenerator
from narlsqr.generators.noise import NoiseGenerator
from narlsqr.rllib.action_mask_model import ActionMaskModel
from narlsqr.rllib.callbacks import RoutingCallbacks
from narlsqr.utils import Factory, circuit_reliability, seed_default_generators, IBM_BASIS_GATES

ROUTING_ENV_NAME: Final[str] = 'RoutingEnv'


@dataclass
class CheckpointConfig:
    model_dir: str
    interval: int = field(default=25, kw_only=True)

class TrainingOrchestrator:
    algorithm: PPO

    def __init__(
        self,
        env_creator: Factory[RoutingEnv],
        circuit_generator: CircuitGenerator,
        noise_generator: NoiseGenerator,
        *,
        recalibration_interval: int = 64,
        episodes_per_circuit: int = 1,
        checkpoint_config: Optional[CheckpointConfig] = None,
        gamma: float = 0.99,
        lr: float = 1e-4,
        lr_schedule: Optional[list[list[int | float]]] = None,
        gae_lambda: float = 0.95,
        hidden_layers: Optional[list[int]] = None,
        activation_fn: str = 'silu',
        embedding_dim: Optional[int] = None,
        batch_size: int = 8192,
        minibatch_size: int = 128,
        sgd_iters: int = 10,
        vf_loss_coeff: float = 0.5,
        entropy_coeff: float = 0.0,
        seed: Optional[int] = None,
        evaluation_interval: Optional[int] = None,
        evaluation_duration: int = 128,
        base_logging_dir: Optional[str] = None,
        num_gpus: float = 0.0,
        num_workers: int = 2,
        envs_per_worker: int = 4,
    ):
        hidden_layers = [64, 64] if hidden_layers is None else hidden_layers
        base_logging_dir = DEFAULT_RESULTS_DIR if base_logging_dir is None else base_logging_dir

        if seed is not None:
            seed_default_generators(seed)

        TrainingOrchestrator.register_routing_env(
            env_creator,
            circuit_generator,
            noise_generator,
            recalibration_interval=recalibration_interval,
            episodes_per_circuit=episodes_per_circuit,
        )

        env_name = env_creator().name
        time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logging_dir = f'{env_name}_{time_str}'

        def logger_creator(logger_config: dict) -> UnifiedLogger:
            if not os.path.exists(base_logging_dir):
                os.makedirs(base_logging_dir)

            return UnifiedLogger(
                logger_config,
                os.path.join(base_logging_dir, logging_dir),
            )

        config = (
            PPOConfig()
            .training(
                gamma=gamma,
                lr=lr,
                lr_schedule=lr_schedule,
                model=dict(
                    custom_model=ActionMaskModel,
                    custom_model_config=dict(
                        embedding_dim=embedding_dim,
                    ),
                    fcnet_hiddens=hidden_layers,
                    fcnet_activation=activation_fn,
                ),
                train_batch_size=batch_size,
                lambda_=gae_lambda,
                sgd_minibatch_size=minibatch_size,
                num_sgd_iter=sgd_iters,
                vf_loss_coeff=vf_loss_coeff,
                entropy_coeff=entropy_coeff,
                kl_coeff=0.0,
                clip_param=0.2,
                grad_clip=0.5,
            )
            .callbacks(RoutingCallbacks)
            .debugging(
                logger_creator=logger_creator,
                log_level='ERROR',
                seed=seed,
            )
            .environment(
                env=ROUTING_ENV_NAME,
                env_config=dict(seed=seed),
            )
            .evaluation(
                evaluation_interval=evaluation_interval,
                evaluation_duration=evaluation_duration,
            )
            .experimental(_enable_new_api_stack=False)
            .fault_tolerance(
                recreate_failed_workers=True,
                restart_failed_sub_environments=True,
            )
            .framework('torch')
            .reporting(
                metrics_num_episodes_for_smoothing=1000,
            )
            .resources(
                num_gpus=num_gpus,
                num_cpus_for_local_worker=0 if num_gpus > 0.0 else None,
            )
            .rollouts(
                num_rollout_workers=num_workers,
                num_envs_per_worker=envs_per_worker,
            )
        )

        self.algorithm = config.build()
        self.checkpoint_config = checkpoint_config
        self.total_iters = 0

    @staticmethod
    def register_routing_env(
        env_creator: Factory[RoutingEnv],
        circuit_generator: CircuitGenerator,
        noise_generator: NoiseGenerator,
        *,
        recalibration_interval: int = 64,
        episodes_per_circuit: int = 1,
    ):
        def create_env(context: EnvContext) -> TrainingWrapper:
            seed = context.get('seed')

            if seed is not None:
                env_seed = ctypes.c_size_t(hash((seed, context.worker_index, context.vector_index))).value

                env_circuit_generator = copy.deepcopy(circuit_generator)
                env_circuit_generator.seed(env_seed)

                env_noise_generator = copy.deepcopy(noise_generator)
                env_noise_generator.seed(env_seed)
            else:
                env_circuit_generator = circuit_generator
                env_noise_generator = noise_generator

            return TrainingWrapper(
                env_creator(),
                env_circuit_generator,
                env_noise_generator,
                recalibration_interval=recalibration_interval,
                episodes_per_circuit=episodes_per_circuit,
            )

        register_env(ROUTING_ENV_NAME, create_env)

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str,
        env_creator: Factory[RoutingEnv],
        circuit_generator: CircuitGenerator,
        noise_generator: NoiseGenerator,
        *,
        recalibration_interval: int = 64,
        episodes_per_circuit: int = 1,
        checkpoint_config: Optional[CheckpointConfig] = None,
    ) -> Self:
        TrainingOrchestrator.register_routing_env(
            env_creator,
            circuit_generator,
            noise_generator,
            recalibration_interval=recalibration_interval,
            episodes_per_circuit=episodes_per_circuit,
        )

        # TODO: Logger creator does not persist after loading model

        obj = cls.__new__(cls)
        super(TrainingOrchestrator, obj).__init__()
        obj.algorithm = PPO.from_checkpoint(model_dir)
        obj.total_iters = obj.algorithm.iteration
        obj.checkpoint_config = checkpoint_config

        return obj

    def train(self, iters: int):
        for _ in range(iters):
            self.algorithm.train()
            self.total_iters += 1

            if self.checkpoint_config is not None and self.total_iters % self.checkpoint_config.interval == 0:
                self.save(self.checkpoint_config.model_dir)

    def save(self, model_dir: str):
        self.algorithm.save(model_dir)


class EvaluationOrchestrator:
    reliability_map: dict[tuple[int, int], float]

    def __init__(
        self,
        policy: Policy,
        env: RoutingEnv,
        circuit_generator: CircuitGenerator,
        backend_properties: BackendProperties,
        *,
        evaluation_episodes: int = 10,
        stochastic: bool = True,
        num_circuits: Optional[int] = None,
        routing_methods: str | Collection[str] = 'sabre',
        optimization_level: int = 0,
        use_tqdm: bool = False,
        seed: Optional[int] = None,
    ):
        if num_circuits is None:
            if isinstance(circuit_generator, DatasetCircuitGenerator):
                num_circuits = len(circuit_generator.dataset)
            else:
                num_circuits = 100

        if num_circuits <= 0:
            raise ValueError(f'Number of circuits must be positive, got {num_circuits}')

        if not stochastic:
            evaluation_episodes = 1

        if evaluation_episodes <= 0:
            raise ValueError(f'Evaluation episodes must be positive, got {evaluation_episodes}')

        if not (0 <= optimization_level <= 3):
            raise ValueError(f'Optimization level must be between 0 and 3, got {optimization_level}')

        if seed is not None:
            seed_default_generators(seed)
            circuit_generator.seed(seed)

        self.policy = policy
        self.eval_env = EvaluationWrapper(
            env,
            circuit_generator,
            backend_properties,
            evaluation_episodes=evaluation_episodes,
        )

        self.stochastic = stochastic
        self.num_circuits = num_circuits
        self.routing_methods = [routing_methods] if isinstance(routing_methods, str) else list(routing_methods)
        self.optimization_level = optimization_level
        self.use_tqdm = use_tqdm
        self.seed = seed

        self.env = self.eval_env.env
        self.routing_env = self.eval_env.unwrapped

        self.qiskit_coupling_map = CouplingMap(self.routing_env.coupling_map.to_directed().edge_list())  # type: ignore

        self.metrics_analyzer = MetricsAnalyzer()

    def log_circuit_metrics(
        self,
        method: str,
        original_circuit: QuantumCircuit,
        routed_circuit: QuantumCircuit,
        *,
        exclude: Optional[Set[str]] = None,
    ):
        exclude = set() if exclude is None else exclude
        routed_ops = cast(dict[str, int], routed_circuit.count_ops())

        def log_metric_checked(metric: str, value: Real):
            if metric not in exclude:
                self.metrics_analyzer.log_metric(method, metric, value)

        for gate in ['swap', 'bridge']:
            log_metric_checked(f'{gate}_count', routed_ops.get(gate, 0))

        routed_circuit = transpile(
            routed_circuit,
            basis_gates=IBM_BASIS_GATES,
            optimization_level=self.optimization_level,
            seed_transpiler=self.seed,
        )

        original_cnot_count = original_circuit.count_ops().get('cx', 0)  # type: ignore
        cnot_count = routed_circuit.count_ops().get('cx', 0)  # type: ignore
        added_cnot_count = cnot_count - original_cnot_count

        original_depth = original_circuit.depth()
        depth = routed_circuit.depth()

        reliability = circuit_reliability(routed_circuit, self.routing_env.edge_to_reliability)
        log_reliability = np.emath.logn(self.routing_env.noise_config.log_base, reliability)

        log_metric_checked('original_cnot_count', original_cnot_count)
        log_metric_checked('cnot_count', cnot_count)
        log_metric_checked('added_cnot_count', added_cnot_count)

        log_metric_checked('original_depth', original_depth)
        log_metric_checked('depth', depth)

        log_metric_checked('reliability', reliability)
        log_metric_checked('log_reliability', log_reliability)

        log_metric_checked('normalized_added_cnot_count', added_cnot_count / original_cnot_count)
        log_metric_checked('normalized_depth', depth / original_depth)
        log_metric_checked('normalized_log_reliability', log_reliability / original_cnot_count)

    def evaluate(self):
        progress = (
            tqdm(total=self.num_circuits * self.eval_env.evaluation_iters)
            if self.use_tqdm else None
        )

        for _ in range(self.num_circuits):
            start_time = time.perf_counter()

            initial_layout = None
            for _ in range(self.eval_env.evaluation_iters):
                obs, _ = self.eval_env.reset()
                if initial_layout is None:
                    initial_layout = self.routing_env.qubit_to_node.tolist()

                terminated = False
                while not terminated:
                    action, *_ = self.policy.compute_single_action(obs, explore=self.stochastic)
                    obs, _, terminated, *_ = self.eval_env.step(action)

                if self.use_tqdm:
                    progress.update()

            self.metrics_analyzer.log_metric('rl', 'routing_time', time.perf_counter() - start_time)

            original_circuit = self.routing_env.circuit
            self.log_circuit_metrics('rl', original_circuit, self.env.best_circuit)

            for method in self.routing_methods:
                start_time = time.perf_counter()
                routed_circuit = transpile(
                    original_circuit,
                    coupling_map=self.qiskit_coupling_map,
                    initial_layout=initial_layout,
                    routing_method=method,
                    optimization_level=0,
                    seed_transpiler=self.seed,
                )
                self.metrics_analyzer.log_metric(method, 'routing_time', time.perf_counter() - start_time)

                self.log_circuit_metrics(method, original_circuit, routed_circuit, exclude={'bridge_count'})

        progress.close()
