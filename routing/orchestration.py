
import pathlib
import random
import time
from collections.abc import Collection, Set
from dataclasses import dataclass, field
from math import inf
from numbers import Real
from typing import Optional, Self, cast, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from ray.rllib import Policy
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env import EnvContext
from ray.tune import register_env
from tqdm.rich import tqdm

from action_mask_model import ActionMaskModel
from routing.callbacks import RoutingCallbacks
from routing.circuit_gen import CircuitGenerator, DatasetCircuitGenerator
from routing.env import RoutingEnv
from routing.env_wrapper import TrainingWrapper, EvaluationWrapper
from routing.noise import NoiseGenerator
from utils import reliability, Factory

ROUTING_ENV_NAME: Final[str] = 'RoutingEnv'


@dataclass
class CheckpointConfig:
    model_dir: str
    interval: int = field(default=25, kw_only=True)

def is_checkpoint(path: str | pathlib.Path) -> bool:
    if isinstance(path, str):
        path = pathlib.Path(path)

    return path.is_dir() and path.name.startswith('checkpoint')

def get_checkpoint_iters(checkpoint_dir: str | pathlib.Path) -> int:
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = pathlib.Path(checkpoint_dir)

    return int(checkpoint_dir.name.removeprefix('checkpoint_'))

def get_latest_checkpoint_dir(model_dir: str | pathlib.Path) -> pathlib.Path:
    if isinstance(model_dir, str):
        model_dir = pathlib.Path(model_dir)

    return max((path for path in model_dir.iterdir() if path.is_dir()), key=get_checkpoint_iters)  # type: ignore


class TrainingOrchestrator:
    algorithm: PPO

    def __init__(
        self,
        env_creator: Factory[RoutingEnv],
        circuit_generator: CircuitGenerator,
        *,
        noise_generator: Optional[NoiseGenerator] = None,
        recalibration_interval: int = 64,
        episodes_per_circuit: int = 1,
        checkpoint_config: Optional[CheckpointConfig] = None,
        lr: float = 1e-4,
        lr_schedule: Optional[list[list[int | float]]] = None,
        hidden_layers: Optional[list[int]] = None,
        activation_fn: str = 'silu',
        embedding_dim: Optional[int] = None,
        batch_size: int = 8192,
        minibatch_size: int = 128,
        sgd_iters: int = 10,
        vf_loss_coeff: float = 0.5,
        entropy_coeff: float = 0.0,
        evaluation_interval: Optional[int] = None,
        evaluation_duration: int = 256,
        num_gpus: float = 0.0,
        num_workers: int = 2,
        envs_per_worker: int = 4,
    ):
        if hidden_layers is None:
            hidden_layers = [64, 64]

        TrainingOrchestrator.register_routing_env(
            env_creator,
            circuit_generator,
            noise_generator=noise_generator,
            recalibration_interval=recalibration_interval,
            episodes_per_circuit=episodes_per_circuit,
        )

        config = (
            PPOConfig()
            .training(
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
                lambda_=0.95,
                sgd_minibatch_size=minibatch_size,
                num_sgd_iter=sgd_iters,
                vf_loss_coeff=vf_loss_coeff,
                entropy_coeff=entropy_coeff,
                clip_param=0.2,
                grad_clip=0.5,
                _enable_learner_api=False,
            )
            .callbacks(RoutingCallbacks)
            .debugging(log_level='ERROR')
            .environment(env=ROUTING_ENV_NAME)
            .evaluation(
                evaluation_interval=evaluation_interval,
                evaluation_duration=evaluation_duration,
            )
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
            .rl_module(_enable_rl_module_api=False)
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
        *,
        noise_generator: Optional[NoiseGenerator] = None,
        recalibration_interval: int = 64,
        episodes_per_circuit: int = 1,
    ):
        # TODO: potentially refactor into env-specific generators
        def create_env(_context: EnvContext) -> TrainingWrapper:
            return TrainingWrapper(
                env_creator(),
                circuit_generator,
                noise_generator=noise_generator,
                recalibration_interval=recalibration_interval,
                episodes_per_circuit=episodes_per_circuit,
            )

        register_env(ROUTING_ENV_NAME, create_env)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        env_creator: Factory[RoutingEnv],
        circuit_generator: CircuitGenerator,
        *,
        noise_generator: Optional[NoiseGenerator] = None,
        recalibration_interval: int = 64,
        episodes_per_circuit: int = 1,
        checkpoint_config: Optional[CheckpointConfig] = None,
    ) -> Self:
        TrainingOrchestrator.register_routing_env(
            env_creator,
            circuit_generator,
            noise_generator=noise_generator,
            recalibration_interval=recalibration_interval,
            episodes_per_circuit=episodes_per_circuit,
        )

        obj = cls.__new__(cls)
        super(TrainingOrchestrator, obj).__init__()
        obj.algorithm = PPO.from_checkpoint(checkpoint_dir)
        obj.total_iters = get_checkpoint_iters(checkpoint_dir)
        obj.checkpoint_config = checkpoint_config

        return obj

    def train(self, iters: int):
        for _ in range(iters):
            self.algorithm.train()
            self.total_iters += 1

            if self.checkpoint_config is not None and self.total_iters % self.checkpoint_config.interval == 0:
                self.algorithm.save(self.checkpoint_config.model_dir)

    def save(self, model_dir: str) -> str:
        return self.algorithm.save(model_dir)


class EvaluationOrchestrator:
    reliability_map: dict[tuple[int, int], float]
    metrics: dict[str, dict[str, list[Real]]]

    def __init__(
        self,
        policy: Policy,
        env: RoutingEnv,
        circuit_generator: CircuitGenerator,
        *,
        noise_generator: Optional[NoiseGenerator] = None,
        stochastic: bool = True,
        evaluation_iters: int = 10,
        num_circuits: Optional[int] = None,
        routing_methods: str | Collection[str] = 'sabre',
        use_tqdm: bool = False,
        seed: Optional[int] = None,
    ):
        if num_circuits is None:
            if isinstance(circuit_generator, DatasetCircuitGenerator):
                num_circuits = len(circuit_generator.dataset)
            else:
                num_circuits = 100

        if num_circuits <= 0:
            raise ValueError(f'Number of evaluation circuits must be positive, got {num_circuits}')

        if not stochastic:
            evaluation_iters = 1

        if evaluation_iters <= 0:
            raise ValueError(f'Evaluation iterations must be positive, got {evaluation_iters}')

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            circuit_generator.seed(seed)
            if noise_generator is not None:
                noise_generator.seed(seed)

        self.policy = policy
        self.eval_env = EvaluationWrapper(
            env,
            circuit_generator,
            noise_generator=noise_generator,
            evaluation_iters=evaluation_iters,
        )

        self.stochastic = stochastic
        self.num_circuits = num_circuits
        self.routing_methods = [routing_methods] if isinstance(routing_methods, str) else list(routing_methods)
        self.use_tqdm = use_tqdm
        self.seed = seed

        self.env = self.eval_env.env
        self.initial_layout = env.qubit_to_node.tolist()
        self.qiskit_coupling_map = CouplingMap(env.coupling_map.to_directed().edge_list())  # type: ignore

        self.reliability_map = {}
        for edge, edge_reliability in zip(env.coupling_map.edge_list(), 1.0 - env.error_rates):  # type: ignore
            self.reliability_map[edge] = edge_reliability
            self.reliability_map[edge[::-1]] = edge_reliability

        self.metrics = {}

    def log_metric(self, method: str, metric: str, value: Real):
        self.metrics.setdefault(method, {}).setdefault(metric, []).append(value)

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
                self.log_metric(method, metric, value)

        for gate in ['swap', 'bridge']:
            log_metric_checked(f'{gate}_count', routed_ops.get(gate, 0))

        routed_circuit = routed_circuit.decompose(['swap', 'bridge'])

        original_cnot_count = original_circuit.count_ops().get('cx', 0)
        cnot_count = routed_circuit.count_ops().get('cx', 0)

        log_metric_checked('cnot_count', cnot_count)
        log_metric_checked('added_cnot_count', cnot_count - original_cnot_count)
        log_metric_checked('depth', routed_circuit.depth())
        log_metric_checked('reliability', reliability(routed_circuit, self.reliability_map))

    def evaluate(self):
        iterable = range(self.num_circuits)
        if self.use_tqdm:
            iterable = tqdm(iterable)

        for _ in iterable:
            start_time = time.perf_counter()
            best_reward = -inf
            routed_circuit = self.env.circuit.copy_empty_like()

            for _ in range(self.eval_env.evaluation_iters):
                obs, _ = self.eval_env.reset()
                terminated = False
                total_reward = 0.0

                while not terminated:
                    action, *_ = self.policy.compute_single_action(obs, explore=self.stochastic)
                    obs, reward, terminated, *_ = self.eval_env.step(action)
                    total_reward += reward

                if total_reward > best_reward:
                    best_reward = total_reward
                    routed_circuit = self.env.routed_circuit()
            self.log_metric('rl', 'routing_time', time.perf_counter() - start_time)

            original_circuit = self.env.circuit
            self.log_circuit_metrics('rl', original_circuit, routed_circuit)

            for method in self.routing_methods:
                start_time = time.perf_counter()
                routed_circuit = transpile(
                    original_circuit,
                    coupling_map=self.qiskit_coupling_map,
                    initial_layout=self.initial_layout,
                    routing_method=method,
                    optimization_level=0,
                    seed_transpiler=self.seed,
                )
                self.log_metric(method, 'routing_time', time.perf_counter() - start_time)

                self.log_circuit_metrics(method, original_circuit, routed_circuit, exclude={'bridge_count'})

    def metric_as_df(self, metric: str) -> pd.DataFrame:
        return pd.DataFrame({method: method_data.get(metric, []) for method, method_data in self.metrics.items()})

    def box_plot(self, metric: str):
        sns.boxplot(self.metric_as_df(metric))
        plt.show()

    def kde_plot(self, metric: str):
        sns.kdeplot(self.metric_as_df(metric))
        plt.show()
