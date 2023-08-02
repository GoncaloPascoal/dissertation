
import time
from collections.abc import Sequence, Collection, Set
from math import inf
from numbers import Real
from typing import Any, ClassVar, Optional, Self, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from ray.rllib import Policy
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune import register_env
from tqdm.rich import tqdm

from action_mask_model import ActionMaskModel
from routing.circuit_gen import CircuitGenerator, DatasetCircuitGenerator
from routing.env import RoutingEnvCreator, RoutingEnv
from routing.env_wrapper import TrainingWrapper, EvaluationWrapper
from routing.noise import NoiseGenerator
from utils import reliability


class TrainingOrchestrator:
    ENV_NAME: ClassVar[str] = 'RoutingEnv'

    algorithm: PPO

    def __init__(
        self,
        env_creator: RoutingEnvCreator,
        circuit_generator: CircuitGenerator,
        *,
        noise_generator: Optional[NoiseGenerator] = None,
        episodes_per_circuit: int = 1,
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
            return TrainingWrapper(env_creator.create(), circuit_generator, noise_generator, episodes_per_circuit)

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
    metrics: dict[str, dict[str, list[Real]]]

    def __init__(
        self,
        policy: Policy,
        env: RoutingEnv,
        circuit_generator: CircuitGenerator,
        *,
        noise_generator: Optional[NoiseGenerator] = None,
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

        self.policy = policy
        self.eval_env = EvaluationWrapper(
            env,
            circuit_generator,
            noise_generator=noise_generator,
            evaluation_iters=evaluation_iters,
        )

        self.num_circuits = num_circuits
        self.routing_methods = [routing_methods] if isinstance(routing_methods, str) else list(routing_methods)
        self.use_tqdm = use_tqdm
        self.seed = seed

        self.env = self.eval_env.env
        self.initial_layout = env.qubit_to_node.tolist()
        self.qiskit_coupling_map = CouplingMap(env.coupling_map.to_directed().edge_list())

        self.reliability_map = {}
        for edge, error_rate in zip(env.coupling_map.edge_list(), env.error_rates):  # type: ignore
            edge_reliability = 1.0 - error_rate
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
                    action, *_ = self.policy.compute_single_action(obs)
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
