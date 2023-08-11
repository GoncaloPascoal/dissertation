
from collections.abc import Callable
from typing import Any, Final, Optional

import rustworkx as rx
import yaml
from ray.rllib import Policy

from routing.circuit_gen import (
    CircuitGenerator, RandomCircuitGenerator, LayeredCircuitGenerator, DatasetCircuitGenerator
)
from routing.env import CircuitMatrix, ObsModule, QubitInteractions, RoutingEnv
from routing.noise import NoiseConfig, UniformNoiseGenerator, NoiseGenerator, KdeNoiseGenerator
from routing.orchestration import TrainingOrchestrator, EvaluationOrchestrator, CheckpointConfig
from routing.topology import t_topology, h_topology, grid_topology, linear_topology, ibm_16q_topology, ibm_27q_topology

OBS_MODULES: Final[dict[str, type[ObsModule]]] = {
    'circuit_matrix': CircuitMatrix,
    'qubit_interactions': QubitInteractions,
}

COUPLING_MAPS: Final[dict[str, Callable[..., rx.PyGraph]]] = {
    't': t_topology,
    'h': h_topology,
    'grid': grid_topology,
    'linear': linear_topology,
    'ibm_16q': ibm_16q_topology,
    'ibm_27q': ibm_27q_topology,
}

CIRCUIT_GENERATORS: Final[dict[str, Callable[..., CircuitGenerator]]] = {
    'random': RandomCircuitGenerator,
    'layered': LayeredCircuitGenerator,
    'dataset': DatasetCircuitGenerator.from_dir,
}

NOISE_GENERATORS: Final[dict[str, Callable[..., NoiseGenerator]]] = {
    'uniform': UniformNoiseGenerator,
    'uniform_samples': UniformNoiseGenerator.from_samples,
    'kde': KdeNoiseGenerator,
}


def parse_yaml(path: str) -> dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_env_config(path: str) -> Callable[[], RoutingEnv]:
    config = parse_yaml(path)

    coupling_map_config: dict[str, Any] | str | list = config.pop('coupling_map')

    if isinstance(coupling_map_config, dict):
        coupling_map = COUPLING_MAPS[coupling_map_config['type']](**coupling_map_config.get('args', {}))
    elif isinstance(coupling_map_config, str):
        coupling_map = COUPLING_MAPS[coupling_map_config]()
    elif isinstance(coupling_map_config, list):
        coupling_map = rx.PyGraph()
        coupling_map.extend_from_edge_list(coupling_map_config)
    else:
        raise ValueError(f'Coupling map configuration has invalid type `{type(coupling_map_config)}`')

    noise_config = NoiseConfig() if config.pop('noise_aware', True) else None

    obs_modules = []
    obs_modules_config: list[str | dict[str, Any]] = config.pop('obs_modules', [])
    for om_config in obs_modules_config:
        if isinstance(om_config, str):
            obs_modules.append(OBS_MODULES[om_config]())
        elif isinstance(om_config, dict):
            obs_modules.append(OBS_MODULES[om_config['type']](**om_config['args']))  # type: ignore
        else:
            raise ValueError(f'Observation module configuration has invalid type `{type(om_config)}`')

    def create_env() -> RoutingEnv:
        return RoutingEnv(
            coupling_map,
            noise_config=noise_config,
            obs_modules=obs_modules,
            **config,
        )

    return create_env


def parse_generators(config: dict[str, Any]) -> tuple[CircuitGenerator, Optional[NoiseGenerator]]:
    circuit_config = config.pop('circuit')
    noise_config = config.pop('noise', None)

    circuit_generator = CIRCUIT_GENERATORS[circuit_config['type']](**circuit_config['args'])

    noise_generator = None
    if noise_config is not None:
        noise_generator = NOISE_GENERATORS[noise_config['type']](**noise_config['args'])

    return circuit_generator, noise_generator


def parse_train_config(
    env_config_path: str,
    train_config_path: str,
    *,
    checkpoint_dir: Optional[str] = None,
    override_args: Optional[dict[str, Any]] = None,
) -> TrainingOrchestrator:
    env_creator = parse_env_config(env_config_path)
    config = parse_yaml(train_config_path)

    generators_config = config.pop('generators')
    circuit_generator, noise_generator = parse_generators(
        parse_yaml(generators_config) if isinstance(generators_config, str) else generators_config
    )

    checkpoint_config = config.pop('checkpoint_config', None)
    if checkpoint_config is not None:
        checkpoint_config = CheckpointConfig(**checkpoint_config)

    args = dict(
        env_creator=env_creator,
        circuit_generator=circuit_generator,
        noise_generator=noise_generator,
        checkpoint_config=checkpoint_config,
        **config,
    )
    args.update(override_args)

    if checkpoint_dir is None:
        orchestrator = TrainingOrchestrator(**args)
    else:
        # Retain only environment or training-related args
        args = {
            k: v for k, v in args.items() if k in {
                'env_creator', 'circuit_generator', 'noise_generator', 'recalibration_interval', 'episodes_per_circuit',
                'checkpoint_config',
            }
        }

        orchestrator = TrainingOrchestrator.from_checkpoint(checkpoint_dir, **args)

    return orchestrator

def parse_eval_config(
    env_config_path: str,
    eval_config_path: str,
    checkpoint_dir: str,
    *,
    override_args: Optional[dict[str, Any]] = None,
):
    if override_args is None:
        override_args = {}

    env = parse_env_config(env_config_path)()
    config = parse_yaml(eval_config_path)

    policy = Policy.from_checkpoint(checkpoint_dir)['default_policy']

    generators_config = config.pop('generators')
    circuit_generator, noise_generator = parse_generators(
        parse_yaml(generators_config) if isinstance(generators_config, str) else generators_config
    )

    args = dict(
        policy=policy,
        env=env,
        circuit_generator=circuit_generator,
        noise_generator=noise_generator,
        **config,
    )
    args.update(override_args)

    return EvaluationOrchestrator(**args)
