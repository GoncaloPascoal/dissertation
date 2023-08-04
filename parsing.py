
from collections.abc import Callable
from typing import Any, Final, Optional

import rustworkx as rx
import yaml
from ray.rllib import Policy

from routing.circuit_gen import CircuitGenerator, RandomCircuitGenerator, LayeredCircuitGenerator, \
    DatasetCircuitGenerator
from routing.env import RoutingEnvCreator, CircuitMatrix, ObsModule
from routing.noise import NoiseConfig, UniformNoiseGenerator, NoiseGenerator, KdeNoiseGenerator
from routing.orchestration import TrainingOrchestrator, EvaluationOrchestrator
from routing.topology import t_topology, h_topology, grid_topology, linear_topology

OBS_MODULES: Final[dict[str, type[ObsModule]]] = {
    'circuit_matrix': CircuitMatrix,
}

COUPLING_MAPS: Final[dict[str, Callable[..., rx.PyGraph]]] = {
    't': t_topology,
    'h': h_topology,
    'grid': grid_topology,
    'linear': linear_topology,
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

def parse_env_config(path: str) -> RoutingEnvCreator:
    config = parse_yaml(path)

    coupling_map_config: dict[str, Any] | str | list = config['coupling_map']

    if isinstance(coupling_map_config, dict):
        coupling_map = COUPLING_MAPS[coupling_map_config['type']](**coupling_map_config.get('args', {}))
    elif isinstance(coupling_map_config, str):
        coupling_map = COUPLING_MAPS[coupling_map_config]()
    elif isinstance(coupling_map_config, list):
        coupling_map = rx.PyGraph()
        coupling_map.extend_from_edge_list(coupling_map_config)
    else:
        raise ValueError(f'Coupling map configuration has invalid type `{type(coupling_map_config)}`')

    noise_config = NoiseConfig() if config.get('noise_aware', True) else None

    obs_modules = []
    obs_modules_config: list[str | dict[str, Any]] = config.get('obs_modules', [])
    for om_config in obs_modules_config:
        if isinstance(om_config, str):
            obs_modules.append(OBS_MODULES[om_config]())
        elif isinstance(om_config, dict):
            obs_modules.append(OBS_MODULES[om_config['type']](**om_config['args']))  # type: ignore
        else:
            raise ValueError(f'Observation module configuration has invalid type `{type(om_config)}`')

    return RoutingEnvCreator(
        coupling_map=coupling_map,
        noise_config=noise_config,
        obs_modules=obs_modules,
        **config.get('args', {})
    )


def _parse_circuit_generator_config(config: dict[str, Any]) -> CircuitGenerator:
    return CIRCUIT_GENERATORS[config['type']](**config['args'])

def _parse_noise_generator_config(config: dict[str, Any]) -> NoiseGenerator:
    return NOISE_GENERATORS[config['type']](**config['args'])

def parse_train_config(
    env_path: str,
    train_path: str,
    override_args: Optional[dict[str, Any]] = None,
) -> TrainingOrchestrator:
    env_creator = parse_env_config(env_path)
    config = parse_yaml(train_path)

    circuit_generator = _parse_circuit_generator_config(config.pop('circuit_generator'))
    noise_generator = _parse_noise_generator_config(config.pop('noise_generator'))

    args = dict(
        env_creator=env_creator,
        circuit_generator=circuit_generator,
        noise_generator=noise_generator,
        **config,
    )
    args.update(override_args)

    return TrainingOrchestrator(**args)

def parse_eval_config(env_path: str, eval_path: str, override_args: Optional[dict[str, Any]] = None):
    if override_args is None:
        override_args = {}

    env = parse_env_config(env_path).create()
    config = parse_yaml(eval_path)

    checkpoint_path = config.pop('checkpoint_path', None) or override_args['checkpoint_path']
    policy = Policy.from_checkpoint(checkpoint_path)['default_policy']

    circuit_generator = _parse_circuit_generator_config(config.pop('circuit_generator'))
    noise_generator = _parse_noise_generator_config(config.pop('noise_generator'))

    args = dict(
        policy=policy,
        env=env,
        circuit_generator=circuit_generator,
        noise_generator=noise_generator,
        **config,
    )
    args.update(override_args)

    return EvaluationOrchestrator(**args)
