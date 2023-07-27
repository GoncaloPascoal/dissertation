
from argparse import ArgumentParser
from collections.abc import Callable
from typing import Any, Final

import rustworkx as rx
import yaml
from ray.rllib import Policy

from routing.circuit_gen import CircuitGenerator, RandomCircuitGenerator, LayeredCircuitGenerator, \
    DatasetCircuitGenerator
from routing.env import RoutingEnvCreator, CircuitMatrix, ObsModule
from routing.noise import NoiseConfig, UniformNoiseGenerator, NoiseGenerator
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


def _parse_circuit_generator_config(config: dict[str, Any]):
    return CIRCUIT_GENERATORS[config['type']](**config['args'])

def _parse_noise_generator_config(config: dict[str, Any]):
    return NOISE_GENERATORS[config['type']](**config['args'])

def parse_train_config(env_path: str, train_path: str) -> TrainingOrchestrator:
    env_creator = parse_env_config(env_path)
    config = parse_yaml(train_path)

    circuit_generator = _parse_circuit_generator_config(config['circuit_generator'])
    noise_generator = _parse_noise_generator_config(config['noise_generator'])

    return TrainingOrchestrator(
        env_creator,
        circuit_generator,
        noise_generator=noise_generator,
        **config.get('args', {}),
    )

def parse_eval_config(env_path: str, eval_path: str):
    env = parse_env_config(env_path).create()
    config = parse_yaml(eval_path)

    policy = Policy.from_checkpoint(config['checkpoint_path'])['default_policy']

    circuit_generator = _parse_circuit_generator_config(config['circuit_generator'])
    noise_generator = _parse_noise_generator_config(config['noise_generator'])

    return EvaluationOrchestrator(
        policy,
        env,
        circuit_generator,
        noise_generator=noise_generator,
        **config.get('args', {})
    )


def main():
    parser = ArgumentParser('yaml', description='Qubit routing with deep reinforcement learning')
    parser.add_argument('env_config', help='path to environment configuration file')
    parser.add_argument('train_config', help='path to training configuration file')

    args = parser.parse_args()

    orchestrator = parse_eval_config(args.env_config, args.train_config)
    orchestrator.evaluate()


if __name__ == '__main__':
    main()
