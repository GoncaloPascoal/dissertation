
from argparse import ArgumentParser
from collections.abc import Callable
from typing import Any, Final

import rustworkx as rx
import yaml

from routing.circuit_gen import CircuitGenerator, RandomCircuitGenerator, LayeredCircuitGenerator
from routing.env import RoutingEnv, RoutingEnvCreator, SequentialRoutingEnv, LayeredRoutingEnv, CircuitMatrix, ObsModule
from routing.noise import NoiseConfig, UniformNoiseGenerator
from routing.orchestration import TrainingOrchestrator
from routing.topology import t_topology, h_topology, grid_topology, linear_topology


ROUTING_ENVS: Final[dict[str, type[RoutingEnv]]] = {
    'sequential': SequentialRoutingEnv,
    'layered': LayeredRoutingEnv,
}

OBS_MODULES: Final[dict[str, type[ObsModule]]] = {
    'circuit_matrix': CircuitMatrix,
}

COUPLING_MAPS: Final[dict[str, Callable[..., rx.PyGraph]]] = {
    't': t_topology,
    'h': h_topology,
    'grid': grid_topology,
    'linear': linear_topology,
}

CIRCUIT_GENERATORS: Final[dict[str, type[CircuitGenerator]]] = {
    'random': RandomCircuitGenerator,
    'layered': LayeredCircuitGenerator,
}


def parse_yaml(path: str) -> dict[str, Any]:
    with open(path, 'rb') as f:
        return yaml.safe_load(f)


def parse_env_config(path: str) -> RoutingEnvCreator:
    config = parse_yaml(path)

    env_class = ROUTING_ENVS[config.get('type', 'sequential')]
    coupling_map_config: dict[str, Any] | str | list = config['coupling_map']

    if isinstance(coupling_map_config, dict):
        coupling_map = COUPLING_MAPS[coupling_map_config['type']](**coupling_map_config.get('args', {}))
    elif isinstance(coupling_map_config, str):
        coupling_map = COUPLING_MAPS[coupling_map_config]()
    elif isinstance(coupling_map_config, list):
        coupling_map = rx.PyGraph()
        coupling_map.extend_from_edge_list(coupling_map_config)
    else:
        raise ValueError('Invalid coupling map configuration')

    noise_config = NoiseConfig() if config.get('noise_aware', True) else None

    obs_modules = []
    obs_modules_config: list[str | dict[str, Any]] = config.get('obs_modules', [])
    for c in obs_modules_config:
        if isinstance(c, str):
            obs_modules.append(OBS_MODULES[c]())
        elif isinstance(c, dict):
            obs_modules.append(OBS_MODULES[c['type']](**c['args']))  # type: ignore
        else:
            raise ValueError('Invalid observation module configuration')

    return RoutingEnvCreator(
        env_class,
        coupling_map=coupling_map,
        noise_config=noise_config,
        obs_modules=obs_modules,
        **config.get('args', {})
    )

def parse_train_config(path: str) -> TrainingOrchestrator:
    config = parse_yaml(path)
    env_creator = parse_env_config(config['env_config_path'])

    circuit_generator_config: dict[str, Any] | str = config['circuit_generator']
    circuit_generator = CIRCUIT_GENERATORS[circuit_generator_config['type']](**circuit_generator_config['args'])

    noise_generator = UniformNoiseGenerator(1e-2, 3e-3)

    return TrainingOrchestrator(
        env_creator,
        circuit_generator,
        noise_generator=noise_generator,
        **config.get('args', {}),
    )

def parse_eval_config(path: str):
    config = parse_yaml(path)
    # TODO


def main():
    parser = ArgumentParser('yaml', description='Qubit routing with deep reinforcement learning')
    parser.add_argument('config_path', help='path to configuration file')

    args = parser.parse_args()

    orchestrator = parse_train_config(args.config_path)
    print(orchestrator.algorithm)


if __name__ == '__main__':
    main()
