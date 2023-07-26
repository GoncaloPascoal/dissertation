
from argparse import ArgumentParser
from collections.abc import Callable
from typing import Any, Final

import rustworkx as rx
import yaml

from routing.circuit_gen import CircuitGenerator, RandomCircuitGenerator, LayeredCircuitGenerator
from routing.env import RoutingEnv, SequentialRoutingEnv, LayeredRoutingEnv, CircuitMatrix, ObsModule
from routing.noise import NoiseConfig
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


def parse_env_config(path: str) -> RoutingEnv:
    config = parse_yaml(path)

    cls = ROUTING_ENVS[config.get('type', 'sequential')]
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

    return cls(
        coupling_map,
        noise_config=noise_config,
        obs_modules=obs_modules,
        **config.get('args', {})
    )

def parse_train_config(path: str):
    pass

def parse_eval_config(path: str):
    pass


def main():
    parser = ArgumentParser('toml', description='Qubit routing with deep reinforcement learning',)
    parser.add_argument('config_path', help='path to configuration file')

    args = parser.parse_args()

    env = parse_env_config(args.config_path)
    print(env.observation_space)


if __name__ == '__main__':
    main()
