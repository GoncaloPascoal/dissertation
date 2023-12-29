
import json
from collections.abc import Callable
from typing import Any, Final, Optional

import rustworkx as rx
import yaml
from qiskit.providers.models import BackendProperties
from qiskit.transpiler import CouplingMap
from ray.rllib import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from narlsqr.env import CircuitMatrix, NoiseConfig, ObsModule, QubitInteractions, RoutingEnv
from narlsqr.generators.circuit import (CircuitGenerator, DatasetCircuitGenerator, LayeredCircuitGenerator,
                                        RandomCircuitGenerator)
from narlsqr.generators.noise import KdeNoiseGenerator, NoiseGenerator, UniformNoiseGenerator
from narlsqr.orchestration import CheckpointConfig, EvaluationOrchestrator, TrainingOrchestrator
from narlsqr.topology import grid_topology, h_topology, ibm_16q_topology, ibm_27q_topology, linear_topology, t_topology

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
    'uniform_calibration': UniformNoiseGenerator.from_calibration_file,
    'kde': KdeNoiseGenerator,
    'kde_calibration': KdeNoiseGenerator.from_calibration_file,
}


def parse_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding='utf8') as f:
        return yaml.safe_load(f)

def parse_coupling_map(coupling_map_config: dict[str, Any] | str | list) -> CouplingMap:
    if isinstance(coupling_map_config, dict):
        return COUPLING_MAPS[coupling_map_config['type']](**coupling_map_config.get('args', {}))

    if isinstance(coupling_map_config, str):
        return COUPLING_MAPS[coupling_map_config]()

    if isinstance(coupling_map_config, list):
        coupling_map = rx.PyGraph()
        coupling_map.extend_from_edge_list(coupling_map_config)
        return coupling_map

    raise ValueError(f'Coupling map configuration has invalid type `{type(coupling_map_config)}`')

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

    noise_config_args = config.pop('noise_config', {})
    noise_config = None if noise_config_args is None else NoiseConfig(**noise_config_args)

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


def parse_circuit_generator(config: dict[str, Any], env: RoutingEnv) -> CircuitGenerator:
    args = config['args']
    args['num_qubits'] = env.num_qubits

    return CIRCUIT_GENERATORS[config['type']](**args)

def parse_noise_generator(config: dict[str, Any], env: RoutingEnv) -> NoiseGenerator:
    type_ = config['type']
    args = config['args']

    if type_ not in {'uniform_calibration', 'kde_calibration'}:
        args['num_edges'] = env.num_edges

    return NOISE_GENERATORS[type_](**args)

def parse_train_config(
    env_config_path: str,
    train_config_path: str,
    *,
    checkpoint_dir: Optional[str] = None,
    override_args: Optional[dict[str, Any]] = None,
) -> TrainingOrchestrator:
    env_creator = parse_env_config(env_config_path)
    config = parse_yaml(train_config_path)

    sample_env = env_creator()
    circuit_generator = parse_circuit_generator(config.pop('circuit_generator'), sample_env)
    noise_generator = parse_noise_generator(config.pop('noise_generator'), sample_env)

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

    policy = Policy.from_checkpoint(checkpoint_dir)[DEFAULT_POLICY_ID]

    calibration_data = config.pop('calibration_data')
    with open(calibration_data, encoding='utf8') as f:
        data = json.load(f)
    backend_properties = BackendProperties.from_dict(data)

    circuit_generator = parse_circuit_generator(config.pop('circuit_generator'), env)

    args = dict(
        policy=policy,
        env=env,
        circuit_generator=circuit_generator,
        backend_properties=backend_properties,
        **config,
    )
    args.update(override_args)

    return EvaluationOrchestrator(**args)
