
import functools
import multiprocessing
import operator

import rustworkx as rx
import torch.nn as nn
from qiskit import QuantumCircuit
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from routing.circuit_gen import RandomCircuitGenerator
from routing.env import QcpRoutingEnv, NoiseConfig


def main():
    from rich import print
    from rustworkx.visualization import mpl_draw
    import matplotlib.pyplot as plt
    from qiskit.converters import dag_to_circuit
    from qiskit.transpiler import CouplingMap
    from qiskit import transpile

    from utils import qubits_to_indices

    # Parameters
    learn = False
    show_topology = False

    n_envs = multiprocessing.cpu_count()
    n_iters_per_env = 100
    n_steps = 1024

    depth = 8
    training_iterations = 4
    noise_config = NoiseConfig(1e-2, 3e-3, log_base=2)

    routing_method = 'sabre'

    g = rx.PyGraph()
    g.add_nodes_from([0, 1, 2, 3, 4])
    g.add_edges_from_no_data([(0, 1), (1, 2), (1, 3), (3, 4)])

    if show_topology:
        rx.visualization.mpl_draw(g, with_labels=True)
        plt.show()

    def env_fn() -> QcpRoutingEnv:
        return QcpRoutingEnv(g, RandomCircuitGenerator(g.num_nodes(), 32), depth,
                             training_iterations=training_iterations, noise_config=noise_config,
                             termination_reward=0.0)

    vec_env = VecMonitor(SubprocVecEnv([env_fn] * n_envs))

    try:
        model = MaskablePPO.load('models/m_qcp_routing3.model', vec_env, tensorboard_log='logs/routing')
        reset = False
    except FileNotFoundError:
        policy_kwargs = {
            'net_arch': [64, 64, 96],
            'activation_fn': nn.Tanh,
        }

        model = MaskablePPO(MaskableMultiInputActorCriticPolicy, vec_env, policy_kwargs=policy_kwargs, n_steps=n_steps,
                            tensorboard_log='logs/routing', learning_rate=5e-5)
        reset = True

    if learn:
        model.learn(n_envs * n_iters_per_env * n_steps, progress_bar=True, tb_log_name='ppo',
                    reset_num_timesteps=reset)
        model.save('models/m_qcp_routing.model')

    env = env_fn()
    obs, _ = env.reset()

    reliability_map = {}
    for edge, value in zip(env.coupling_map.edge_list(), env.error_rates):
        value = 1.0 - value
        reliability_map[edge] = value
        reliability_map[edge[::-1]] = value

    def reliability(circuit: QuantumCircuit) -> float:
        return functools.reduce(operator.mul, [
            reliability_map[qubits_to_indices(circuit, instruction.qubits)]
            for instruction in circuit.get_instructions('cx')
        ])

    initial_layout = env.qubit_to_node.copy().tolist()
    print(f'Initial depth: {env.circuit.depth()}')

    terminated = False
    total_reward = 0.0

    while not terminated:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
        action = int(action)

        obs, reward, terminated, *_ = env.step(action)
        total_reward += reward

    print(f'Total reward: {total_reward:.3f}\n')
    routed_circuit = dag_to_circuit(env.routed_dag)

    print('[b blue]RL Routing[/b blue]')
    print(f'Swaps: {routed_circuit.count_ops().get("swap", 0)}')
    if env.allow_bridge_gate:
        print(f'Bridges: {routed_circuit.count_ops().get("bridge", 0)}')

    routed_circuit = routed_circuit.decompose()
    print(f'CNOTs after decomposition: {routed_circuit.count_ops()["cx"]}')
    print(f'Depth after decomposition: {routed_circuit.depth()}')
    print(f'Reliability after decomposition: {reliability(routed_circuit):.3%}\n')

    coupling_map = CouplingMap(g.to_directed().edge_list())
    t_qc = transpile(env.circuit, coupling_map=coupling_map, initial_layout=initial_layout,
                     routing_method=routing_method, basis_gates=['u', 'swap', 'cx'], optimization_level=0)

    print(f'[b blue]Qiskit Compiler ({routing_method} routing)[/b blue]')
    print(f'Swaps: {t_qc.count_ops().get("swap", 0)}')

    t_qc = t_qc.decompose()
    print(f'CNOTs after decomposition: {t_qc.count_ops().get("cx")}')
    print(f'Depth after decomposition: {t_qc.depth()}')
    print(f'Reliability after decomposition: {reliability(t_qc):.3%}\n')


if __name__ == '__main__':
    main()
