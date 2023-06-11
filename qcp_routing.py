
import functools
import multiprocessing
import operator
from argparse import ArgumentParser
from math import inf

import matplotlib.pyplot as plt
import rustworkx as rx
import torch.nn as nn
from qiskit import QuantumCircuit, transpile
from qiskit.converters import dag_to_circuit
from qiskit.transpiler import CouplingMap
from rich import print
from rustworkx.visualization import mpl_draw
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from routing.circuit_gen import RandomCircuitGenerator
from routing.env import QcpRoutingEnv, NoiseConfig
from utils import qubits_to_indices


def t_topology() -> rx.PyGraph:
    g = rx.PyGraph()

    g.add_nodes_from(range(5))
    g.add_edges_from_no_data([(0, 1), (1, 2), (1, 3), (3, 4)])

    return g


def h_topology() -> rx.PyGraph:
    g = rx.PyGraph()

    g.add_nodes_from(range(7))
    g.add_edges_from_no_data([(0, 1), (1, 2), (1, 3), (3, 5), (4, 5), (5, 6)])

    return g


def grid_topology(rows: int, cols: int) -> rx.PyGraph:
    g = rx.PyGraph()

    g.add_nodes_from(range(rows * cols))

    for row in range(rows):
        for col in range(cols):
            if col != cols - 1:
                g.add_edge(row * cols + col, row * cols + col + 1, None)

            if row != rows - 1:
                g.add_edge(row * cols + col, (row + 1) * cols + col, None)

    return g


def main():
    parser = ArgumentParser(
        'qcp_routing',
        description='Qubit routing with deep reinforcement learning',
    )

    parser.add_argument('-l', '--learn', action='store_true', help='whether or not to train the agent')
    parser.add_argument('-m', '--model', metavar='M', help='name of the model')
    parser.add_argument('-t', '--training-iters', metavar='I', help='training iterations per environment', default=100,
                        type=int)
    parser.add_argument('-r', '--routing-method', choices=['basic', 'stochastic', 'sabre'],
                        help='routing method for Qiskit compiler', default='sabre')
    parser.add_argument('-d', '--depth', help='depth of circuit observations', default=8, type=int)
    parser.add_argument('-e', '--envs', help='number of environments (for vectorization)',
                        default=multiprocessing.cpu_count(), type=int)
    parser.add_argument('-i', '--iters', help='routing iterations for evaluation', default=10, type=int)
    parser.add_argument('--show-topology', action='store_true', help='show circuit topology')

    args = parser.parse_args()
    args.model_path = f'models/{args.model}.model'

    # Parameters
    n_steps = 1024
    training_iterations = 4
    noise_config = NoiseConfig(1.0e-2, 3.0e-3, log_base=2.0)

    g = t_topology()
    circuit_generator = RandomCircuitGenerator(g.num_nodes(), 16)

    if args.show_topology:
        rx.visualization.mpl_draw(g, with_labels=True)
        plt.show()

    def env_fn() -> QcpRoutingEnv:
        return QcpRoutingEnv(g, circuit_generator, args.depth, training_iterations=training_iterations,
                             noise_config=noise_config, termination_reward=0.0)

    vec_env = VecMonitor(SubprocVecEnv([env_fn] * args.envs))

    try:
        model = MaskablePPO.load(args.model_path, vec_env, tensorboard_log='logs/routing')
        reset = False
    except FileNotFoundError:
        policy_kwargs = {
            'net_arch': [64, 64, 96],
            'activation_fn': nn.Tanh,
        }

        model = MaskablePPO(MaskableMultiInputActorCriticPolicy, vec_env, policy_kwargs=policy_kwargs, n_steps=n_steps,
                            tensorboard_log='logs/routing', learning_rate=5e-5)
        args.learn = True
        reset = True

    if args.learn:
        model.learn(args.envs * args.training_iters * n_steps, progress_bar=True, tb_log_name='ppo',
                    reset_num_timesteps=reset)
        model.save(args.model_path)

    env = env_fn()
    obs, _ = env.reset()

    env.training = False
    env.initial_mapping = env.node_to_qubit.copy()

    reliability_map = {}
    for edge, value in zip(env.coupling_map.edge_list(), env.error_rates):  # type: ignore
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

    best_reward = -inf
    routed_circuit = env.circuit.copy_empty_like()

    for _ in range(args.iters):
        terminated = False
        total_reward = 0.0

        while not terminated:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
            action = int(action)

            obs, reward, terminated, *_ = env.step(action)
            total_reward += reward

        if total_reward > best_reward:
            best_reward = total_reward
            routed_circuit = dag_to_circuit(env.routed_dag)

        obs, _ = env.reset()

    print(f'Total reward: {best_reward:.3f}\n')

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
                     routing_method=args.routing_method, basis_gates=['u', 'swap', 'cx'], optimization_level=0)

    print(f'[b blue]Qiskit Compiler ({args.routing_method} routing)[/b blue]')
    print(f'Swaps: {t_qc.count_ops().get("swap", 0)}')

    t_qc = t_qc.decompose()
    print(f'CNOTs after decomposition: {t_qc.count_ops().get("cx")}')
    print(f'Depth after decomposition: {t_qc.depth()}')
    print(f'Reliability after decomposition: {reliability(t_qc):.3%}\n')


if __name__ == '__main__':
    main()
