
from argparse import ArgumentParser
from math import inf

from qiskit import transpile
from qiskit.transpiler import CouplingMap
from ray.rllib import Policy
from rich import print
from tqdm.rich import tqdm

from routing.circuit_gen import RandomCircuitGenerator, DatasetCircuitGenerator
from routing.env import CircuitMatrixRoutingEnv
from routing.env_wrapper import EvaluationWrapper
from routing.noise import NoiseConfig, UniformNoiseGenerator
from routing.topology import t_topology
from utils import reliability


def main():
    parser = ArgumentParser('eval', description='Noise-Resilient Reinforcement Learning Strategies for Quantum '
                                                'Compiling (model evaluation script)')

    parser.add_argument('-m', '--model-checkpoint', metavar='M', help='path to model checkpoint', required=True)
    parser.add_argument('-d', '--depth', metavar='N', type=int, default=8, help='depth of circuit observations')
    parser.add_argument('-i', '--iters', metavar='N', type=int, default=12, help='evaluation iterations per circuit')
    parser.add_argument('-r', '--routing-method', choices=['basic', 'stochastic', 'sabre'],
                        help='routing method for Qiskit compiler', default='sabre')
    parser.add_argument('--dataset-dir', metavar='D', help='directory containing benchmark circuits in OpenQASM format')
    parser.add_argument('--circuit-size', metavar='N', type=int, default=64, help='random circuit gate count')
    parser.add_argument('--evaluation-circuits', metavar='N', type=int, default=100,
                        help='number of (random) evaluation circuits')
    parser.add_argument('--seed', metavar='N', help='seed for random number generation', type=int)

    args = parser.parse_args()

    g = t_topology()
    circuit_generator = (
        RandomCircuitGenerator(g.num_nodes(), args.circuit_size, seed=args.seed) if args.dataset_dir is None
        else DatasetCircuitGenerator.from_dir(args.dataset_dir)
    )
    noise_generator = UniformNoiseGenerator(1e-2, 3e-3)

    if args.dataset_dir is not None:
        args.evaluation_circuits = len(circuit_generator.dataset)

    policy = Policy.from_checkpoint(f'{args.model_checkpoint}')['default_policy']

    eval_env = EvaluationWrapper(CircuitMatrixRoutingEnv(g, depth=args.depth, noise_config=NoiseConfig()),
                                 circuit_generator, noise_generator=noise_generator, evaluation_iters=args.iters)
    env = eval_env.env
    initial_layout = env.qubit_to_node.tolist()

    reliability_map = {}
    for edge, value in zip(env.coupling_map.edge_list(), env.error_rates):  # type: ignore
        value = 1.0 - value
        reliability_map[edge] = value
        reliability_map[edge[::-1]] = value

    coupling_map = CouplingMap(g.to_directed().edge_list())  # type: ignore

    avg_episode_reward = 0.0
    avg_swaps_rl, avg_bridges_rl, avg_swaps_qiskit = 0.0, 0.0, 0.0
    avg_cnots_rl, avg_cnots_qiskit = 0.0, 0.0
    avg_depth_rl, avg_depth_qiskit = 0.0, 0.0
    avg_reliability_rl, avg_reliability_qiskit = 0.0, 0.0

    for _ in tqdm(range(args.evaluation_circuits)):  # type: ignore
        best_reward = -inf
        routed_circuit = env.circuit.copy_empty_like()

        for _ in range(args.iters):
            obs, _ = eval_env.reset()
            terminated = False
            total_reward = 0.0

            while not terminated:
                action, *_ = policy.compute_single_action(obs)
                obs, reward, terminated, *_ = env.step(action)
                total_reward += reward

            if total_reward > best_reward:
                best_reward = total_reward
                routed_circuit = env.routed_circuit()

        t_qc = transpile(env.circuit, coupling_map=coupling_map, initial_layout=initial_layout,
                         routing_method=args.routing_method, basis_gates=['u', 'swap', 'cx'], optimization_level=0,
                         seed_transpiler=args.seed)

        rl_ops = routed_circuit.count_ops()
        qiskit_ops = t_qc.count_ops()

        avg_episode_reward += best_reward

        avg_swaps_rl += rl_ops.get("swap", 0)
        avg_bridges_rl += rl_ops.get("bridge", 0)
        avg_swaps_qiskit += qiskit_ops.get("swap", 0)

        routed_circuit = routed_circuit.decompose()
        t_qc = t_qc.decompose()

        avg_cnots_rl += routed_circuit.count_ops().get("cx", 0)
        avg_cnots_qiskit += t_qc.count_ops().get("cx", 0)

        avg_depth_rl += routed_circuit.depth()
        avg_depth_qiskit += t_qc.depth()

        avg_reliability_rl += reliability(routed_circuit, reliability_map)
        avg_reliability_qiskit += reliability(t_qc, reliability_map)

    # Calculate averages
    avg_episode_reward /= args.evaluation_circuits

    avg_swaps_rl /= args.evaluation_circuits
    avg_bridges_rl /= args.evaluation_circuits
    avg_swaps_qiskit /= args.evaluation_circuits

    avg_cnots_rl /= args.evaluation_circuits
    avg_cnots_qiskit /= args.evaluation_circuits

    avg_depth_rl /= args.evaluation_circuits
    avg_depth_qiskit /= args.evaluation_circuits

    avg_reliability_rl /= args.evaluation_circuits
    avg_reliability_qiskit /= args.evaluation_circuits

    avg_added_cnots_rl = avg_cnots_rl - args.circuit_size
    avg_added_cnots_qiskit = avg_cnots_qiskit - args.circuit_size

    # Print results
    print(f'\nAverage episode reward: {avg_episode_reward:.3f}\n')

    print('[b blue]RL Routing[/b blue]')
    print(f'- Average swaps: {avg_swaps_rl:.2f}')
    if env.allow_bridge_gate:
        print(f'- Average bridges: {avg_bridges_rl:.2f}')

    print('[b yellow]After Decomposition[/b yellow]')
    print(f'- Average CNOTs: {avg_cnots_rl:.2f}')
    print(f'- Average added CNOTs: {avg_added_cnots_rl:.2f}')
    print(f'- Average depth: {avg_depth_rl:.2f}')
    print(f'- Average reliability: {avg_reliability_rl:.3%}\n')

    print(f'[b blue]Qiskit Compiler ({args.routing_method} routing)[/b blue]')
    print(f'- Average swaps: {avg_swaps_qiskit:.3f}')

    print('[b yellow]After Decomposition[/b yellow]')
    print(f'- Average CNOTs: {avg_cnots_qiskit:.2f}')
    print(f'- Average added CNOTs: {avg_added_cnots_qiskit:.2f}')
    print(f'- Average depth: {avg_depth_qiskit:.2f}')
    print(f'- Average reliability: {avg_reliability_qiskit:.3%}\n')

    print(f'[b blue]RL vs Qiskit[/b blue]')
    avg_cnot_reduction = (avg_cnots_qiskit - avg_cnots_rl) / avg_cnots_qiskit
    print(f'Average CNOT reduction: [magenta]{avg_cnot_reduction:.3%}[/magenta]')
    avg_added_cnot_reduction = (avg_added_cnots_qiskit - avg_added_cnots_rl) / avg_added_cnots_qiskit
    print(f'Average added CNOT reduction: [magenta]{avg_added_cnot_reduction:.3%}[/magenta]')
    avg_depth_reduction = (avg_depth_qiskit - avg_depth_rl) / avg_depth_qiskit
    print(f'Average depth reduction: [magenta]{avg_depth_reduction:.3%}[/magenta]')
    avg_reliability_increase = (avg_reliability_rl - avg_reliability_qiskit) / avg_reliability_qiskit
    print(f'Average reliability increase: [magenta]{avg_reliability_increase:.3%}[/magenta]')


if __name__ == '__main__':
    main()

