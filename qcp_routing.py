
import rustworkx as rx
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv

from routing.env import QcpRoutingEnv
from routing.circuit_gen import RandomCircuitGenerator


def main():
    from rich import print
    from rustworkx.visualization import mpl_draw
    import matplotlib.pyplot as plt
    from qiskit.converters import dag_to_circuit
    from qiskit.transpiler import CouplingMap
    from qiskit import transpile

    learn = False
    show_topology = False

    g = rx.PyGraph()
    g.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7])
    g.add_edges_from_no_data([(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7)])

    if show_topology:
        rx.visualization.mpl_draw(g, with_labels=True)
        plt.show()

    def env_fn() -> QcpRoutingEnv:
        return QcpRoutingEnv(g, RandomCircuitGenerator(g.num_nodes(), 16), 8)

    vec_env = SubprocVecEnv([env_fn] * 8)

    try:
        model = MaskablePPO.load('m_qcp_routing.model', vec_env)
    except FileNotFoundError:
        policy_kwargs = {
            'net_arch': [64, 64, 96],
        }

        model = MaskablePPO(MaskableActorCriticPolicy, vec_env, policy_kwargs=policy_kwargs, n_steps=1024,
                            tensorboard_log='routing_logs', learning_rate=5e-5)

    if learn:
        model.learn(256 * 1024, progress_bar=True)
        model.save('m_qcp_routing.model')

    env = QcpRoutingEnv(g, RandomCircuitGenerator(g.num_nodes(), 16), 8)
    obs, _ = env.reset()

    initial_layout = env.qubit_to_node.copy().tolist()
    print(initial_layout)
    print(env.circuit.depth())

    terminated = False
    total_reward = 0.0

    while not terminated:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        action = int(action)

        obs, reward, terminated, *_ = env.step(action)
        total_reward += reward

    print(f'Total reward: {total_reward:.2f}')
    routed_circuit = dag_to_circuit(env.routed_dag)
    print(routed_circuit.count_ops())

    print(routed_circuit)
    print(routed_circuit.depth(), routed_circuit.depth() / env.circuit.depth())

    coupling_map = CouplingMap(g.to_directed().edge_list())
    t_qc = transpile(env.circuit, coupling_map=coupling_map, initial_layout=initial_layout,
                     routing_method='basic', basis_gates=['u', 'swap', 'cx'])

    print(t_qc)
    print(t_qc.depth(), t_qc.depth() / env.circuit.depth())
    print(t_qc.count_ops())


if __name__ == '__main__':
    main()
