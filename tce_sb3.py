
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv

from conv import CnnFeaturesExtractor
from exact import CollapseFourAlternatingCnots, CommuteGates, InvertCnot, CommuteRzBetweenCnots
from gate_class import GateClass, generate_two_qubit_gate_classes_from_coupling_map
from tce import ExactTransformationCircuitEnv


def main():
    from qiskit.circuit.library import RZGate, SXGate
    from qiskit.circuit import Parameter
    from rich import print

    max_depth = 64
    num_qubits = 8

    # Nearest neighbor coupling
    coupling_map = [(q, q + 1) for q in range(num_qubits - 1)]
    coupling_map_qiskit = CouplingMap.from_line(num_qubits)

    u = QuantumCircuit(num_qubits)
    u.toffoli(0, 1, 2)
    u = transpile(u, basis_gates=['cx', 'sx', 'rz'], coupling_map=coupling_map_qiskit,
                  approximation_degree=0.0, seed_transpiler=1)

    print(u)
    print(f'[bold blue]Target unitary depth:[/bold blue] {u.depth()}')

    rz = RZGate(Parameter('x'))
    sx = SXGate()

    gate_classes = [
        GateClass(rz),
        GateClass(sx),
        *generate_two_qubit_gate_classes_from_coupling_map(coupling_map),
    ]

    transformation_rules = [
        CommuteGates(),
        CommuteRzBetweenCnots(),
        CollapseFourAlternatingCnots(),
        InvertCnot(),
    ]

    def env_fn() -> ExactTransformationCircuitEnv:
        return ExactTransformationCircuitEnv(max_depth, num_qubits, gate_classes, transformation_rules)
    vector_env = SubprocVecEnv([env_fn] * 6)

    policy_kwargs = {
        'features_extractor_class': CnnFeaturesExtractor,
    }

    try:
        model = MaskablePPO.load('tce_sb3_ppo.model', vector_env)
    except FileNotFoundError:
        model = MaskablePPO(MaskableActorCriticCnnPolicy, vector_env, policy_kwargs=policy_kwargs, n_steps=320,
                            batch_size=16, tensorboard_log='./tce_logs', learning_rate=1e-3)

    learn = True
    if learn:
        model.learn(1920, progress_bar=True)
        model.save('tce_sb3_ppo.model')

    env = env_fn()
    env.target_circuit = u
    env.max_time_steps = 128
    obs, _ = env.reset()
    terminated = False

    total_reward = 0.0
    while not terminated:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        action = int(action)

        print(env.format_action(action))

        obs, reward, terminated, *_ = env.step(action)
        total_reward += reward

    print(env.current_circuit)
    print(f'[bold blue]Optimized depth:[/bold blue] {env.current_circuit.depth()}')
    print(f'[bold yellow]Total reward:[/bold yellow] {total_reward}')


if __name__ == '__main__':
    main()
