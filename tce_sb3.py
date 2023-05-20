
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from transformation.cnn import CnnFeaturesExtractor, MaskableActorCriticFcnPolicy
from transformation.exact import CommuteGates, InvertCnot, CommuteRzBetweenCnots, ExactTransformationCircuitEnv
from transformation.gate_class import GateClass, generate_two_qubit_gate_classes_from_coupling_map


def main():
    from qiskit.circuit.library import RZGate, SXGate, QFT
    from qiskit.circuit import Parameter
    from rich import print

    max_depth = 32
    num_qubits = 4

    # Nearest neighbor coupling
    coupling_map = [(q, q + 1) for q in range(num_qubits - 1)]
    coupling_map_qiskit = CouplingMap.from_line(num_qubits)

    u = QuantumCircuit(num_qubits)
    u = QFT(3)
    u = transpile(u, basis_gates=['cx', 'sx', 'rz'], coupling_map=coupling_map_qiskit,
                  approximation_degree=0.0, seed_transpiler=1)

    print(u)
    print(f'[bold magenta]Target unitary gate count:[/bold magenta] {u.size()}')
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
        InvertCnot(),
    ]

    n_envs = 12
    n_steps = 64
    n_iters = 5

    def env_fn() -> ExactTransformationCircuitEnv:
        return ExactTransformationCircuitEnv(max_depth, num_qubits, gate_classes, transformation_rules)
    vector_env = SubprocVecEnv([env_fn] * n_envs)

    policy_kwargs = {
        'features_extractor_class': CnnFeaturesExtractor,
    }

    try:
        model = MaskablePPO.load('tce_sb3_ppo.model', vector_env)
    except FileNotFoundError:
        model = MaskablePPO(MaskableActorCriticFcnPolicy, vector_env, policy_kwargs=policy_kwargs, n_steps=n_steps,
                            batch_size=8, tensorboard_log='./tce_logs')

    learn = True
    if learn:
        model.learn(n_iters * n_envs * n_steps, progress_bar=True)
        model.save('tce_sb3_ppo.model')

    env = env_fn()
    env.target_circuit = u
    env.training = False
    env.max_time_steps = 64

    obs, _ = env.reset()
    terminated = False

    total_reward = 0.0
    while not terminated:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        action = int(action)

        obs, reward, terminated, *_ = env.step(action)
        total_reward += reward

    print(env.current_circuit)
    print(f'[bold magenta]Optimized gate count:[/bold magenta] {env.current_circuit.size()}')
    print(f'[bold blue]Optimized depth:[/bold blue] {env.current_circuit.depth()}')
    print(f'[bold yellow]Total reward:[/bold yellow] {total_reward}')


if __name__ == '__main__':
    main()
