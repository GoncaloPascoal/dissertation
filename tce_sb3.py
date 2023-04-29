
from qiskit import QuantumCircuit, transpile
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from tce import ExactTransformationCircuitEnv
from exact import CommuteGates, InvertCnot
from gate_class import GateClass, generate_two_qubit_gate_classes_from_coupling_map


def main():
    from qiskit.circuit.library import RZGate, SXGate, QFT
    from qiskit.circuit import Parameter
    from rich import print

    max_depth = 48
    num_qubits = 3

    # Nearest neighbor coupling
    coupling_map = [(q, q + 1) for q in range(num_qubits - 1)]

    u = QuantumCircuit(3)
    u.toffoli(0, 1, 2)
    u = transpile(u, basis_gates=['cx', 'sx', 'rz'], coupling_map=[list(pair) for pair in coupling_map],
                  approximation_degree=0.0, seed_transpiler=1)
    print(u)
    print(u.depth())

    rz = RZGate(Parameter('x'))
    sx = SXGate()

    gate_classes = [
        GateClass(rz),
        GateClass(sx),
        *generate_two_qubit_gate_classes_from_coupling_map(coupling_map),
    ]

    transformation_rules = [
        CommuteGates(),
        InvertCnot(),
    ]

    env = ExactTransformationCircuitEnv(max_depth, num_qubits, gate_classes, transformation_rules)

    try:
        model = MaskablePPO.load('tce_sb3_ppo.model', env)
    except FileNotFoundError:
        model = MaskablePPO(MaskableActorCriticPolicy, env, n_steps=512, batch_size=16, tensorboard_log='./tce_logs',
                            learning_rate=1e-3, device='cuda')

    learn = False
    if learn:
        model.learn(4096, progress_bar=True)
        model.save('tce_sb3_ppo.model')

    env.target_circuit = u
    env.max_time_steps = 128
    obs, _ = env.reset()
    terminated = False

    total_reward = 0.0
    while not terminated:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
        obs, reward, terminated, *_ = env.step(action)
        total_reward += reward

    print(env.current_circuit)
    print(env.current_circuit.depth())
    print(f'{total_reward = }')


if __name__ == '__main__':
    main()
