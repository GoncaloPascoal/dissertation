
from argparse import ArgumentParser
from math import pi
from typing import Callable, Tuple

import numpy as np

from qiskit import QuantumCircuit, QiskitError, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import CXGate, QFT, RZGate, SXGate
from qiskit.providers import Backend
from qiskit.providers.fake_provider import FakeTenerife, FakeRueschlikon

from rich import print

from gradient_based import gradient_based_hst_weighted
from gradient_free import gradient_free_hst
from simulated_annealing import StructuralSimulatedAnnealing


def run_small_scale_implementation(u: QuantumCircuit, backend: Backend, max_iterations: int):
    continuous_optimization = lambda u, v: gradient_based_hst_weighted(u, v)

    print('[bold]Circuit to Compile (U)[/bold]')
    print(u.draw())

    try:
        max_instructions = len(transpile(u, basis_gates=['cx', 'rz', 'sx']))
    except QiskitError:
        # Qiskit transpiler cannot compile identity into native gate set
        max_instructions = 1

    cx = CXGate()
    rz = RZGate(Parameter('x'))
    sx = SXGate()

    num_qubits = u.num_qubits

    native_instructions = (
        [(sx, (q,)) for q in range(num_qubits)] +
        [(rz, (q,)) for q in range(num_qubits)] +
        [
            (cx, tuple(qp)) for qp in backend.configuration().coupling_map
            if all(q < num_qubits for q in qp)
        ]
    )

    sa = StructuralSimulatedAnnealing(max_iterations, u, native_instructions, continuous_optimization, max_instructions)
    v, cost = sa.run()

    print('[bold]Compilation Result (V)[/bold]')
    print(v.draw())

    print(f'The best cost was {cost:.4f}')


def dressed_cnot(param_num: int) -> Tuple[QuantumCircuit, int]:
    qc = QuantumCircuit(2)

    for qubit in [0, 1]:
        qc.rz(Parameter(f'p{param_num}'), qubit)
        param_num += 1
        qc.ry(Parameter(f'p{param_num}'), qubit)
        param_num += 1
        qc.rz(Parameter(f'p{param_num}'), qubit)
        param_num += 1

    qc.cx(0, 1)

    for qubit in [0, 1]:
        qc.rz(Parameter(f'p{param_num}'), qubit)
        param_num += 1
        qc.ry(Parameter(f'p{param_num}'), qubit)
        param_num += 1
        qc.rz(Parameter(f'p{param_num}'), qubit)
        param_num += 1

    return qc, param_num

def alternating_pair_ansatz(num_qubits: int, num_layers: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    param_num = 0
    for l in range(2 * num_layers):
        start = 0 if num_qubits == 2 else l % 2

        for control in range(start, num_qubits - 1, 2):
            d_cx, param_num = dressed_cnot(param_num)
            qc.compose(d_cx.to_gate(label='D.CX'), [control, control + 1], inplace=True)

    return qc

def rz_layer(num_qubits: int, param_num: int) -> Tuple[QuantumCircuit, int]:
    qc = QuantumCircuit(num_qubits)

    for qubit in range(num_qubits):
        qc.rz(Parameter(f'p{param_num}'), qubit)
        param_num += 1

    return qc, param_num

def scaling_ansatz_1(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    l, _ = rz_layer(num_qubits, 0)
    qc.compose(l.to_gate(label='U'), inplace=True)

    return qc

def scaling_ansatz_2(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    u1, param_num = rz_layer(num_qubits, 0)
    qc.compose(u1.to_gate(label='U1'), inplace=True)

    u2 = QuantumCircuit(num_qubits)
    for control in range(0, num_qubits - 1, 2):
        u2.cx(control, control + 1)
    qc.compose(u2.to_gate(label='U2'), inplace=True)

    u3 = QuantumCircuit(num_qubits)
    for control in range(1, num_qubits - 1, 2):
        u3.cx(control, control + 1)
    qc.compose(u3.to_gate(label='U3'), inplace=True)

    u4, _ = rz_layer(num_qubits, param_num)
    qc.compose(u4.to_gate(label='U4'), inplace=True)

    return qc

def run_large_scale_implementation(
    create_ansatz: Callable[[int], QuantumCircuit],
    num_qubits: int,
    q: float,
    max_iterations: int,
):
    u = create_ansatz(num_qubits)
    v = create_ansatz(num_qubits)

    u_params = np.random.uniform(-pi, pi, (u.num_parameters,))
    u.assign_parameters(u_params, inplace=True)
    print('[bold]Target Circuit Parameters (U)[/bold]')
    print([f'{p:.5f}' for p in u_params])

    params, cost = gradient_based_hst_weighted(u, v, q=q, max_iterations=max_iterations)
    params_fmt = [f'{p:.5f}' for p in params]
    print(f'The best parameters were {params_fmt} with a cost of {cost:5f}.')


if __name__ == '__main__':
    qc_i = QuantumCircuit(1)
    qc_i.i(0)

    qc_t = QuantumCircuit(1)
    qc_t.t(0)

    qc_x = QuantumCircuit(1)
    qc_x.x(0)

    qc_h = QuantumCircuit(1)
    qc_h.h(0)

    qc_cz = QuantumCircuit(2)
    qc_cz.cz(0, 1)

    qc_ch = QuantumCircuit(2)
    qc_ch.ch(0, 1)

    small_circuits = {
        'i': qc_i,
        't': qc_t,
        'x': qc_x,
        'h': qc_h,
        'cz': qc_cz,
        'ch': qc_ch,
        'qft2': QFT(2),
    }

    parser = ArgumentParser(description='QAQC Experiment Runner (https://doi.org/10.22331/q-2019-05-13-140)')
    parser.add_argument('-n', '--noiseless', action='store_true', help='run noiseless simulations')

    subparsers = parser.add_subparsers(dest='experiment_class', help='class of experiments to run')

    # Small-scale implementations
    backends = {
        'ibmqx4': FakeTenerife(),
        'ibmqx5': FakeRueschlikon(),
    }

    parser_small = subparsers.add_parser('small', help='small-scale implementations (variable structure)')
    parser_small.add_argument(
        'experiment',
        choices=small_circuits.keys(),
        help='experiment to run',
    )
    parser_small.add_argument(
        '-b', '--backend',
        metavar='B',
        choices=backends.keys(),
        help='Backend to simulate (ibmqx4 - 5 qubits; ibmqx5 - 16 qubits) [default: ibmqx4]',
        default='ibmqx4',
    )
    parser_small.add_argument('-i', '--iterations', metavar='I', type=int, default=50,
        help='maximum number of gradient-free optimization iterations [default: 50]')

    # Large-scale implementations
    ansatz_functions = {
        'sa1': scaling_ansatz_1,
        'sa2': scaling_ansatz_2,
    }

    parser_large = subparsers.add_parser('large', help='large-scale implementations (fixed structure)')
    parser_large.add_argument('ansatz_type', choices=ansatz_functions.keys(),
        help='ansatz type: sa1 - scaling ansatz #1 (single qubit rotations), ' \
             'sa2 - scaling ansatz #2 (entanglement)')
    parser_large.add_argument('num_qubits', type=int, help='qubit count')
    parser_large.add_argument('-q', metavar='Q', type=float, default=1.0,
        help='weight factor for the mixed HST / LHST cost function (0.0 = LHST, 1.0 = HST) [default: 1.0]')
    parser_large.add_argument('-i', '--iterations', metavar='I', type=int, default=50,
        help='maximum number of gradient descent iterations [default: 100]')

    args = parser.parse_args()

    if args.experiment_class == 'small':
        print(f'[bold][blue]Running small-scale experiment [green]{args.experiment}[/green]...[/blue][/bold]\n')
        run_small_scale_implementation(small_circuits[args.experiment], backends[args.backend],
            args.iterations)
    else:
        print(f'[bold][blue]Running small-scale experiment [green]{args.ansatz_type}[/green] ' \
            f'with [green]{args.num_qubits}[/green] qubits...[/blue][/bold]\n')
        run_large_scale_implementation(ansatz_functions[args.ansatz_type], args.num_qubits, args.q,
            args.iterations)
