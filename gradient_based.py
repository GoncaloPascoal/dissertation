
from math import pi
from typing import Callable, Tuple, List

import numpy as np

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.algorithms.gradients import ParamShiftEstimatorGradient
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit import Parameter
from rich import print

from hst import hst, lhst, cost_hst_weighted

def compute_gradient(
    cost_function: Callable[[np.ndarray], float],
    params: np.ndarray
) -> np.ndarray:
    gradient = []

    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += pi / 2

        params_minus = params.copy()
        params_minus[i] -= pi / 2

        cost_plus = cost_function(params_plus)
        cost_minus = cost_function(params_minus)

        partial = 0.5 * (cost_plus - cost_minus)
        gradient.append(partial)

    return np.array(gradient)

def gradient_based_hst_weighted(
    u: QuantumCircuit,
    v: QuantumCircuit,
    q: float = 1.0,
    tolerance: float = 0.01,
    max_iterations: int = 50,
    sample_precision: float = 1.0e-3,
) -> Tuple[List[float], float]:
    assert 0 < tolerance < 1
    assert 0 <= q <= 1
    assert max_iterations > 0
    assert sample_precision > 0

    v_adj = v.inverse()

    sim = AerSimulator(shots=int(1 / sample_precision))
    qc_hst = transpile(hst(u, v_adj), sim)
    qc_lhst = transpile([lhst(u, v_adj, i) for i in range(u.num_qubits)], sim)

    def cost_function(params: np.ndarray) -> float:
        qc_hst_bound = qc_hst.bind_parameters(params)
        counts = sim.run(qc_hst_bound).result().get_counts()

        qc_lhst_bound = [c.bind_parameters(params) for c in qc_lhst]
        counts_list = sim.run(qc_lhst_bound).result().get_counts()
        if isinstance(counts_list, dict):
            counts_list = [counts_list]

        return cost_hst_weighted(counts, counts_list, q)

    params = np.array([0.0 for _ in range(v.num_parameters)])
    cost = cost_function(params)

    i = 0
    grad_count = 0
    learning_rate = 1.0

    while i < max_iterations and grad_count < 4:
        gradient = compute_gradient(cost_function, params)

        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm <= tolerance:
            grad_count += 1

        params_1 = params - learning_rate * gradient
        params_2 = params_1 - learning_rate * gradient

        cost_2 = cost_function(params_2)
        if cost - cost_2 >= learning_rate * gradient_norm:
            learning_rate *= 2
            params = params_2
            cost = cost_2
        else:
            cost_1 = cost_function(params_1)
            if cost - cost_1 < learning_rate / 2 * gradient_norm:
                learning_rate /= 2
            params = params_1
            cost = cost_1

        i += 1

    return params.tolist(), cost

u = QuantumCircuit(1)
u.h(0)

v = QuantumCircuit(1)
v.rz(Parameter('a'), 0)
v.rx(pi / 2, 0)
v.rz(Parameter('b'), 0)

best_params, best_cost = gradient_based_hst_weighted(u, v)
best_params_pi = [f'{p / pi:4f}π' for p in best_params]
print(f'The best parameters were {best_params_pi} with a cost of {best_cost}.')
