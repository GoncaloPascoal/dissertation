
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

from hst import hst, lhst, cost_hst_weighted, cost_hst

def compute_gradient(
    qc: QuantumCircuit,
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
    q: float,
    tolerance: float,
    max_iterations: int,
    sample_precision: float = 1.0e-3,
) -> Tuple[List[float], float]:
    assert 0 < tolerance < 1
    assert 0 <= q <= 1
    assert max_iterations > 0
    assert sample_precision > 0

    sim = AerSimulator(shots=int(1 / sample_precision))
    qc = transpile(hst(u, v.inverse()), sim)
    dimensions = [(-pi, pi) for _ in range(qc.num_parameters)]

    def cost_function(params: np.ndarray) -> float:
        qc_bound = qc.bind_parameters(params)
        counts = sim.run(qc_bound).result().get_counts()
        return cost_hst(counts)

    best_params = np.array([0.0 for _ in range(qc.num_parameters)])
    best_cost = cost_function(best_params)

    i = 0
    grad_count = 0
    learning_rate = 1.0

    while i < max_iterations and grad_count < 4:
        gradient = compute_gradient(qc, cost_function, best_params)
        
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm <= tolerance:
            grad_count += 1

        i += 1

    return best_params.tolist(), best_cost

u = QuantumCircuit(1)
u.h(0)

v = QuantumCircuit(1)
v.rz(Parameter('a'), 0)
v.rx(pi / 2, 0)
v.rz(Parameter('b'), 0)

best_params, best_cost = gradient_based_hst_weighted(u, v, 0.5, 0.01, 50)
best_params_pi = [f'{p / pi:4f}π' for p in best_params]
print(f'The best parameters were {best_params_pi} with a cost of {best_cost}.')
