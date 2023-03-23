
from math import pi
from typing import Callable, Tuple, List

import matplotlib.pyplot as plt

import numpy as np

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.algorithms.gradients import ParamShiftEstimatorGradient
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit import Parameter
from rich import print

from hst import hst, lhst, cost_hst_weighted

def _create_cost_function(
    u: QuantumCircuit,
    v: QuantumCircuit,
    q: float,
    sample_precision: float,
) -> Callable[[np.ndarray], float]:
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

    return cost_function

def compute_gradient(
    cost_function: Callable[[np.ndarray], float],
    params: np.ndarray,
) -> np.ndarray:
    """
    Computes the gradient of a cost function using the parameter shift rule,
    under the assumption that the only parametrized gates are single-qubit
    rotations.

    :param cost_function: a callable that accepts a NumPy array of parameters
     and returns the cost value
    :param params: point in parameter space where the gradient will be computed
    """

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

def visualize_gradient(u: QuantumCircuit, v: QuantumCircuit, samples: int = 17):
    assert v.num_parameters == 2

    cost_function = _create_cost_function(u, v, 0.5, 1.0e-3)
    fig, ax = plt.subplots()

    x, y, u, v, cost = [], [], [], [], []
    param_values = np.linspace(-pi, pi, samples)

    for theta_0 in param_values:
        for theta_1 in param_values:
            params = np.array([theta_0, theta_1])
            gradient = compute_gradient(cost_function, params)
            
            x.append(theta_0)
            y.append(theta_1)
            u.append(-gradient[0])
            v.append(-gradient[1])

            cost.append(cost_function(params))

    ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)
    ax.scatter(x, y, c=cost, cmap='inferno')
    ax.set_xticks(param_values)
    ax.set_yticks(param_values)

    plt.show()
    

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

    cost_function = _create_cost_function(u, v, q, sample_precision)

    params = np.array([0.0 for _ in range(v.num_parameters)])
    cost = cost_function(params)

    i = 0
    grad_count = 0
    learning_rate = 1.0

    while i < max_iterations and grad_count < 4:
        gradient = compute_gradient(cost_function, params)

        gradient_norm = np.linalg.norm(gradient) ** 2
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

if __name__ == '__main__':
    u = QuantumCircuit(1)
    u.h(0)

    v = QuantumCircuit(1)
    v.rz(Parameter('a'), 0)
    v.sx(0)
    v.rz(Parameter('b'), 0)

    visualize_gradient(u, v)

    best_params, best_cost = gradient_based_hst_weighted(u, v)
    best_params_pi = [f'{p / pi:4f}π' for p in best_params]
    print(f'The best parameters were {best_params_pi} with a cost of {best_cost}.')
