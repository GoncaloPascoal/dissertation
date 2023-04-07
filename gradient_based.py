
from math import pi
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt

import numpy as np

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit import QuantumCircuit, transpile
from qiskit.providers.backend import Backend

from hst import HilbertSchmidt, LocalHilbertSchmidt, cost_hst_weighted
from utils import ContinuousOptimizationResult

def _create_cost_function(
    u: QuantumCircuit,
    v: QuantumCircuit,
    q: float,
    sample_precision: float,
    backend: Optional[Backend] = None,
) -> Callable[[np.ndarray], float]:
    if backend is None:
        sim = AerSimulator()
    else:
        sim = AerSimulator.from_backend(backend)
    sim.set_option('shots', int(1 / sample_precision))

    qc_hst = transpile(HilbertSchmidt(u, v), sim)
    qc_lhst = transpile([LocalHilbertSchmidt(u, v, i) for i in range(u.num_qubits)], sim)

    def cost_function(params: np.ndarray) -> float:
        counts = {}
        if q != 0.0:
            qc_hst_bound = qc_hst.bind_parameters(params)
            counts = sim.run(qc_hst_bound).result().get_counts()

        counts_list = []
        if q != 1.0:
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

def visualize_gradient(
    u: QuantumCircuit,
    v: QuantumCircuit,
    q: float = 1.0,
    samples: int = 17,
    base_params: Optional[np.ndarray] = None,
    param_indices: Tuple[int, int] = (0, 1),
):
    assert v.num_parameters >= 2

    if base_params is None:
        base_params = np.zeros((v.num_parameters,))

    assert len(base_params) == v.num_parameters

    cost_function = _create_cost_function(u, v, q, 1.0e-3)
    ax = plt.axes()

    x, y, u, v, cost = [], [], [], [], []
    param_values = np.linspace(-pi, pi, samples)

    for theta_0 in param_values:
        for theta_1 in param_values:
            base_params[param_indices[0]] = theta_0
            base_params[param_indices[1]] = theta_1
            gradient = compute_gradient(cost_function, base_params)
            
            x.append(theta_0)
            y.append(theta_1)
            u.append(-gradient[param_indices[0]])
            v.append(-gradient[param_indices[1]])

            cost.append(cost_function(base_params))

    ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)
    ax.scatter(x, y, c=cost, cmap='inferno')

    ax.set_title('Gradient Visualization')
    ax.set_xlabel(f'Parameter #{param_indices[0]}')
    ax.set_ylabel(f'Parameter #{param_indices[1]}')
    ax.set_xticks(param_values)
    ax.set_yticks(param_values)

    plt.show()

def visualize_cost_function(
    u: QuantumCircuit,
    v: QuantumCircuit,
    q: float = 1.0,
    samples: int = 17,
    base_params: Optional[np.ndarray] = None,
    param_indices: Tuple[int, int] = (0, 1),
):
    assert v.num_parameters >= 2

    if base_params is None:
        params = np.zeros((v.num_parameters,))
    else:
        params = base_params.copy()

    assert len(params) == v.num_parameters

    cost_function = _create_cost_function(u, v, q, 1e-3)
    ax = plt.axes(projection='3d')

    @np.vectorize
    def f(x: float, y: float) -> float:
        params[param_indices[0]] = x
        params[param_indices[1]] = y
        return cost_function(params)

    param_values = np.linspace(-pi, pi, samples)
    x, y = np.meshgrid(param_values, param_values)

    ax.plot_surface(x, y, f(x, y), cmap='inferno')

    ax.set_title('Cost Function Visualization')
    ax.set_xlabel(f'Parameter #{param_indices[0]}')
    ax.set_ylabel(f'Parameter #{param_indices[1]}')

    plt.show()


def gradient_based_hst_weighted(
    u: QuantumCircuit,
    v: QuantumCircuit,
    q: float = 1.0,
    tolerance: float = 1.0e-3,
    max_iterations: int = 50,
    sample_precision: float = 1.0e-3,
    noise_model: Optional[NoiseModel] = None,
) -> ContinuousOptimizationResult:
    assert 0 < tolerance < 1
    assert 0 <= q <= 1
    assert max_iterations > 0
    assert sample_precision > 0

    cost_function = _create_cost_function(u, v, q, sample_precision, noise_model)

    params = np.zeros(v.num_parameters)
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

    return ContinuousOptimizationResult(params.tolist(), cost)


if __name__ == '__main__':
    from qiskit.circuit import Parameter
    from rich import print
    from utils import normalize_angles

    # u = QuantumCircuit(1)
    # u.h(0)

    # v = QuantumCircuit(1)
    # v.rz(Parameter('a'), 0)
    # v.sx(0)
    # v.rz(Parameter('b'), 0)

    # visualize_gradient(u, v)

    # Controlled Hadamard
    u = QuantumCircuit(2)
    u.ch(0, 1)

    v = QuantumCircuit(2)
    v.rz(Parameter('a'), 1)
    v.sx(1)
    v.rz(Parameter('b'), 1)
    v.cx(0, 1)
    v.rz(Parameter('c'), 1)
    v.sx(1)

    best_params, best_cost = gradient_based_hst_weighted(u, v, q=0.5, tolerance=1e-8)
    best_params_pi = [f'{p / pi:4f}π' for p in normalize_angles(best_params)]
    print(f'The best parameters were {best_params_pi} with a cost of {best_cost}.')
