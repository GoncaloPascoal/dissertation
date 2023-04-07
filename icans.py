
from typing import Callable, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend
from qiskit_aer import AerSimulator

from hst import HilbertSchmidt, LocalHilbertSchmidt, cost_hst_weighted
from utils import ContinuousOptimizationResult


def icans(
    cost_function: Callable[[np.ndarray, int], float],
    params: np.ndarray,
    min_shots: int,
    total_shots: int,
    lipschitz: float,
    learning_rate: float,
    running_average: float,
    gradient_norm_bias: float,
) -> np.ndarray:
    assert min_shots > 0
    assert total_shots > 0
    assert learning_rate > 0

    num_shots = 0
    shots_arr = np.full(params.shape, min_shots)
    chi_prime = np.zeros(params.shape)
    xi_prime = np.zeros(params.shape)
    k = 0

    while num_shots < total_shots:
        gradient = np.zeros(params.shape)
        variances = np.zeros(params.shape)
        for i, shots in enumerate(shots_arr):
            gradient[i] = (cost_function(params, shots))

        num_shots += 2 * shots_arr.sum()

        chi_prime = running_average * chi_prime + (1 - running_average) * gradient
        xi_prime = running_average * xi_prime + (1 - running_average) * variances

        div = 1 - pow(running_average, k + 1)
        chi = chi_prime / div
        xi = xi_prime / div

        params -= learning_rate * gradient

        shots_arr: np.ndarray = np.ceil(
            (2 * lipschitz * learning_rate) / (2 - lipschitz * learning_rate) *
            xi / (pow(chi, 2) + gradient_norm_bias * pow(running_average, k))
        )

        expected_gain_per_shot: np.ndarray = (
            (learning_rate - (lipschitz * pow(learning_rate, 2)) / 2) * pow(chi, 2) -
            (learning_rate * pow(learning_rate, 2)) / (2 * shots_arr) * xi
        ) / shots_arr

        max_shots = shots_arr[expected_gain_per_shot.argmax()[0]]
        shots_arr = np.clip(shots_arr, min_shots, max_shots)

        k += 1

    return params


def icans_hst_weighted(
    u: QuantumCircuit,
    v: QuantumCircuit,
    min_shots: int,
    total_shots: int,
    lipschitz: float,
    learning_rate: float = 0.1,
    running_average: float = 0.99,
    gradient_norm_bias: float = 1e-6,
    q: float = 1.0,
    backend: Optional[Backend] = None,
) -> ContinuousOptimizationResult:
    if backend is None:
        sim = AerSimulator()
    else:
        sim = AerSimulator.from_backend(backend)

    qc_hst = transpile(HilbertSchmidt(u, v), sim)
    qc_lhst = transpile([LocalHilbertSchmidt(u, v, i) for i in range(u.num_qubits)], sim)

    def cost_function(params: np.ndarray, shots: int) -> float:
        counts = {}
        if q != 0.0:
            qc_hst_bound = qc_hst.bind_parameters(params)
            counts = sim.run(qc_hst_bound, shots=shots).result().get_counts()

        counts_list = []
        if q != 1.0:
            qc_lhst_bound = [c.bind_parameters(params) for c in qc_lhst]
            counts_list = sim.run(qc_lhst_bound, shots=shots).result().get_counts()
            if isinstance(counts_list, dict):
                counts_list = [counts_list]

        return cost_hst_weighted(counts, counts_list, q)

    best_params = icans(
        cost_function, np.zeros(v.num_parameters), min_shots, total_shots,
        lipschitz, learning_rate, running_average, gradient_norm_bias,
    )
    best_cost = cost_function(best_params, 1000)

    return ContinuousOptimizationResult(best_params.tolist(), best_cost)
