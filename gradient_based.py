
from math import pi
from typing import Tuple, List

import numpy as np

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile

from hst import hst, lhst, cost_hst_weighted

def compute_gradient(qc: QuantumCircuit, params: np.ndarray) -> np.ndarray:
    gradient = []
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
        return cost_hst_weighted(counts, counts_list, q)

    best_params = np.array([0.0 for _ in range(qc.num_parameters)])
    best_cost = cost_function(best_params)

    learning_rate = 1.0

    return best_params.tolist(), best_cost
