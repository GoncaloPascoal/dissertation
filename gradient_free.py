
import random
from math import pi
from typing import List, Tuple

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
import skopt

from qiskit.circuit import Parameter
from rich import print

from hst import hst, cost_hst

def gradient_free_hst(
    u: QuantumCircuit,
    v: QuantumCircuit,
    tolerance: float = 0.01,
    max_starting_points: int = 10,
    max_iterations: int = 50,
    sample_precision: float = 1.25e-4,
) -> Tuple[List[float], float]:
    """Performs gradient-free optimization for QAQC using the HST."""
    assert 0 < tolerance < 1
    assert max_starting_points > 0
    assert max_iterations > 0
    assert sample_precision > 0

    sim = AerSimulator(shots=int(1 / sample_precision))
    qc = transpile(hst(u, v.inverse()), sim)
    dimensions = [(-pi, pi) for _ in range(qc.num_parameters)]

    def cost_function(params: List[float]) -> float:
            qc_bound = qc.bind_parameters(params)
            counts = sim.run(qc_bound).result().get_counts()
            return cost_hst(counts)

    best_params = [0.0 for _ in range(qc.num_parameters)]
    best_cost = 1.0

    i = 0
    while best_cost > tolerance and i < max_starting_points:
        current_params = [random.uniform(-pi, pi) for _ in range(qc.num_parameters)]
        res = skopt.gp_minimize(cost_function, dimensions, n_calls=max_iterations, x0=current_params)
        params, cost = res.x, res.fun

        if cost <= best_cost:
            best_params, best_cost = params, cost
        i += 1

    return best_params, best_cost

u = QuantumCircuit(1)
u.h(0)

v = QuantumCircuit(1)
v.rz(Parameter('a'), 0)
v.rx(pi / 2, 0)
v.rz(Parameter('b'), 0)

best_params, best_cost = gradient_free_hst(u, v)
best_params_pi = [f'{p / pi:4f}π' for p in best_params]
print(f'The best parameters were {best_params_pi} with a cost of {best_cost}.')
