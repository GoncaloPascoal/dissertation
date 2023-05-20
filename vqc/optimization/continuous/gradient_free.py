
import logging
from fractions import Fraction
from math import pi
from typing import Optional, Sequence, Callable, List

import numpy as np
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.providers.backend import Backend
import skopt

from hst import HilbertSchmidt, cost_hst
from vqc.optimization.discrete.simulated_annealing import SimulatedAnnealing
from utils import ContinuousOptimizationResult


def gradient_free_hst(
    u: QuantumCircuit,
    v: QuantumCircuit,
    tolerance: float = 0.01,
    max_starting_points: int = 10,
    max_iterations: int = 50,
    sample_precision: float = 1.0e-5,
    backend: Optional[Backend] = None,
) -> ContinuousOptimizationResult:
    """Performs gradient-free optimization for QAQC using the HST."""
    assert 0 < tolerance < 1
    assert max_starting_points > 0
    assert max_iterations > 0
    assert sample_precision > 0

    if backend is None:
        sim = AerSimulator()
    else:
        sim = AerSimulator.from_backend(backend)
    sim.set_option('shots', int(1 / sample_precision))

    qc = transpile(HilbertSchmidt(u, v), sim)
    dimensions = [(-pi, pi)] * qc.num_parameters

    def cost_function(params: Sequence[float]) -> float:
        qc_bound = qc.bind_parameters(params)
        counts = sim.run(qc_bound).result().get_counts()
        return cost_hst(counts)

    best_params = [0.0] * qc.num_parameters
    best_cost = cost_function(best_params)

    if qc.num_parameters != 0:
        i = 0
        while best_cost > tolerance and i < max_starting_points:
            params = np.random.uniform(-pi, pi, qc.num_parameters).tolist()

            res = skopt.gp_minimize(cost_function, dimensions, n_calls=max_iterations, x0=params)
            params, cost = res.x, res.fun

            if cost <= best_cost:
                best_params, best_cost = params, cost
            i += 1

    return ContinuousOptimizationResult(best_params, best_cost)


class BisectionSimulatedAnnealing(SimulatedAnnealing[List[float]]):
    def __init__(
        self,
        max_iterations: int,
        v: QuantumCircuit,
        initial_params: List[float],
        cost_function: Callable[[Sequence[float]], float],
        angles: List[Fraction],
        **kwargs,
    ):
        self.v = v
        self.initial_params = initial_params
        self.cost_function = cost_function

        super().__init__(max_iterations, **kwargs)

        self.angles = angles

    def _generate_initial_solution(self) -> List[float]:
        return self.initial_params

    def _generate_neighbor(self) -> List[float]:
        param_idx = self.rng.randrange(len(self.current_solution))

        neighbor = self.current_solution.copy()
        neighbor[param_idx] = pi * self.rng.choice(self.angles)

        return neighbor

    def _evaluate_solution(self, solution: List[float]) -> float:
        return self.cost_function(solution)

    def _copy_solution(self, solution: List[float]) -> List[float]:
        return solution.copy()

    def _terminated(self) -> bool:
        return self.best_cost <= self.tolerance


def gradient_free_bisection(
    u: QuantumCircuit,
    v: QuantumCircuit,
    tolerance: float = 0.01,
    max_iterations: int = 50,
    max_bisections: int = 5,
    sample_precision: float = 1.25e-4,
    backend: Optional[Backend] = None,
) -> ContinuousOptimizationResult:
    assert 0 < tolerance < 1
    assert max_iterations > 0
    assert max_bisections > 0
    assert sample_precision > 0

    if backend is None:
        sim = AerSimulator()
    else:
        sim = AerSimulator.from_backend(backend)
    sim.set_option('shots', int(1 / sample_precision))

    qc = transpile(HilbertSchmidt(u, v), sim)

    def cost_function(params: Sequence[float]) -> float:
        qc_bound = qc.bind_parameters(params)
        counts = sim.run(qc_bound).result().get_counts()
        return cost_hst(counts)

    best_params = [0.0] * qc.num_parameters
    best_cost = cost_function(best_params)

    if qc.num_parameters == 0:
        return ContinuousOptimizationResult(best_params, best_cost)

    angles = {Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2)}

    for param_idx in range(qc.num_parameters):
        params = best_params.copy()

        for angle in angles:
            params[param_idx] = pi * angle
            cost = cost_function(params)
            if cost < best_cost:
                best_params, best_cost = params.copy(), cost

    simulated_annealing = BisectionSimulatedAnnealing(
        max_iterations, v, best_params, cost_function, list(angles),
        log_level=logging.ERROR,
    )

    for t in range(1, max_bisections + 1):
        inc = Fraction(1, 2 ** (t + 1))

        new_angles = set()
        for angle in angles:
            for new_angle in [angle + inc, angle - inc]:
                if 0 <= new_angle < 2:
                    new_angles.add(new_angle)

        angles.update(new_angles)
        simulated_annealing.angles = list(angles)
        simulated_annealing.current_solution = best_params

        best_params, best_cost = simulated_annealing.run()

        if best_cost <= tolerance:
            break

    return ContinuousOptimizationResult(best_params, best_cost)


def main():
    from qiskit.circuit import Parameter
    from rich import print

    u = QuantumCircuit(2)
    u.ch(0, 1)

    v = QuantumCircuit(2)
    v.rz(Parameter('a'), 1)
    v.sx(1)
    v.rz(Parameter('b'), 1)
    v.cx(0, 1)
    v.rz(Parameter('c'), 1)
    v.sx(1)

    best_params, best_cost = gradient_free_bisection(u, v)
    best_params_pi = [f'{p / pi:4f}π' for p in best_params]
    print(f'The best parameters were {best_params_pi} with a cost of {best_cost}.')


if __name__ == '__main__':
    main()
