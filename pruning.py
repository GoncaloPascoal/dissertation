
from math import pi
from typing import Sequence, Tuple

from qiskit import QuantumCircuit

from utils import ContinuousOptimizationFunction, remove_parametrized_instruction


def iterative_pruning(
    u: QuantumCircuit,
    v: QuantumCircuit,
    continuous_optimization: ContinuousOptimizationFunction,
    tolerance: float = 0.01,
) -> Tuple[QuantumCircuit, Sequence[float], float]:
    def select_param(params: Sequence[float]) -> int:
        normalized = [p / (2 * pi) for p in params]
        deltas = [abs(p - round(p)) for p in normalized]

        idx = 0
        for i, delta in enumerate(deltas):
            if delta < deltas[idx]:
                idx = i

        return idx

    params, cost = continuous_optimization(u, v)

    while params:
        idx = select_param(params)

        new_v = remove_parametrized_instruction(v, idx)
        new_params, new_cost = continuous_optimization(u, new_v)

        cost_delta = abs(new_cost - cost)
        if cost_delta > tolerance:
            break

        v, params, cost = new_v, new_params, new_cost

    return v, params, cost


def main():
    from qiskit.circuit import Parameter

    from gradient_based import gradient_based_hst_weighted

    u = QuantumCircuit(1)
    v = QuantumCircuit(1)

    u.h(0)

    v.rz(Parameter('a'), 0)
    v.sx(0)
    v.rz(Parameter('b'), 0)
    v.rz(Parameter('c'), 0)

    v, params, cost = iterative_pruning(u, v, gradient_based_hst_weighted)
    print(v.draw())

    params_pi = [f'{p / pi:4f}π' for p in params]
    print(f'The best parameters were {params_pi} with a cost of {cost}.')


if __name__ == '__main__':
    main()
