
import logging
import random
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, List, Optional

import numpy as np
from qiskit import QuantumCircuit
from rich.logging import RichHandler

from auto_parameter import ParameterGenerator
from gradient_based import gradient_based_hst_weighted
from rl import NativeInstruction
from utils import create_native_instruction_dict, ContinuousOptimizationFunction

Genome = List[Tuple[str, Tuple[int, ...]]]


class Crossover(ABC):
    def __init__(self, crossover_rate: float):
        if not (0 <= crossover_rate <= 1):
            raise ValueError(f'Crossover rate must be between 0 and 1, got {crossover_rate}')

        self.crossover_rate = crossover_rate

    def __call__(self, parent_a: Genome, parent_b: Genome) -> Tuple[Genome, Genome]:
        if random.random() < self.crossover_rate:
            return self._crossover(parent_a, parent_b)
        return parent_a, parent_b

    @abstractmethod
    def _crossover(self, parent_a: Genome, parent_b: Genome) -> Tuple[Genome, Genome]:
        raise NotImplementedError


class KPointCrossover(Crossover):
    def __init__(self, crossover_rate: float, k: int):
        if k <= 0:
            raise ValueError(f'The value of k must be positive, got {k}')

        super().__init__(crossover_rate)
        self.k = k

    def _crossover(self, parent_a: Genome, parent_b: Genome) -> Tuple[Genome, Genome]:
        if self.k > len(parent_a):
            raise ValueError('Genome length is smaller than the value of k')

        child_a, child_b = [], []
        indices = list(range(len(parent_a)))
        crossover_points = sorted(random.sample(indices, self.k))

        current = 0
        for i, point in enumerate(crossover_points):
            if i % 2 == 0:
                child_a += parent_a[current:point + 1]
                child_b += parent_b[current:point + 1]
            else:
                child_a += parent_b[current:point + 1]
                child_b += parent_a[current:point + 1]
            current = point + 1

        if len(crossover_points) % 2 == 0:
            child_a += parent_a[current:]
            child_b += parent_b[current:]
        else:
            child_a += parent_b[current:]
            child_b += parent_a[current:]

        return child_a, child_b


class UniformCrossover(Crossover):
    def _crossover(self, parent_a: Genome, parent_b: Genome) -> Tuple[Genome, Genome]:
        child_a, child_b = [], []

        for gene_a, gene_b in zip(parent_a, parent_b):
            if random.random() < 0.5:
                child_a.append(gene_a)
                child_b.append(gene_b)
            else:
                child_a.append(gene_b)
                child_b.append(gene_a)

        return child_a, child_b


class GeneticAlgorithm:
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    _logger.addHandler(RichHandler())

    def __init__(
            self,
            u: QuantumCircuit,
            native_instructions: Sequence[NativeInstruction],
            continuous_optimization: ContinuousOptimizationFunction,
            max_instructions: int,
            tolerance: float,
            max_generations: int,
            crossover: Crossover,
            population_size: int,
            mutation_rate: float = 1e-2,
            num_elites: int = 0,
            seed: Optional[int] = None,
    ):
        if population_size < 4:
            raise ValueError(f'Population size must be at least 4, got {population_size}')
        if population_size % 2 != 0:
            raise ValueError(f'Population size must be even, got {population_size}')

        if max_instructions <= 0:
            raise ValueError(f'Maximum number of instructions must be positive, got {max_instructions}')

        if num_elites < 0:
            raise ValueError(f'Number of elite solutions must not be negative, got {num_elites}')
        if num_elites % 2 != 0:
            raise ValueError(f'Number of elite solutions must be even, got {num_elites}')

        self.u = u
        self.instruction_dict = create_native_instruction_dict(native_instructions)
        self.continuous_optimization = lambda v: continuous_optimization(self.u, v)
        self.max_instructions = max_instructions
        self.tolerance = tolerance

        self.max_generations = max_generations
        self.crossover = crossover
        self.population = [self._generate_solution() for _ in range(population_size)]
        self.generation = 1
        self.mutation_rate = mutation_rate
        self.num_elites = num_elites

        self.best_v = QuantumCircuit(u.num_qubits)
        self.best_params = []
        self.best_cost = 1.0

        random.seed(seed)

    def _generate_solution(self) -> Genome:
        solution = []

        for _ in range(self.max_instructions):
            name = random.choice(list(self.instruction_dict))
            qubits = random.choice(list(self.instruction_dict[name][1]))
            solution.append((name, qubits))

        return solution

    def _parse_solution(self, solution: Genome) -> QuantumCircuit:
        v = QuantumCircuit(self.u.num_qubits)
        gen = ParameterGenerator()

        for name, qubits in solution:
            instruction = self.instruction_dict[name][0]
            if instruction.is_parameterized():
                instruction = gen.parametrize(instruction)
            v.append(instruction, qubits)

        return v

    def _calculate_fitness(self, solution: Genome) -> float:
        v = self._parse_solution(solution)
        params, cost = self.continuous_optimization(v)

        if cost < self.best_cost:
            self.best_v, self.best_params, self.best_cost = v.copy(), params, cost
            self._logger.info(f'Cost decreased to {cost:.4f} in generation {self.generation}')

        return 1.0 - cost

    def _mutate_name(self, name: str) -> str:
        return random.choice([n for n in self.instruction_dict.keys() if n != name])

    def _mutation(self, solution: Genome):
        genes_to_mutate = [g for g in enumerate(solution) if random.random() < self.mutation_rate]

        for i, (name, qubits) in genes_to_mutate:
            if random.random() < 0.5:
                # Mutate name and change qubits only if the number of qubits required by the instruction changes
                # The new set of qubits must share at least one qubit with the previous set
                name = self._mutate_name(name)
                qubit_set = set(qubits)
                possible_qubits = {q for q in self.instruction_dict[name][1] if qubit_set.intersection(q)}
            else:
                # Name wasn't mutated, so change the qubits the instruction is applied to
                possible_qubits = self.instruction_dict[name][1] - {qubits}

            if qubits not in possible_qubits:
                # Qubits are invalid for the target instruction or a mutation in the qubit set occurred
                qubits = random.choice(list(possible_qubits))

            solution[i] = name, qubits

    def run(self) -> Tuple[QuantumCircuit, Sequence[float], float]:
        def get_fitness(s: Tuple[Genome, float]) -> float:
            return s[1]

        fitness_values = []
        for _ in range(self.max_generations):
            if fitness_values:
                # Reuse fitness values from elite solutions
                fitness_values = fitness_values[:self.num_elites]
            fitness_values += [
                (solution, self._calculate_fitness(solution))
                for solution in self.population[self.num_elites:]
            ]

            if self.best_cost <= self.tolerance:
                break

            avg_fitness = np.mean([get_fitness(s) for s in fitness_values])
            self._logger.info(f'Average fitness for generation {self.generation} was {avg_fitness:.4f}')

            new_population = []
            if self.num_elites > 0:
                elites = sorted(fitness_values, key=get_fitness, reverse=True)[:self.num_elites]
                new_population.extend(s[0] for s in elites)

            for _ in range(len(self.population) // 2):
                first_a, first_b, second_a, second_b = random.sample(fitness_values, 4)

                first_parent, _ = max(first_a, first_b, key=get_fitness)
                second_parent, _ = max(second_a, second_b, key=get_fitness)

                first_child, second_child = self.crossover(first_parent, second_parent)

                self._mutation(first_child)
                self._mutation(second_child)

                new_population.append(first_child)
                new_population.append(second_child)

            self.population = new_population
            self.generation += 1

        return self.best_v, self.best_params, self.best_cost


def main():
    from math import pi

    from qiskit.circuit.library import SXGate, RZGate, CXGate
    from qiskit.circuit import Parameter

    u = QuantumCircuit(2)
    u.swap(0, 1)

    sx = SXGate()
    rz = RZGate(Parameter('x'))
    cx = CXGate()

    actions = [
        NativeInstruction(sx, (0,)),
        NativeInstruction(sx, (1,)),
        NativeInstruction(rz, (0,)),
        NativeInstruction(rz, (1,)),
        NativeInstruction(cx, (0, 1)),
        NativeInstruction(cx, (1, 0)),
    ]

    def continuous_optimization(u: QuantumCircuit, v: QuantumCircuit):
        return gradient_based_hst_weighted(u, v)

    algo = GeneticAlgorithm(u, actions, continuous_optimization, 3, 1e-2, 10, KPointCrossover(0.75, 1), 10,
                            mutation_rate=3e-2)

    v, params, cost = algo.run()
    print(v.draw())

    params_pi = [f'{p / pi:.4f}π' for p in params]
    print(f'The best parameters were {params_pi} with a cost of {cost:.4f}.')


if __name__ == '__main__':
    main()
