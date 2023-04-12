
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Sequence, Tuple, List, Optional

import numpy as np
from qiskit import QuantumCircuit
from rich.logging import RichHandler

from auto_parameter import ParameterGenerator
from gradient_based import gradient_based_hst_weighted
from rl import NativeInstruction
from utils import create_native_instruction_dict, ContinuousOptimizationFunction


Genome = List[Tuple[str, Tuple[int, ...]]]


@dataclass(slots=True)
class Solution:
    genome: Genome
    fitness: float = 0.0

    # Class variables
    similarity_name_weight: ClassVar[float] = 0.75
    similarity_qubits_weight: ClassVar[float] = 1.0 - similarity_name_weight

    def sort_key(self) -> float:
        """
        Used as a key callable for sorting solutions. Fitness should take priority when sorting,
        but it is an optional field therefore it must be defined after the required genome field.
        """
        return self.fitness

    def similarity(self, other: 'Solution') -> float:
        """
        Rates the similarity between two solutions, based on their genomes.
        A value of 0 indicates the genomes are completely different, while 1 is returned if they are identical.
        """
        if len(self.genome) != len(other.genome):
            raise ValueError('Solutions have genomes with different lengths')

        score = 0.0
        gene_inc = 1.0 / len(self.genome)

        for (name_a, qubits_a), (name_b, qubits_b) in zip(self.genome, other.genome):
            if name_a == name_b:
                score += Solution.similarity_name_weight * gene_inc

            common_qubits = set(qubits_a) & set(qubits_b)
            if common_qubits:
                score += (
                    Solution.similarity_qubits_weight * gene_inc *
                    len(common_qubits) / max(len(qubits_a), len(qubits_b))
                )

        return score


class Selection(ABC):
    def __init__(self, selection_ratio: float = 1.0):
        if not (0 <= selection_ratio <= 1):
            raise ValueError(f'Selection ratio must be between 0 and 1, got {selection_ratio}')

        self.selection_ratio = selection_ratio

    def num_selections(self, population: Sequence[Solution]):
        # Divide population size by two since two parents are selected in each iteration
        return round(self.selection_ratio * len(population) / 2)

    @abstractmethod
    def __call__(self, population: List[Solution]) -> List[Tuple[Solution, Solution]]:
        raise NotImplementedError


class TournamentSelection(Selection):
    def __init__(
        self,
        selection_ratio: float = 1.0,
        tournament_size: int = 2,
        with_replacement: bool = True,
        similarity_penalty: Optional[float] = None,
    ):
        super().__init__(selection_ratio)

        if tournament_size <= 0:
            raise ValueError(f'Tournament size must be positive, got {tournament_size}')

        self.tournament_size = tournament_size
        self.with_replacement = with_replacement
        self.similarity_penalty = similarity_penalty

    def __call__(self, population: List[Solution]) -> List[Tuple[Solution, Solution]]:
        pairs = []

        if not self.with_replacement:
            population = population.copy()

        for _ in range(self.num_selections(population)):
            it_population = population.copy()

            first_parent = max(random.sample(it_population, self.tournament_size), key=Solution.sort_key)
            it_population.remove(first_parent)

            key = Solution.sort_key
            if self.similarity_penalty is not None:
                def key(s: Solution): return s.sort_key() - self.similarity_penalty * s.similarity(first_parent)

            second_parent = max(random.sample(it_population, self.tournament_size), key=key)

            pairs.append((first_parent, second_parent))

            if not self.with_replacement:
                population.remove(first_parent)
                population.remove(second_parent)

        return pairs


class RouletteWheelSelection(Selection):
    def __init__(self, selection_ratio: float = 1.0, with_replacement: bool = True):
        super().__init__(selection_ratio)

        self.with_replacement = with_replacement

    def __call__(self, population: List[Solution]) -> List[Tuple[Solution, Solution]]:
        pairs = []

        if not self.with_replacement:
            population = population.copy()

        for _ in range(self.num_selections(population)):
            it_population = population.copy()

            first_parent = random.choices(it_population, weights=[s.sort_key() for s in it_population])[0]
            it_population.remove(first_parent)
            second_parent = random.choices(it_population, weights=[s.sort_key() for s in it_population])[0]

            pairs.append((first_parent, second_parent))

            if not self.with_replacement:
                population.remove(first_parent)
                population.remove(second_parent)

        return pairs


class StochasticUniversalSampling(Selection):
    def __call__(self, population: List[Solution]) -> List[Tuple[Solution, Solution]]:
        selected = []
        num_selections = 2 * self.num_selections(population)

        # Sort population in descending order of fitness
        population = sorted(population, key=Solution.sort_key, reverse=True)
        total_fitness = sum(s.fitness for s in population)
        interval = total_fitness / num_selections

        accumulator = 0.0
        r = random.random() * interval
        for solution in population:
            accumulator += solution.fitness

            if r <= accumulator:
                selected.append(solution)
                r += interval

            if r > total_fitness:
                break

        random.shuffle(selected)
        pairs = [(selected[i], selected[i + 1]) for i in range(0, len(selected), 2)]
        return pairs


class Crossover(ABC):
    def __init__(self, crossover_rate: float = 1.0):
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
    def __init__(self, k: int, crossover_rate: float = 1.0):
        super().__init__(crossover_rate)

        if k <= 0:
            raise ValueError(f'The value of k must be positive, got {k}')

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
        selection: Selection,
        crossover: Crossover,
        population_size: int,
        mutation_rate: float = 1e-2,
        seed: Optional[int] = None,
        cache_fitness: bool = True,
    ):
        if population_size < 4:
            raise ValueError(f'Population size must be at least 4, got {population_size}')
        if population_size % 2 != 0:
            raise ValueError(f'Population size must be even, got {population_size}')

        if max_instructions <= 0:
            raise ValueError(f'Maximum number of instructions must be positive, got {max_instructions}')

        self.u = u
        self.instruction_dict = create_native_instruction_dict(native_instructions)
        self.continuous_optimization = lambda v: continuous_optimization(self.u, v)
        self.max_instructions = max_instructions
        self.tolerance = tolerance

        self.max_generations = max_generations
        self.selection = selection
        self.crossover = crossover
        self.population = [self._generate_solution() for _ in range(population_size)]
        self.generation = 1
        self.mutation_rate = mutation_rate
        self.num_elites = population_size - 2 * self.selection.num_selections(self.population)

        self.best_v = QuantumCircuit(u.num_qubits)
        self.best_params = []
        self.best_cost = 1.0

        random.seed(seed)

        self.cache_fitness = cache_fitness
        self.solution_cache = {}

    def _generate_solution(self) -> Solution:
        genome = []

        for _ in range(self.max_instructions):
            name = random.choice(list(self.instruction_dict))
            qubits = random.choice(list(self.instruction_dict[name][1]))
            genome.append((name, qubits))

        return Solution(genome)

    def _parse_genome(self, genome: Genome) -> QuantumCircuit:
        v = QuantumCircuit(self.u.num_qubits)
        gen = ParameterGenerator()

        for name, qubits in genome:
            instruction = self.instruction_dict[name][0]
            if instruction.is_parameterized():
                instruction = gen.parametrize(instruction)
            v.append(instruction, qubits)

        return v

    def _calculate_fitness(self, solution: Solution) -> float:
        if self.cache_fitness:
            try:
                return self.solution_cache[tuple(solution.genome)]
            except KeyError:
                pass

        v = self._parse_genome(solution.genome)
        params, cost = self.continuous_optimization(v)

        if cost < self.best_cost:
            self.best_v, self.best_params, self.best_cost = v.copy(), params, cost
            self._logger.info(f'Cost decreased to {cost:.4f} in generation {self.generation}')

        fitness = 1.0 - cost
        if self.cache_fitness:
            self.solution_cache[tuple(solution.genome)] = fitness

        return fitness

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
        for _ in range(self.max_generations):
            # Reuse fitness values from elite solutions
            to_calculate = self.population[self.num_elites:] if self.generation != 1 else self.population
            for solution in to_calculate:
                solution.fitness = self._calculate_fitness(solution)

            if self.best_cost <= self.tolerance:
                break

            avg_fitness = np.mean([solution.fitness for solution in self.population])
            self._logger.info(f'Average fitness for generation {self.generation} was {avg_fitness:.4f}')

            if self.cache_fitness:
                self._logger.info(f'Explored {len(self.solution_cache)} unique solutions')

            new_population = []
            if self.num_elites > 0:
                elites = sorted(self.population, key=Solution.sort_key, reverse=True)[:self.num_elites]
                new_population.extend(elites)

            for first_parent, second_parent in self.selection(self.population):
                first_genome, second_genome = self.crossover(first_parent.genome, second_parent.genome)

                self._mutation(first_genome)
                self._mutation(second_genome)

                new_population.append(Solution(first_genome))
                new_population.append(Solution(second_genome))

            self.population = new_population
            self.generation += 1

        return self.best_v, self.best_params, self.best_cost


def main():
    from math import pi

    from qiskit.circuit.library import SXGate, RZGate, CXGate, QFT
    from qiskit.circuit import Parameter

    u = QuantumCircuit(2)
    u.ch(0, 1)

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
        return gradient_based_hst_weighted(u, v, sample_precision=5e-4)

    algo = GeneticAlgorithm(u, actions, continuous_optimization, 6, 1e-2, 15,
                            TournamentSelection(),
                            KPointCrossover(1), 20, mutation_rate=3e-2)

    v, params, cost = algo.run()
    print(v.draw())

    params_pi = [f'{p / pi:.4f}π' for p in params]
    print(f'The best parameters were {params_pi} with a cost of {cost:.4f}.')


if __name__ == '__main__':
    main()
