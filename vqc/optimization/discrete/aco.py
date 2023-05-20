
import random

from typing import List, Sequence, Tuple

from qiskit import QuantumCircuit
import rustworkx as rx

from utils import ContinuousOptimizationFunction, NativeInstruction


class AntColonyOptimization:
    def __init__(
        self,
        u: QuantumCircuit,
        native_instructions: Sequence[NativeInstruction],
        continuous_optimization: ContinuousOptimizationFunction,
        max_depth: int,
        num_ants: int,
        evaporation_rate: float,
    ):
        assert num_ants > 0
        assert 0.0 <= evaporation_rate <= 1.0

        self.u = u
        self.native_instructions = native_instructions
        self.continuous_optimization = continuous_optimization

        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate

        self.pheromone_graph = rx.PyDAG()
        self.start_node = self.pheromone_graph.add_node(None)
        self.end_node = self.pheromone_graph.add_node(None)

        previous_layer = None
        for l in range(max_depth):
            layer = self.pheromone_graph.add_nodes_from([i for i, _ in enumerate(native_instructions)])

            if l == 0:
                self.pheromone_graph.add_edges_from([(self.start_node, node, 1.0) for node in layer])

            if previous_layer:
                for parent in previous_layer:
                    self.pheromone_graph.add_edges_from([(parent, child, 1.0) for child in layer])

            self.pheromone_graph.add_edges_from([(node, self.end_node, 1.0) for node in layer])
            previous_layer = layer

        self.best_v = QuantumCircuit(u.num_qubits)
        self.best_params: List[float] = []
        self.best_cost = 1.0

    @property
    def best_solution(self) -> Tuple[QuantumCircuit, List[float], float]:
        return self.best_v, self.best_params, self.best_cost

    def _run_iteration(self):
        solutions = self._generate_solutions()
        self._update_pheromones(solutions)

    def _generate_solutions(self) -> List[List[Tuple[int, int]]]:
        solutions = []

        for k in range(self.num_ants):
            solution = []
            current_node = self.start_node

            while current_node != self.end_node:
                edges = [e[:2] for e in self.pheromone_graph.out_edges(current_node)]
                edge_weights = [self._edge_probability(*e) for e in edges]

                edge = random.choices(edges, edge_weights)[0]

                solution.append(edge)
                current_node = edge[1]

            solutions.append(solution)

        return solutions

    def _update_pheromones(self, solutions: List[List[Tuple[int, int]]]):
        # Pheromone evaporation
        for idx, (_, _, pheromones) in self.pheromone_graph.edge_index_map().items():
            self.pheromone_graph.update_edge_by_index(idx, (1 - self.evaporation_rate) * pheromones)

        # Ants deposit pheromones
        for solution in solutions:
            node_data = [self.pheromone_graph.get_node_data(e[0]) for e in solution[1:]]
            instructions = [self.native_instructions[i] for i in node_data]

            v = self.u.copy_empty_like()

            for instruction, qubits in instructions:
                if instruction.is_parameterized():
                    instruction = instruction.copy()
                    instruction.params = [Parameter(f'p{random.randrange(1_000_000)}') for _ in instruction.params]
                v.append(instruction, qubits)

            params, cost = self.continuous_optimization(self.u, v)

            if cost < self.best_cost:
                self.best_v, self.best_params, self.best_cost = v, params, cost

            delta = (1.0 - cost) / len(solution)
            for i, j in solution:
                current = self.pheromone_graph.get_edge_data(i, j)
                self.pheromone_graph.update_edge(i, j, current + delta)

    def _edge_probability(self, i: int, j: int) -> float:
        edge_pheromones = self.pheromone_graph.get_edge_data(i, j)
        sum_pheromones = sum(edge[2] for edge in self.pheromone_graph.out_edges(i))
        return edge_pheromones / sum_pheromones


if __name__ == '__main__':
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import SXGate, RZGate, CXGate

    from vqc.optimization.continuous.gradient_based import gradient_based_hst_weighted

    u = QuantumCircuit(1)
    u.h(0)

    sx = SXGate()
    rz = RZGate(Parameter('x'))
    cx = CXGate()

    actions = [
        (sx, (0,)),
        (rz, (0,)),
    ]

    aco = AntColonyOptimization(u, actions, gradient_based_hst_weighted, 3, 10, 0.5)

    for _ in range(10):
        aco._run_iteration()

    print(aco.best_solution[0].draw())
    print(aco.best_solution[1:])




