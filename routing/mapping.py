
from copy import deepcopy
from typing import Optional

import torch
from evotorch import Problem, SolutionBatch
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from evotorch.operators import CrossOver, CopyingOperator
from evotorch.tools import ObjectArray
from sb3_contrib import MaskablePPO

from routing.env import RoutingEnv


class OrderedCrossOver(CrossOver):
    def __init__(
        self,
        problem: Problem,
        *,
        tournament_size: int,
        obj_index: Optional[int] = None,
        num_children: Optional[int] = None,
        cross_over_rate: Optional[float] = None,
    ):
        super().__init__(
            problem,
            tournament_size=tournament_size,
            obj_index=obj_index,
            num_children=num_children,
            cross_over_rate=cross_over_rate
        )

    @staticmethod
    def _combine(
        crossover_mask: torch.Tensor,
        parents1: torch.Tensor | ObjectArray,
        parents2: torch.Tensor | ObjectArray
    ) -> torch.Tensor:
        children = torch.where(crossover_mask, parents1, -1)

        for i, sol in enumerate(children):
            parent_sol = parents2[i]
            sol[sol == -1] = parent_sol[torch.isin(parent_sol, sol, invert=True)]

        return children

    @torch.no_grad()
    def _do_cross_over(
        self,
        parents1: torch.Tensor | ObjectArray,
        parents2: torch.Tensor | ObjectArray,
    ) -> SolutionBatch:
        num_pairings = parents1.shape[0]

        device = parents1[0].device
        solution_length = self.problem.solution_length
        num_points = 2

        # For each pairing, generate all gene indices (i.e. [0, 1, 2, ...] for each pairing).
        gene_indices = (
            torch.arange(0, solution_length, device=device)
                 .unsqueeze(0)
                 .expand(num_pairings, solution_length)
        )

        # For each pairing, generate gene indices at which the parent solutions will be cut and recombined.
        crossover_points = self.problem.make_randint(
            torch.Size((num_pairings, num_points)), n=solution_length + 1, device=device
        )

        # From `crossover_points`, extract each cutting point for each solution.
        cutting_points = [crossover_points[:, i].reshape(-1, 1) for i in range(num_points)]

        # Initialize `crossover_mask` as a tensor filled with False.
        crossover_mask = torch.zeros((num_pairings, solution_length), dtype=torch.bool, device=device)

        # For each cutting point p, toggle the boolean values of `crossover_mask`
        # for indices bigger than the index pointed to by p
        for p in cutting_points:
            crossover_mask ^= gene_indices >= p

        # Using the mask, generate two children.
        children1 = OrderedCrossOver._combine(crossover_mask, parents1, parents2)
        children2 = OrderedCrossOver._combine(crossover_mask, parents2, parents1)

        # Combine children tensors in one tensor.
        children = torch.cat([children1, children2], dim=0)

        # Write the children solutions into a new SolutionBatch, and return the new batch.
        result = self._make_children_batch(children)
        return result


class InitialMappingProblem(Problem):
    def __init__(self, model: MaskablePPO, env: RoutingEnv):
        self.env = env

        env.reset()
        env.training = False

        def objective_func(mapping: torch.Tensor) -> torch.Tensor:
            env.initial_mapping = mapping.numpy()
            obs, _ = env.reset()

            terminated = False
            total_reward = 0.0

            while not terminated:
                action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
                action = int(action)

                obs, reward, terminated, *_ = env.step(action)
                total_reward += reward

            return torch.tensor([total_reward])

        super().__init__(
            'max',
            objective_func,
            bounds=(0, env.num_qubits - 1),
            solution_length=env.num_qubits,
            dtype=torch.int32,
        )

    def _fill(self, values: torch.Tensor) -> torch.Tensor:
        for sol in values:
            torch.randperm(self.solution_length, out=sol)
        return values


class SwapMutation(CopyingOperator):
    problem: InitialMappingProblem

    def __init__(self, problem: InitialMappingProblem, *, mutation_probability: Optional[float] = None):
        super().__init__(problem)
        self._mutation_probability = 1.0 if mutation_probability is None else float(mutation_probability)
        self._coupling_map = self.problem.env.coupling_map

    def _swap(self, data: torch.Tensor):
        edges = torch.tensor([
            self._coupling_map.edge_list()[i]
            for i in self.problem.make_randint(data.shape[0], n=self._coupling_map.num_edges())
        ])

        for sol, edge in zip(data, edges):
            sol[edge] = sol[reversed(edge)]

        return data

    @torch.no_grad()
    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = deepcopy(batch)

        data = result.access_values()
        to_mutate = self.problem.make_uniform(data.shape[0]) <= self._mutation_probability
        data[to_mutate] = self._swap(data[to_mutate])

        return result


def initial_mapping_and_routing(
    model: MaskablePPO,
    env: RoutingEnv,
    *,
    population_size: int = 40,
    cross_over_rate: float = 1.0,
    mutation_probability: float = 0.1,
) -> GeneticAlgorithm:
    problem = InitialMappingProblem(model, env)

    operators = [
        OrderedCrossOver(problem, tournament_size=4, cross_over_rate=cross_over_rate),
        SwapMutation(problem, mutation_probability=mutation_probability),
    ]

    ga = GeneticAlgorithm(problem, operators=operators, popsize=population_size)
    _ = StdOutLogger(ga)

    return ga


def main():
    from rich import print
    import rustworkx as rx
    from qiskit import transpile
    from qiskit.converters import dag_to_circuit
    from qiskit.transpiler import CouplingMap

    from routing.env import QcpRoutingEnv
    from routing.circuit_gen import RandomCircuitGenerator
    from routing.env import NoiseConfig

    g = rx.PyGraph()
    g.add_nodes_from([0, 1, 2, 3, 4])
    g.add_edges_from_no_data([(0, 1), (1, 2), (1, 3), (3, 4)])
    noise_config = NoiseConfig(1e-2, 3e-3, log_base=2)

    env = QcpRoutingEnv(g, RandomCircuitGenerator(g.num_nodes(), 16), 8,
                        training_iterations=4, noise_config=noise_config, termination_reward=0.0)
    model = MaskablePPO.load('models/m_qcp_routing.model', env, tensorboard_log='logs/routing')

    ga = initial_mapping_and_routing(model, env)
    ga.run(10)

    env.initial_mapping = ga.status['best'].values.numpy()
    obs, _ = env.reset()

    terminated = False
    total_reward = 0.0

    while not terminated:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
        action = int(action)

        obs, reward, terminated, *_ = env.step(action)
        total_reward += reward

    layout_method = 'sabre'
    routing_method = 'sabre'

    print(f'Total reward: {total_reward:.3f}\n')
    routed_circuit = dag_to_circuit(env.routed_dag)

    print('[b blue]RL Routing[/b blue]')
    print(f'Layout: {env.initial_mapping.tolist()}')
    print(f'Swaps: {routed_circuit.count_ops().get("swap", 0)}')
    if env.allow_bridge_gate:
        print(f'Bridges: {routed_circuit.count_ops().get("bridge", 0)}')

    routed_circuit = routed_circuit.decompose()
    print(f'CNOTs after decomposition: {routed_circuit.count_ops()["cx"]}')
    print(f'Depth after decomposition: {routed_circuit.depth()}\n')

    coupling_map = CouplingMap(g.to_directed().edge_list())
    t_qc = transpile(env.circuit, coupling_map=coupling_map, layout_method=layout_method,
                     routing_method=routing_method, basis_gates=['u', 'swap', 'cx'], optimization_level=0)

    print(f'[b blue]Qiskit Compiler ({routing_method} routing)[/b blue]')
    print(f'Layout: {list(t_qc.layout.initial_layout.get_virtual_bits().values())}')
    print(f'Swaps: {t_qc.count_ops().get("swap", 0)}')

    t_qc = t_qc.decompose()
    print(f'CNOTs after decomposition: {t_qc.count_ops()["cx"]}')
    print(f'Depth after decomposition: {t_qc.depth()}')


if __name__ == '__main__':
    main()
