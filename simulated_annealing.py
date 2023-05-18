
import itertools
import logging
import random
from abc import ABC, abstractmethod

from enum import auto, Enum
from math import exp
from typing import Sequence, Tuple, Optional, TypeVar, Generic

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, CircuitInstruction, Instruction
from qiskit.circuit.library import CXGate, SXGate, RZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode

from rich.logging import RichHandler
from tqdm.rich import tqdm_rich

from parameter_generator import ParameterGenerator
from utils import ContinuousOptimizationFunction, ContinuousOptimizationResult

from rich import print

from gradient_based import gradient_based_hst_weighted
from utils import NativeInstruction


SolutionType = TypeVar('SolutionType')


class SimulatedAnnealing(ABC, Generic[SolutionType]):
    def __init__(
        self,
        max_iterations: int,
        *,
        seed: Optional[int] = None,
        log_level: int = logging.INFO,
        tolerance: float = 1e-2,
        use_acceptance_iterations: bool = False,
        initial_temperature: float = 0.2,
        beta: Optional[float] = 1.5,
    ):
        assert max_iterations > 0
        assert tolerance > 0.0
        assert initial_temperature > 0.0
        if beta is not None:
            assert beta > 0.0

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_acceptance_iterations = use_acceptance_iterations
        self.initial_temperature = initial_temperature
        self.beta = beta

        self.temperature = initial_temperature

        self.rng = random.Random(seed)

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self._logger.addHandler(RichHandler())

        self.best_solution = self._generate_initial_solution()
        self.best_cost = self._evaluate_solution(self.best_solution)
        self._logger.info(f'Initial solution has cost {self.best_cost:.4f}')

        self.current_solution = self._copy_solution(self.best_solution)
        self.current_cost = self.best_cost

    @abstractmethod
    def _generate_initial_solution(self) -> SolutionType:
        raise NotImplementedError

    @abstractmethod
    def _evaluate_solution(self, solution: SolutionType) -> float:
        raise NotImplementedError

    @abstractmethod
    def _copy_solution(self, solution: SolutionType) -> SolutionType:
        raise NotImplementedError

    @abstractmethod
    def _generate_neighbor(self) -> SolutionType:
        raise NotImplementedError

    def run(self) -> Tuple[SolutionType, float]:
        i = 1
        self.temperature = self.initial_temperature

        if self._logger.level <= logging.INFO:
            progress = tqdm_rich(initial=i, total=self.max_iterations)

        while i < self.max_iterations:
            if self._terminated():
                break

            # Generate and evaluate neighbor
            neighbor = self._generate_neighbor()
            cost = self._evaluate_solution(neighbor)

            # Update best solution
            if cost < self.best_cost:
                self.best_solution, self.best_cost = neighbor, cost
                self._logger.info(f'Cost decreased to {cost:.4f} in iteration {i}')

            # Determine probability of accepting neighboring solution
            cost_penalty = cost - self.current_cost
            accepted = False
            if cost_penalty <= 0 or self.rng.random() <= exp(-cost_penalty / self.temperature):
                self.current_solution, self.current_cost = self._copy_solution(neighbor), cost
                accepted = True

            if not self.use_acceptance_iterations or accepted:
                i += 1
                if self._logger.level <= logging.INFO:
                    progress.update(1)

            # Annealing schedule
            if self.beta is not None:
                self.temperature = self.initial_temperature * pow(1.0 - i / self.max_iterations, self.beta)

        return self.best_solution, self.best_cost

    def _terminated(self) -> bool:
        return self.best_cost <= self.tolerance


class NeighborhoodType(Enum):
    ADD_INSTRUCTION = auto()
    CHANGE_INSTRUCTION = auto()
    CHANGE_QUBITS = auto()
    REMOVE_INSTRUCTION = auto()
    SWAP_INSTRUCTIONS = auto()


class StructuralSimulatedAnnealing(SimulatedAnnealing[QuantumCircuit]):
    def __init__(
        self,
        max_iterations: int,
        u: QuantumCircuit,
        native_instructions: Sequence[NativeInstruction],
        continuous_optimization: ContinuousOptimizationFunction,
        max_instructions: int,
        min_instructions: int = 0,
        **kwargs,
    ):
        assert max_instructions >= min_instructions >= 0

        self.u = u
        self.native_instructions = native_instructions

        instructions = [instruction[0] for instruction in native_instructions]
        self.instruction_set = [
            instruction for i, instruction in enumerate(instructions)
            if instruction not in instructions[:i]
        ]

        self.continuous_optimization = lambda v: continuous_optimization(self.u, v)
        self.max_iterations = max_iterations
        self.max_instructions = max_instructions
        self.min_instructions = min_instructions

        self.param_gen = ParameterGenerator()

        super().__init__(max_iterations, **kwargs)

    def _is_valid_instruction(self, instruction: NativeInstruction) -> bool:
        return (instruction[0].name not in {'delay', 'id', 'reset'} and
                instruction[0].num_clbits == 0 and
                instruction[0].num_qubits <= self.u.num_qubits)

    def _generate_initial_solution(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.u.num_qubits)

        for _ in range(self.max_instructions):
            instruction, qubits = self.rng.choice(self.native_instructions)

            if instruction.is_parameterized():
                instruction = self.param_gen.parameterize(instruction)

            qc.append(instruction, qubits)

        return qc

    def _evaluate_solution(self, solution: QuantumCircuit) -> float:
        _, cost = self.continuous_optimization(solution)
        return cost

    def _copy_solution(self, solution: QuantumCircuit) -> QuantumCircuit:
        return solution.copy()

    def _resolve_random_identity(self, new_instruction: Instruction) -> QuantumCircuit:
        dag = circuit_to_dag(self.current_solution)

        options = []
        for layer in dag.layers():
            op_node = layer['graph'].op_nodes(include_directives=False)[0]
            active_qubits = [self.current_solution.find_bit(q).index for q in itertools.chain(*layer['partition'])]
            idle_qubits = set(range(self.current_solution.num_qubits)).difference(active_qubits)

            for instruction, qubits in self.native_instructions:
                if instruction == new_instruction and idle_qubits.issuperset(qubits):
                    options.append((op_node, qubits))

        if not options:
            raise ValueError('Circuit contains no resolvable identities.')

        op_node, qubits = self.rng.choice(options)

        idx = 0
        for i, n in enumerate(dag.op_nodes(include_directives=False)):
            if DAGOpNode.semantic_eq(n, op_node):
                idx = i
                break

        return self._add_instruction(idx, new_instruction, qubits)

    def _add_instruction(self, idx: int, new_instruction: Instruction, qubits: Sequence[int]) -> QuantumCircuit:
        qc = self.current_solution.copy_empty_like()

        if new_instruction.is_parameterized():
            new_instruction = self.param_gen.parameterize(new_instruction)

        for i, instruction in enumerate(self.current_solution.data):
            if i == idx:
                qc.append(new_instruction, qubits)
            qc.append(instruction)

        return qc

    def _change_instruction(self, idx: int, new_instruction: Instruction, qubits: Sequence[int]) -> QuantumCircuit:
        qc = self.current_solution.copy_empty_like()

        if new_instruction.is_parameterized():
            new_instruction = self.param_gen.parameterize(new_instruction)

        for i, instruction in enumerate(self.current_solution.data):
            if i == idx:
                qc.append(new_instruction, qubits)
            else:
                qc.append(instruction)

        return qc

    def _change_qubits(self, idx: int, qubits: Sequence[int]) -> QuantumCircuit:
        dag = circuit_to_dag(self.current_solution)
        dag.op_nodes()[idx].qargs = tuple(self.current_solution.qubits[i] for i in qubits)
        return dag_to_circuit(dag, copy_operations=False)

    def _remove_instruction(self, idx: int) -> QuantumCircuit:
        dag = circuit_to_dag(self.current_solution)
        dag.remove_op_node(dag.op_nodes()[idx])
        return dag_to_circuit(dag, copy_operations=False)

    def _swap_instructions(self, idx: int) -> QuantumCircuit:
        qc = self.current_solution.copy_empty_like()

        data = self.current_solution.data.copy()
        data[idx], data[idx + 1] = data[idx + 1], data[idx]

        for instruction in data:
            qc.append(instruction)

        return qc

    def _generate_neighbor(self) -> QuantumCircuit:
        types = set(NeighborhoodType)
        num_instructions = len(self.current_solution)

        # TODO: Remove
        types.remove(NeighborhoodType.ADD_INSTRUCTION)
        types.remove(NeighborhoodType.REMOVE_INSTRUCTION)

        if self.current_solution.num_qubits <= 1:
            types.discard(NeighborhoodType.CHANGE_QUBITS)

        if num_instructions == self.max_instructions:
            types.discard(NeighborhoodType.ADD_INSTRUCTION)

        if num_instructions == self.min_instructions:
            types.discard(NeighborhoodType.REMOVE_INSTRUCTION)

        if num_instructions == 0:
            types.discard(NeighborhoodType.CHANGE_QUBITS)
            types.discard(NeighborhoodType.CHANGE_INSTRUCTION)
            types.discard(NeighborhoodType.REMOVE_INSTRUCTION)

        if num_instructions < 2:
            types.discard(NeighborhoodType.SWAP_INSTRUCTIONS)

        n_type = self.rng.choice(list(types))
        match n_type:
            case NeighborhoodType.ADD_INSTRUCTION:
                new_instruction = self.rng.choice(self.instruction_set)
                try:
                    neighbor = self._resolve_random_identity(new_instruction)
                except ValueError:
                    qubits = self.rng.choice([
                        i[1] for i in self.native_instructions
                        if i[0] == new_instruction
                    ])

                    if new_instruction.is_parameterized():
                        new_instruction = self.param_gen.parameterize(new_instruction)

                    neighbor = self.current_solution.compose(new_instruction, qubits)
            case NeighborhoodType.CHANGE_INSTRUCTION:
                idx = self.rng.randrange(num_instructions)
                instruction: CircuitInstruction = self.current_solution.data[idx]
                qubits = tuple(self.current_solution.find_bit(q).index for q in instruction.qubits)
                qubit_set = set(qubits)

                new_instruction = self.rng.choice([
                    i for i in self.instruction_set
                    if i.name != instruction.operation.name
                ])

                if new_instruction.num_qubits < len(qubits):
                    # Apply single-qubit instruction to one of the qubits from the two-qubit instruction
                    qubits = self.rng.choice([
                        i[1] for i in self.native_instructions
                        if i[0].name == new_instruction.name and qubit_set.issuperset(i[1])
                    ])
                elif new_instruction.num_qubits > len(qubits):
                    # Apply two-qubit instruction to a pair containing the qubit from the single-qubit instruction
                    qubits = self.rng.choice([
                        i[1] for i in self.native_instructions
                        if i[0].name == new_instruction.name and qubit_set.issubset(i[1])
                    ])

                neighbor = self._change_instruction(idx, new_instruction, qubits)
            case NeighborhoodType.CHANGE_QUBITS:
                idx = self.rng.randrange(num_instructions)
                instruction: CircuitInstruction = self.current_solution.data[idx]

                old_qubits = tuple(self.current_solution.find_bit(q).index for q in instruction.qubits)
                filtered_instructions = [
                    i for i in self.native_instructions
                    if i[0].name == instruction.operation.name and i[1] != old_qubits
                ]
                _, new_qubits = self.rng.choice(filtered_instructions)

                neighbor = self._change_qubits(idx, new_qubits)
            case NeighborhoodType.REMOVE_INSTRUCTION:
                idx = self.rng.randrange(num_instructions)
                neighbor = self._remove_instruction(idx)
            case NeighborhoodType.SWAP_INSTRUCTIONS:
                idx = self.rng.randrange(num_instructions - 1)
                neighbor = self._swap_instructions(idx)

        return neighbor


def main():
    u = QuantumCircuit(2)
    u.ch(0, 1)

    sx = SXGate()
    rz = RZGate(Parameter('x'))
    cx = CXGate()

    native_instructions = [
        NativeInstruction(sx, (0,)),
        NativeInstruction(sx, (1,)),
        NativeInstruction(rz, (0,)),
        NativeInstruction(rz, (1,)),
        NativeInstruction(cx, (0, 1)),
        NativeInstruction(cx, (1, 0)),
    ]

    def continuous_optimization(
        u: QuantumCircuit,
        v: QuantumCircuit,
    ) -> ContinuousOptimizationResult:
        return gradient_based_hst_weighted(u, v)

    algo = StructuralSimulatedAnnealing(100, u, native_instructions, continuous_optimization, 6)

    circuit, cost = algo.run()

    print(circuit.draw())
    print(f'The best circuit had a cost of {cost}.')


if __name__ == '__main__':
    main()
