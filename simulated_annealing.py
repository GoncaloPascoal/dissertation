
import itertools
import logging
import random

from enum import auto, Enum
from math import exp, pi
from typing import Sequence, Tuple, Optional

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


class NeighborhoodType(Enum):
    ADD_INSTRUCTION = auto()
    CHANGE_INSTRUCTION = auto()
    CHANGE_QUBITS = auto()
    REMOVE_INSTRUCTION = auto()
    SWAP_INSTRUCTIONS = auto()


class SimulatedAnnealing:
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    _logger.addHandler(RichHandler())

    def __init__(
        self,
        u: QuantumCircuit,
        native_instructions: Sequence[NativeInstruction],
        continuous_optimization: ContinuousOptimizationFunction,
        max_iterations: int,
        max_instructions: int,
        min_instructions: int = 0,
        tolerance: float = 1e-2,
        initial_temperature: float = 0.2,
        beta: Optional[float] = 1.5,
    ):
        assert max_iterations > 0
        assert max_instructions >= min_instructions >= 0
        assert tolerance > 0.0
        assert initial_temperature > 0.0
        if beta is not None:
            assert beta > 0.0

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
        self.tolerance = tolerance
        self.initial_temperature = initial_temperature
        self.beta = beta

        self.param_gen = ParameterGenerator()

        self.v = self._generate_random_circuit(u.num_qubits, max_instructions)
        self.best_v = self.v
        self.best_params, self.best_cost = self.continuous_optimization(self.v)
        self.current_cost = self.best_cost
        self.temperature = initial_temperature

        self._logger.info(f'Initial circuit has cost {self.best_cost:.4f}')

    def _is_valid_instruction(self, instruction: NativeInstruction) -> bool:
        return (instruction[0].name not in {'delay', 'id', 'reset'} and
                instruction[0].num_clbits == 0 and
                instruction[0].num_qubits <= self.u.num_qubits)

    def _generate_random_circuit(self, num_qubits: int, num_instructions: int) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)

        for _ in range(num_instructions):
            instruction, qubits = random.choice(self.native_instructions)

            if instruction.is_parameterized():
                instruction = self.param_gen.parameterize(instruction)

            qc.append(instruction, qubits)

        return qc

    def _resolve_random_identity(self, new_instruction: Instruction) -> QuantumCircuit:
        dag = circuit_to_dag(self.v)

        options = []
        for layer in dag.layers():
            op_node = layer['graph'].op_nodes(include_directives=False)[0]
            active_qubits = [self.v.find_bit(q).index for q in itertools.chain(*layer['partition'])]
            idle_qubits = set(range(self.v.num_qubits)).difference(active_qubits)

            for instruction, qubits in self.native_instructions:
                if instruction == new_instruction and idle_qubits.issuperset(qubits):
                    options.append((op_node, qubits))

        if not options:
            raise ValueError('Circuit contains no resolvable identities.')

        op_node, qubits = random.choice(options)

        idx = 0
        for i, n in enumerate(dag.op_nodes(include_directives=False)):
            if DAGOpNode.semantic_eq(n, op_node):
                idx = i
                break

        return self._add_instruction(idx, new_instruction, qubits)

    def _add_instruction(self, idx: int, new_instruction: Instruction, qubits: Sequence[int]) -> QuantumCircuit:
        qc = self.v.copy_empty_like()

        if new_instruction.is_parameterized():
            new_instruction = self.param_gen.parameterize(new_instruction)

        for i, instruction in enumerate(self.v.data):
            if i == idx:
                qc.append(new_instruction, qubits)
            qc.append(instruction)

        return qc

    def _change_instruction(self, idx: int, new_instruction: Instruction, qubits: Sequence[int]) -> QuantumCircuit:
        qc = self.v.copy_empty_like()

        if new_instruction.is_parameterized():
            new_instruction = self.param_gen.parameterize(new_instruction)

        for i, instruction in enumerate(self.v.data):
            if i == idx:
                qc.append(new_instruction, qubits)
            else:
                qc.append(instruction)

        return qc

    def _change_qubits(self, idx: int, qubits: Sequence[int]) -> QuantumCircuit:
        dag = circuit_to_dag(self.v)
        dag.op_nodes()[idx].qargs = tuple(self.v.qubits[i] for i in qubits)
        return dag_to_circuit(dag)

    def _remove_instruction(self, idx: int) -> QuantumCircuit:
        dag = circuit_to_dag(self.v)
        dag.remove_op_node(dag.op_nodes()[idx])
        return dag_to_circuit(dag)

    def _swap_instructions(self, idx: int) -> QuantumCircuit:
        qc = self.v.copy_empty_like()

        data = self.v.data.copy()
        data[idx], data[idx + 1] = data[idx + 1], data[idx]

        for instruction in data:
            qc.append(instruction)

        return qc

    def _generate_neighbor(self) -> QuantumCircuit:
        types = set(NeighborhoodType)
        num_instructions = len(self.v)

        # TODO: Remove
        types.remove(NeighborhoodType.SWAP_INSTRUCTIONS)
        types.remove(NeighborhoodType.CHANGE_QUBITS)

        if self.v.num_qubits <= 1:
            types.discard(NeighborhoodType.CHANGE_QUBITS)

        if num_instructions == self.max_instructions:
            types.discard(NeighborhoodType.ADD_INSTRUCTION)

        if num_instructions == self.min_instructions:
            types.discard(NeighborhoodType.REMOVE_INSTRUCTION)

        if num_instructions == 0:
            types.discard(NeighborhoodType.CHANGE_INSTRUCTION)
            types.discard(NeighborhoodType.REMOVE_INSTRUCTION)

        if num_instructions < 2:
            types.discard(NeighborhoodType.SWAP_INSTRUCTIONS)

        n_type = random.choice(list(types))
        match n_type:
            case NeighborhoodType.ADD_INSTRUCTION:
                new_instruction = random.choice(self.instruction_set)
                try:
                    neighbor = self._resolve_random_identity(new_instruction)
                except ValueError:
                    qubits = random.choice([
                        i[1] for i in self.native_instructions
                        if i[0] == new_instruction
                    ])

                    if new_instruction.is_parameterized():
                        new_instruction = self.param_gen.parameterize(new_instruction)

                    neighbor = self.v.compose(new_instruction, qubits)
            case NeighborhoodType.CHANGE_INSTRUCTION:
                idx = random.randrange(num_instructions)
                instruction: CircuitInstruction = self.v.data[idx]
                qubits = tuple(self.v.find_bit(q).index for q in instruction.qubits)
                qubit_set = set(qubits)

                new_instruction = random.choice([
                    i for i in self.instruction_set
                    if i.name != instruction.operation.name
                ])

                if new_instruction.num_qubits < len(qubits):
                    # Apply single-qubit instruction to one of the qubits from the two-qubit instruction
                    qubits = random.choice([
                        i[1] for i in self.native_instructions
                        if i[0].name == new_instruction.name and qubit_set.issuperset(i[1])
                    ])
                elif new_instruction.num_qubits > len(qubits):
                    # Apply two-qubit instruction to a pair containing the qubit from the single-qubit instruction
                    qubits = random.choice([
                        i[1] for i in self.native_instructions
                        if i[0].name == new_instruction.name and qubit_set.issubset(i[1])
                    ])

                neighbor = self._change_instruction(idx, new_instruction, qubits)
            case NeighborhoodType.CHANGE_QUBITS:
                idx = random.randrange(num_instructions)
                instruction: CircuitInstruction = self.v.data[idx]

                old_qubits = tuple(self.v.find_bit(q).index for q in instruction.qubits)
                filtered_instructions = [
                    i for i in self.native_instructions
                    if i[0].name == instruction.operation.name and i[1] != old_qubits
                ]
                _, new_qubits = random.choice(filtered_instructions)

                neighbor = self._change_qubits(idx, new_qubits)
            case NeighborhoodType.REMOVE_INSTRUCTION:
                idx = random.randrange(num_instructions)
                neighbor = self._remove_instruction(idx)
            case NeighborhoodType.SWAP_INSTRUCTIONS:
                idx = random.randrange(num_instructions - 1)
                neighbor = self._swap_instructions(idx)

        return neighbor

    def run(self) -> Tuple[QuantumCircuit, Sequence[float], float]:
        i = 1
        progress = tqdm_rich(initial=i, total=self.max_iterations)
        while i < self.max_iterations:
            if self.best_cost < self.tolerance:
                break

            # Generate neighbor and optimize continuous parameters
            neighbor = self._generate_neighbor()
            params, cost = self.continuous_optimization(neighbor)

            # Update best solution
            if cost < self.best_cost:
                self.best_v, self.best_params, self.best_cost = neighbor, params, cost
                self._logger.info(f'Cost decreased to {cost:.4f} in iteration {i}')

            # Determine probability of accepting neighboring solution
            cost_penalty = cost - self.current_cost
            if cost_penalty <= 0 or random.random() <= exp(-cost_penalty / self.temperature):
                self.v, self.current_cost = neighbor, cost
                i += 1
                progress.update(1)

            # Annealing schedule
            if self.beta is not None:
                self.temperature = self.initial_temperature * pow(1.0 - i / self.max_iterations, self.beta)

        return self.best_v, self.best_params, self.best_cost


def main():
    from qiskit.circuit.library import QFT

    u = QuantumCircuit(2)
    u.swap(0, 1)

    sx = SXGate()
    rz = RZGate(Parameter('x'))
    cx = CXGate()

    native_instructions = [
        (sx, (0,)),
        (sx, (1,)),
        (rz, (0,)),
        (rz, (1,)),
        (cx, (0, 1)),
        (cx, (1, 0)),
    ]

    def continuous_optimization(
        u: QuantumCircuit,
        v: QuantumCircuit,
    ) -> ContinuousOptimizationResult:
        return gradient_based_hst_weighted(u, v)

    algo = SimulatedAnnealing(u, native_instructions, continuous_optimization, 100, 3)

    circuit, params, cost = algo.run()
    print(circuit.draw())

    params_pi = [f'{p / pi:.4f}π' for p in params]
    print(f'The best parameters were {params_pi} with a cost of {cost}.')


if __name__ == '__main__':
    main()
