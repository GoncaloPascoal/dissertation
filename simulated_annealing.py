
import itertools
import random

from math import exp, pi

from enum import auto, Enum
from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, CircuitInstruction, Instruction
from qiskit.circuit.library import CXGate, SXGate, RZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode

from rich import print

from rl import CircuitAction
from gradient_based import gradient_based_hst_weighted

class NeighborhoodType(Enum):
    ADD_INSTRUCTION = auto()
    CHANGE_INSTRUCTION = auto()
    CHANGE_QUBITS = auto()
    REMOVE_INSTRUCTION = auto()
    SWAP_INSTRUCTIONS = auto()

def _resolve_random_identity(qc: QuantumCircuit, native_instructions: Sequence[CircuitAction]) -> QuantumCircuit:
    dag = circuit_to_dag(qc)

    options = []
    for layer in dag.layers():
        op_node = layer['graph'].op_nodes(include_directives=False)[0]
        active_qubits = [qc.find_bit(q).index for q in itertools.chain(*layer['partition'])]
        idle_qubits = set(range(qc.num_qubits)).difference(active_qubits)

        for instruction, qubits in native_instructions:
            if idle_qubits.issuperset(qubits):
                options.append((op_node, instruction, qubits))

    if not options:
        raise ValueError('Circuit contains no resolvable identities.')

    op_node, new_instruction, qubits = random.choice(options)

    idx = 0
    for i, n in enumerate(dag.op_nodes(include_directives=False)):
        if DAGOpNode.semantic_eq(n, op_node):
            idx = i
            break

    return _add_instruction(dag_to_circuit(dag), idx, new_instruction, qubits)

def _change_instruction(qc: QuantumCircuit, idx: int, instruction: Instruction) -> QuantumCircuit:
    dag = circuit_to_dag(qc)
    dag.substitute_node(dag.op_nodes()[idx], instruction, inplace=True)
    return dag_to_circuit(dag)

def _change_qubits(qc: QuantumCircuit, idx: int, qubits: Sequence[int]) -> QuantumCircuit:
    dag = circuit_to_dag(qc)
    dag.op_nodes()[idx].qargs = tuple(qc.qubits[i] for i in qubits)
    return dag_to_circuit(dag)

def _remove_instruction(qc: QuantumCircuit, idx: int) -> QuantumCircuit:
    dag = circuit_to_dag(qc)
    dag.remove_op_node(dag.op_nodes()[idx])
    return dag_to_circuit(dag)

def _add_instruction(qc: QuantumCircuit, idx: int, new_instruction: Instruction, qubits: Sequence[int]) -> QuantumCircuit:
    new_qc = qc.copy_empty_like()

    if new_instruction.is_parameterized():
        new_instruction = new_instruction.copy()
        new_instruction.params = [Parameter(f'p{random.randrange(1_000_000)}') for _ in new_instruction.params]

    for i, instruction in enumerate(qc.data):
        if i == idx:
            new_qc.append(new_instruction, qubits)
        new_qc.append(instruction)

    return new_qc

def _swap_instructions(qc: QuantumCircuit, idx_a: int, idx_b: int) -> QuantumCircuit:
    dag = circuit_to_dag(qc)
    op_nodes = dag.op_nodes()

    node_a, node_b = op_nodes[idx_a], op_nodes[idx_b]
    temp_a = node_a.op
    dag.substitute_node(node_a, node_b.op, inplace=True)
    dag.substitute_node(node_b, temp_a, inplace=True)

    return dag_to_circuit(dag)

class SimulatedAnnealing:
    ContinuousOptimizationFunction = Callable[
        [QuantumCircuit, QuantumCircuit],
        Tuple[List[float], float]
    ]

    def _is_valid_instruction(self, instruction: CircuitAction) -> bool:
        return (instruction[0].name not in {'delay', 'id', 'reset'} and
            instruction[0].num_clbits == 0 and
            instruction[0].num_qubits <= self.u.num_qubits)

    def _generate_random_circuit(self, num_qubits: int, num_instructions: int) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)

        for _ in range(num_instructions):
            instruction, qubits = random.choice(self.native_instructions)

            # TODO: Refactor parameter renaming
            if instruction.is_parameterized():
                instruction = instruction.copy()
                instruction.params = [Parameter(f'p{random.randrange(1_000_000)}') for _ in instruction.params]

            qc.append(instruction, qubits)

        return qc

    # TODO: Replace CircuitAction in signature
    def __init__(
        self,
        u: QuantumCircuit,
        native_instructions: Sequence[CircuitAction],
        continuous_optimization: ContinuousOptimizationFunction,
        max_iterations: int,
        initial_temperature: float = 0.2,
        beta: float = 1.5,
    ):
        assert max_iterations > 0

        self.u = u
        self.native_instructions = native_instructions
        self.continuous_optimization = lambda v: continuous_optimization(self.u, v)
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.beta = beta

        self.v = self._generate_random_circuit(u.num_qubits, 3)
        self.best_v = self.v
        self.best_params, self.best_cost = self.continuous_optimization(self.v)
        self.current_cost = self.best_cost
        self.temperature = initial_temperature

    def _generate_neighbor(self) -> QuantumCircuit:
        types = set(NeighborhoodType)
        num_instructions = len(self.v)

        # TODO: Remove
        types.remove(NeighborhoodType.CHANGE_INSTRUCTION)
        types.remove(NeighborhoodType.SWAP_INSTRUCTIONS)

        if self.v.num_qubits <= 1:
            types.discard(NeighborhoodType.CHANGE_QUBITS)

        if num_instructions == 1:
            types.discard(NeighborhoodType.REMOVE_INSTRUCTION)
            types.discard(NeighborhoodType.SWAP_INSTRUCTIONS)

        n_type = random.choice(list(types))
        match n_type:
            case NeighborhoodType.ADD_INSTRUCTION:
                try:
                    neighbor = _resolve_random_identity(self.v, self.native_instructions)
                except ValueError:
                    new_instruction, qubits = random.choice(self.native_instructions)
                    neighbor = self.v.compose(new_instruction, qubits)
            case NeighborhoodType.CHANGE_INSTRUCTION:
                pass
            case NeighborhoodType.CHANGE_QUBITS:
                idx = random.randrange(num_instructions)
                instruction: CircuitInstruction = self.v.data[idx]

                old_qubits = tuple(self.v.find_bit(q).index for q in instruction.qubits)
                filtered_instructions = [
                    i for i in self.native_instructions
                    if i[0].name == instruction.operation.name and i[1] != old_qubits
                ]
                _, new_qubits = random.choice(filtered_instructions)

                neighbor = _change_qubits(self.v, idx, new_qubits)
            case NeighborhoodType.REMOVE_INSTRUCTION:
                idx = random.randrange(num_instructions)
                neighbor = _remove_instruction(self.v, idx)
            case NeighborhoodType.SWAP_INSTRUCTIONS:
                idx_a, idx_b = random.sample(range(num_instructions), 2)
                neighbor = _swap_instructions(self.v, idx_a, idx_b)

        return neighbor

    def run(self) -> Tuple[QuantumCircuit, Sequence[float], float]:
        for i in range(1, self.max_iterations + 1):
            # Generate neighbor and optimize continuous parameters
            neighbor = self._generate_neighbor()
            params, cost = self.continuous_optimization(neighbor)

            # Update best solution
            if cost <= self.best_cost:
                self.best_v, self.best_params, self.best_cost = neighbor, params, cost

            # Determine probability of accepting neighboring solution
            cost_penalty = cost - self.current_cost
            if cost_penalty <= 0 or random.random() <= exp(-cost_penalty / self.temperature):
                self.v, self.current_cost = neighbor, cost

            # Annealing schedule
            self.temperature = self.initial_temperature * pow(1.0 - i / self.max_iterations, self.beta)

        return self.best_v, self.best_params, self.best_cost

if __name__ == '__main__':
    u = QuantumCircuit(1)
    u.t(0)

    sx = SXGate()
    rz = RZGate(Parameter('a'))

    native_instructions = [
        (sx, (0,)),
        (rz, (0,)),
    ]

    def continuous_optimization(
        u: QuantumCircuit,
        v: QuantumCircuit,
    ) -> Tuple[List[float], float]:
        return gradient_based_hst_weighted(u, v)

    algo = SimulatedAnnealing(u, native_instructions, continuous_optimization, 50)

    circuit, params, cost = algo.run()
    print(circuit.draw())

    params_pi = [f'{p / pi:4f}π' for p in params]
    print(f'The best parameters were {params_pi} with a cost of {cost}.')
