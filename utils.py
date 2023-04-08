
import collections
from math import pi
from typing import Callable, Dict, List, Sequence, Set, Tuple, Type

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.converters import circuit_to_dag, dag_to_circuit


# Use collections.namedtuple instead of typing.NamedTuple to avoid exception cased by cloudpickle (Ray)
NativeInstruction: Type[Tuple[Instruction, Tuple[int, ...]]] = collections.namedtuple(
    'NativeInstruction',
    ['instruction', 'qubits']
)

NativeInstructionDict = Dict[str, Tuple[Instruction, Set[Tuple[int, ...]]]]

ContinuousOptimizationResult: Type[Tuple[List[float], float]] = collections.namedtuple(
    'ContinuousOptimizationResult', ['parameters', 'cost']
)

ContinuousOptimizationFunction = Callable[
    [QuantumCircuit, QuantumCircuit],
    ContinuousOptimizationResult,
]


def create_native_instruction_dict(native_instructions: Sequence[NativeInstruction]) -> NativeInstructionDict:
    d = {}

    for instruction, qubits in native_instructions:
        _, possible_qubits = d.setdefault(instruction.name, (instruction, set()))
        possible_qubits.add(qubits)

    return d

def counts_to_ratios(counts: Dict[str, int]) -> Dict[str, float]:
    """Converts a dictionary of counts to one of ratios (or proportions) between 0 and 1."""
    s = sum(counts.values())
    return {k: v / s for k, v in counts.items()}

def remove_instruction(qc: QuantumCircuit, idx: int) -> QuantumCircuit:
    """
    Remove the instruction at index ``idx`` from the circuit, returning a
    new circuit.
    """
    dag = circuit_to_dag(qc)
    dag.remove_op_node(dag.op_nodes(include_directives=False)[idx])
    return dag_to_circuit(dag)

def remove_parametrized_instruction(qc: QuantumCircuit, idx: int) -> QuantumCircuit:
    """
    Remove the parametrized instruction at index ``idx`` from the circuit, returning a
    new circuit.
    """
    dag = circuit_to_dag(qc)
    parametrized_op_nodes = list(filter(
        lambda n: n.op.params,
        dag.op_nodes(include_directives=False)
    ))
    dag.remove_op_node(parametrized_op_nodes[idx])
    return dag_to_circuit(dag)

def normalize_angles(angles: Sequence[float]) -> List[float]:
    return [theta % (2 * pi) for theta in angles]

