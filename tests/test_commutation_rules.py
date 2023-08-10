
import pytest
from qiskit import QuantumCircuit

from routing.env import RoutingEnv
from routing.topology import linear_topology


@pytest.fixture
def env_linear_4q() -> RoutingEnv:
    return RoutingEnv(linear_topology(4), commutation_analysis=True)


def assert_commutation(env: RoutingEnv, circuit: QuantumCircuit, expected: QuantumCircuit):
    env.circuit = circuit
    env.reset()
    routed_circuit = env.routed_circuit()
    assert routed_circuit == expected, f'\nActual: {routed_circuit}\n\nExpected: {expected}'


def test_basic(env_linear_4q: RoutingEnv):
    env = env_linear_4q

    # CNOTs with the same control qubit commute
    qc = QuantumCircuit(4)
    qc.cx(0, [2, 1])

    expected = QuantumCircuit(4)
    expected.cx(0, 1)

    assert_commutation(env, qc, expected)

    # CNOTs with the same target qubit commute
    qc = QuantumCircuit(4)
    qc.cx([0, 1], 2)

    expected = QuantumCircuit(4)
    expected.cx(1, 2)

    assert_commutation(env, qc, expected)

    # Z-rotations commute with CNOTs through the control qubit
    qc = QuantumCircuit(4)
    qc.cx(0, 2)
    qc.z(0)

    expected = QuantumCircuit(4)
    expected.z(0)

    assert_commutation(env, qc, expected)

    # X-rotations commute with CNOTs through the target qubit
    qc = QuantumCircuit(4)
    qc.cx(0, 2)
    qc.x(2)

    expected = QuantumCircuit(4)
    expected.x(2)

    assert_commutation(env, qc, expected)


def test_commutation_order(env_linear_4q: RoutingEnv):
    env = env_linear_4q

    # Gates that commute with an infeasible gate but do not commute amongst themselves should be scheduled
    # in the same order they appear in the DAG
    qc = QuantumCircuit(4)
    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.cx(0, 1)

    expected = QuantumCircuit(4)
    expected.cx(1, 2)
    expected.cx(0, 1)

    assert_commutation(env, qc, expected)


def test_1q_blocking(env_linear_4q: RoutingEnv):
    env = env_linear_4q

    # Non-commuting single-qubit gates can block other commuting gates
    qc = QuantumCircuit(4)
    qc.cx(0, 2)
    qc.z(2)  # Does not commute with CX(1, 2)
    qc.cx(1, 2)
    qc.cx(0, 1)

    expected = QuantumCircuit(4)

    assert_commutation(env, qc, expected)


def test_2q_blocking(env_linear_4q: RoutingEnv):
    env = env_linear_4q

    # Non-commuting two-qubit gates can block other commuting gates
    qc = QuantumCircuit(4)
    qc.cx(0, 2)
    qc.cx(3, 1)  # Does not commute with CX(1, 2)
    qc.cx(1, 2)
    qc.cx(0, 1)

    expected = QuantumCircuit(4)

    assert_commutation(env, qc, expected)


def test_trivially_commuting(env_linear_4q: RoutingEnv):
    env = env_linear_4q

    # Gates in the same layer that do not share qubits commute trivially
    qc = QuantumCircuit(4)
    qc.cx(0, 2)
    qc.cx(1, 3)  # Does not share qubits with CX(0, 2) and commutes with CX(1, 2)
    qc.cx(1, 2)
    qc.cx(0, 1)

    expected = QuantumCircuit(4)
    expected.cx(1, 2)

    assert_commutation(env, qc, expected)

