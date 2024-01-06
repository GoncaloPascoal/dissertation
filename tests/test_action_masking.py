
import numpy as np
import pytest
from qiskit import QuantumCircuit

from narlsqr.env import RoutingEnv
from narlsqr.topology import linear_topology


@pytest.fixture()
def env_linear_5q() -> RoutingEnv:
    return RoutingEnv(linear_topology(5))


def test_bridge(env_linear_5q: RoutingEnv):
    env = env_linear_5q

    qc = QuantumCircuit(5)
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.cx(2, 4)

    env.circuit = qc

    obs, *_ = env.reset()
    np.testing.assert_array_equal(obs['action_mask'], np.array([1, 1, 1, 1, 1, 1, 0]))

    obs, *_ = env.step(4)
    np.testing.assert_array_equal(obs['action_mask'], np.array([1, 1, 1, 1, 0, 1, 1]))


def test_blocked_swap(env_linear_5q: RoutingEnv):
    env = env_linear_5q

    qc = QuantumCircuit(5)
    qc.cx(0, 4)

    env.circuit = qc

    env.reset()
    obs, *_ = env.step(0)

    assert env._blocked_swaps == {0}
    np.testing.assert_array_equal(obs['action_mask'], np.array([0, 1, 0, 1, 0, 0, 0]))

    obs, *_ = env.step(1)

    assert env._blocked_swaps == {1}
    np.testing.assert_array_equal(obs['action_mask'], np.array([0, 0, 1, 1, 0, 0, 1]))

    env.step(6)

    assert env._blocked_swaps == set()
