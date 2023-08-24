
import numpy as np
import pytest
from qiskit import QuantumCircuit

from narlsqr.env import RoutingEnv
from narlsqr.topology import linear_topology


@pytest.fixture
def env_linear_4q() -> RoutingEnv:
    return RoutingEnv(linear_topology(4))


def test_blocked_swap(env_linear_4q: RoutingEnv):
    env = env_linear_4q

    qc = QuantumCircuit(4)
    qc.cx(0, 3)

    env.circuit = qc

    env.reset()
    obs, *_ = env.step(0)

    assert env._blocked_swap == 0
    np.testing.assert_array_equal(obs['action_mask'], np.array([0, 1, 1, 0, 1]))
