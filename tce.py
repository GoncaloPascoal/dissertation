
from abc import ABC, abstractmethod
from math import pi
from typing import Any, Dict, List, Literal, NamedTuple, Optional, SupportsFloat, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from nptyping import Int8, NDArray
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.exceptions import CircuitError
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

from gate_class import GateClass
from utils import index_with_key


class TransformationRule(ABC):
    @abstractmethod
    def is_valid(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def apply(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> QuantumCircuit:
        raise NotImplementedError


class TransformationCircuitEnv(gym.Env, ABC):
    ActType = int
    ObsType = gym.core.ObsType

    class DecodedAction(NamedTuple):
        layer: int
        qubit: int
        rule: int

    def __init__(
        self,
        max_depth: int,
        num_qubits: int,
        gate_classes: List[GateClass],
        transformation_rules: List[TransformationRule],
        *,
        max_time_steps: int = 32,
        target_circuit: Optional[QuantumCircuit] = None,
    ):
        self.max_depth = max_depth
        self.num_qubits = num_qubits
        self.gate_classes = gate_classes
        self.transformation_rules = transformation_rules

        self.action_space: spaces.Discrete = spaces.Discrete(max_depth * num_qubits * len(transformation_rules))

        self.target_circuit = target_circuit
        self.max_time_steps = max_time_steps

        self.current_circuit = QuantumCircuit(num_qubits)
        self.next_circuit = QuantumCircuit(num_qubits)

        self.basis_gates = list({gc.gate.name for gc in self.gate_classes})
        self.rng = np.random.default_rng()
        self.time_step = 0
        self._set_valid_actions()

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        decoded = self.decode_action(action)
        if not self.valid_actions[action]:
            layer, qubit, rule = decoded
            rule_type = self.transformation_rules[rule]
            raise ValueError(f'Invalid action selected: {rule_type.__class__.__name__} on layer {layer},'
                             f'qubit {qubit}')

        self.next_circuit = transpile(
            self.generate_next_circuit(decoded),
            approximation_degree=0.0,
            basis_gates=self.basis_gates,
        )

        reward = self.reward()

        self.current_circuit = self.next_circuit
        self._set_valid_actions()

        self.time_step += 1
        terminated = self.time_step == self.max_time_steps or not self.valid_actions.any()

        return self.observation(), reward, terminated, False, {}

    def reset(self, *, seed=None, options=None) -> Tuple[ObsType, Dict[str, Any]]:
        self.reset_circuits()
        self.time_step = 0

        return self.observation(), {}

    @abstractmethod
    def reward(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def observation(self) -> ObsType:
        raise NotImplementedError

    @abstractmethod
    def generate_next_circuit(self, decoded_action: DecodedAction) -> QuantumCircuit:
        raise NotImplementedError

    @abstractmethod
    def reset_circuits(self):
        raise NotImplementedError

    def action_masks(self) -> NDArray[Literal['*'], Int8]:
        return self.valid_actions

    def decode_action(self, action: ActType) -> DecodedAction:
        return TransformationCircuitEnv.DecodedAction(*np.unravel_index(
            action, shape=(self.max_depth, self.num_qubits, len(self.transformation_rules))
        ))

    def format_action(self, action: ActType | DecodedAction) -> str:
        if isinstance(action, TransformationCircuitEnv.ActType):
            action = self.decode_action(action)
        rule_type = self.transformation_rules[action.rule]

        return f'{rule_type.__class__.__name__} (layer {action.layer}, qubit {action.qubit})'

    def circuit_to_obs(self, qc: QuantumCircuit) -> NDArray[Literal['*, *, *'], Int8]:
        """
        Converts the given ``QuantumCircuit`` to an equivalent observation based on the environment configuration.
        The observation will be a binary NumPy array of shape (``max_depth``, ``num_qubits``, ``num_gate_classes``).

        :raises ValueError: if the circuit's depth or qubit count exceed the maximum values set for the environment.
        """

        if qc.depth() > self.max_depth:
            print(qc)
            raise ValueError(f'Circuit depth must not exceed {self.max_depth}')

        if qc.num_qubits > self.num_qubits:
            raise ValueError(f'Circuit qubit count must not exceed {self.num_qubits}')

        circuit_obs_shape = (self.max_depth, self.num_qubits, len(self.gate_classes))

        obs = np.zeros(shape=circuit_obs_shape, dtype=Int8)
        dag = circuit_to_dag(qc)

        for layer_idx, layer in enumerate(dag.layers()):
            for op_node in layer['graph'].gate_nodes():
                qubit, gate_idx = self.indices_from_op_node(qc, op_node)
                obs[layer_idx, qubit, gate_idx] = 1

        return obs

    def indices_from_op_node(self, qc: QuantumCircuit, op_node: DAGOpNode) -> Tuple[int, int]:
        """
        Returns the qubit and gate class indices corresponding to a specific ``DAGOpNode`` in a
        ``QuantumCircuit``.

        :raises ValueError: if the gate represented by the ``DAGOpNode`` is not in the native gate set.
        """
        qubits = tuple(qc.find_bit(q)[0] for q in op_node.qargs)
        qubit = qubits[0]

        try:
            gate_idx = index_with_key(
                self.gate_classes,
                lambda c: c.equals_gate(qubit, op_node.op, qubits)
            )
        except ValueError as ex:
            print(op_node.op, qubits)
            raise ValueError('Circuit is incompatible with target gate set') from ex

        return qubit, gate_idx

    def _generate_random_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        while qc.depth() < self.max_depth:
            try:
                gate_class: GateClass = self.rng.choice(self.gate_classes)

                gate = gate_class.gate.copy()
                gate.params = self.rng.uniform(-pi, pi, len(gate.params)).tolist()
                qubits = gate_class.qubits(self.rng.integers(self.num_qubits))

                if -1 not in qubits:
                    qc.append(gate, qubits)
            except CircuitError:
                pass

        qc = transpile(qc, basis_gates=self.basis_gates)

        return qc

    def _set_valid_actions(self):
        decoded_actions = [self.decode_action(a) for a in range(self.action_space.n)]

        self.valid_actions = np.array([
            self.transformation_rules[rule].is_valid(self, layer, qubit)
            for layer, qubit, rule in decoded_actions
        ], dtype=Int8)


class ExactTransformationCircuitEnv(TransformationCircuitEnv):
    ObsType = NDArray[Literal['*, *, *'], Int8]

    def __init__(
        self,
        max_depth: int,
        num_qubits: int,
        gate_classes: List[GateClass],
        transformation_rules: List[TransformationRule],
        *,
        weight_depth: float = 1.0,
        weight_gate_count: float = 0.2,
        **kwargs,
    ):
        super().__init__(max_depth, num_qubits, gate_classes, transformation_rules, **kwargs)

        self.weight_depth = weight_depth
        self.weight_gate_count = weight_gate_count

        self.observation_space: spaces.MultiBinary = spaces.MultiBinary(
            (max_depth, num_qubits, len(gate_classes))
        )

    def reward(self) -> float:
        depth_diff = self.current_circuit.depth() - self.next_circuit.depth()
        gate_count_diff = self.current_circuit.size() - self.next_circuit.size()

        return self.weight_depth * depth_diff + self.weight_gate_count * gate_count_diff

    def observation(self) -> ObsType:
        return self.circuit_to_obs(self.current_circuit)

    def generate_next_circuit(self, decoded_action: TransformationCircuitEnv.DecodedAction) -> QuantumCircuit:
        rule = self.transformation_rules[decoded_action.rule]
        return rule.apply(self, decoded_action.layer, decoded_action.qubit)

    def reset_circuits(self):
        if self.target_circuit:
            self.current_circuit = self.target_circuit.copy()
            self._set_valid_actions()
        else:
            while True:
                self.current_circuit = self._generate_random_circuit()
                self._set_valid_actions()
                if self.valid_actions.any():
                    break
