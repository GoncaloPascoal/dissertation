
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from math import pi
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from parameter_generator import ParameterGenerator
from utils import ContinuousOptimizationFunction, index_with_key


Observation = Tuple[np.ndarray, np.ndarray]
Action = int


def default_qubit_callback(qubit: int) -> Tuple[int, ...]:
    return qubit,

def nn_qubit_callback(qubit: int) -> Tuple[int, ...]:
    return qubit, qubit + 1

def nn_qubit_callback_reversed(qubit: int) -> Tuple[int, ...]:
    return qubit + 1, qubit


@dataclass
class InstructionCallback:
    QubitCallback = Callable[[int], Tuple[int, ...]]

    instruction: Instruction
    qubit_callback: QubitCallback = default_qubit_callback

    def equals_instruction(
        self,
        other: Instruction,
        other_qubits: Tuple[int, ...],
        qubit: int,
    ) -> bool:
        if (
            self.instruction.name != other.name or
            self.qubit_callback(qubit) != other_qubits or
            len(self.instruction.params) != len(other.params)
        ):
            return False

        for param, other_param in zip(self.instruction.params, other.params):
            if not isinstance(param, ParameterExpression) and param != other_param:
                return False

        return True

    def __call__(self, param_gen: ParameterGenerator, qc: QuantumCircuit, qubit: int):
        instruction = self.instruction
        if self.instruction.is_parameterized():
            instruction = param_gen.parameterize(instruction)

        qc.append(instruction, self.qubit_callback(qubit))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.instruction.name!r}, {self.qubit_callback!r})'


class TransformationRule(ABC):
    @abstractmethod
    def is_valid(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def perform(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> QuantumCircuit:
        raise NotImplementedError

    @staticmethod
    def _dag_layers(env: 'TransformationCircuitEnv') -> List[DAGCircuit]:
        # TODO: Try to optimize this to reduce the cost of modifying circuits
        dag = circuit_to_dag(env.current_circuit)
        return [dct['graph'] for dct in dag.layers()]

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class SwapInstructions(TransformationRule):
    def is_valid(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> bool:
        qc = env.current_circuit
        layers = TransformationRule._dag_layers(env)

        if layer >= len(layers) - 1:
            return False

        inst_a = env.op_node_at(qc, layers[layer], qubit)
        inst_b = env.op_node_at(qc, layers[layer + 1], qubit)

        if inst_a is None or inst_b is None:
            # Couldn't find instructions to swap at the specified location
            return False

        if inst_a.op == inst_b.op and inst_a.qargs == inst_b.qargs:
            # Same instruction, swap operation will have no effect
            return False

        # Ensure that swapping instructions will not increase circuit depth (there is no overlap
        # with other instructions)
        qubits_a = set(qc.find_bit(q)[0] for q in inst_a.qargs)
        qubits_b = set(qc.find_bit(q)[0] for q in inst_b.qargs)

        for q in qubits_a - qubits_b:
            if env.op_node_at(qc, layers[layer + 1], q) is not None:
                return False

        for q in qubits_b - qubits_a:
            if env.op_node_at(qc, layers[layer], q) is not None:
                return False

        return True

    def perform(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> QuantumCircuit:
        qc = env.current_circuit
        dag = circuit_to_dag(qc)
        layers = TransformationRule._dag_layers(env)

        inst_a = env.op_node_at(qc, layers[layer], qubit)
        inst_b = env.op_node_at(qc, layers[layer + 1], qubit)

        dag = dag.copy_empty_like()
        for i, dag_layer in enumerate(layers):
            for op_node in dag_layer.op_nodes(include_directives=False):
                qubits = [qc.find_bit(q)[0] for q in op_node.qargs]

                if i == layer and qubit in qubits:
                    op_node = inst_b
                elif i == layer + 1 and qubit in qubits:
                    op_node = inst_a

                dag.apply_operation_back(op_node.op, op_node.qargs, op_node.cargs)

        return dag_to_circuit(dag)


class ShiftQubit(TransformationRule):
    def __init__(self, down: bool = True):
        self.offset = 1 if down else -1

    def is_valid(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> bool:
        qc = env.current_circuit
        layers = TransformationRule._dag_layers(env)

        if layer >= len(layers):
            return False

        dag_layer = layers[layer]
        inst = env.op_node_at(qc, dag_layer, qubit)

        # Target location must contain an instruction
        if inst is None:
            return False

        _, first_qubit = env.indices_from_op_node(qc, inst)

        # Eliminate duplicate transformations for 2-qubit instructions
        if first_qubit != qubit:
            return False

        qubits_dst = tuple(qc.find_bit(q)[0] + self.offset for q in inst.qargs)

        # Destination qubits are valid
        if any(q not in range(env.num_qubits) for q in qubits_dst):
            return False

        for q in qubits_dst:
            inst_dst = env.op_node_at(qc, dag_layer, q)
            if inst_dst is not None and inst_dst != inst:
                return False

        return True

    def perform(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> QuantumCircuit:
        qc = env.current_circuit
        dag = circuit_to_dag(qc)
        layers = TransformationRule._dag_layers(env)
        dag_layer = layers[layer]

        instruction = env.op_node_at(qc, dag_layer, qubit)
        qubits = tuple(qc.find_bit(q)[0] for q in instruction.qargs)
        qargs_dst = tuple(qc.qubits[q + self.offset] for q in qubits)

        dag = dag.copy_empty_like()
        for i, dag_layer in enumerate(layers):
            if i == layer:
                for op_node in dag_layer.op_nodes(include_directives=False):
                    if op_node.qargs == instruction.qargs:
                        op_node.qargs = qargs_dst
                    dag.apply_operation_back(op_node.op, op_node.qargs, op_node.cargs)
            else:
                dag.compose(dag_layer, inplace=True)

        return dag_to_circuit(dag)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.offset})'


class RemoveInstruction(TransformationRule):
    def is_valid(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> bool:
        qc = env.current_circuit
        layers = TransformationRule._dag_layers(env)

        if layer >= len(layers):
            return False

        dag_layer = layers[layer]
        op_node = env.op_node_at(qc, dag_layer, qubit)

        if op_node is None:
            return False

        _, first_qubit = env.indices_from_op_node(qc, op_node)

        # Eliminate duplicate transformations for 2-qubit instructions
        if first_qubit != qubit:
            return False

        return True

    def perform(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> QuantumCircuit:
        qc = env.current_circuit
        dag = circuit_to_dag(qc)
        layers = TransformationRule._dag_layers(env)
        dag_layer = layers[layer]

        dag.remove_op_node(env.op_node_at(qc, dag_layer, qubit))
        return dag_to_circuit(dag)


class AddInstruction(TransformationRule):
    def __init__(self, instruction_callback: InstructionCallback):
        self.instruction_callback = instruction_callback

    def is_valid(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> bool:
        qc = env.current_circuit
        depth = qc.depth()

        if layer > depth:
            return False

        qubits = self.instruction_callback.qubit_callback(qubit)
        if any(q >= env.num_qubits for q in qubits):
            # Instruction qubits must be within bounds
            return False

        if layer == depth and depth < env.max_depth:
            # Can insert instruction the end of the circuit without exceeding max depth
            return True

        layers = TransformationRule._dag_layers(env)
        dag_layer = layers[layer]
        if any(env.op_node_at(qc, dag_layer, q) is not None for q in qubits):
            return False

        if layer > 0:
            previous_layer = layers[layer - 1]
            if all(env.op_node_at(qc, previous_layer, q) is None for q in qubits):
                return False

        return True

    def perform(self, env: 'TransformationCircuitEnv', layer: int, qubit: int) -> QuantumCircuit:
        qc = env.current_circuit

        if layer == qc.depth():
            # Append instruction at the end of the circuit
            qc = qc.copy()
            self.instruction_callback(env.param_gen, qc, qubit)
            return qc

        dag = circuit_to_dag(qc)
        layers = TransformationRule._dag_layers(env)
        dag = dag.copy_empty_like()

        instruction = self.instruction_callback.instruction
        if instruction.is_parameterized():
            instruction = env.param_gen.parameterize(instruction)
        qargs = tuple(qc.qubits[q] for q in self.instruction_callback.qubit_callback(qubit))

        for i, dag_layer in enumerate(layers):
            if i == layer:
                dag_layer.apply_operation_front(instruction, qargs)
            dag.compose(dag_layer, qc.qubits, qc.clbits, inplace=True)

        return dag_to_circuit(dag)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.instruction_callback!r})'


class TransformationCircuitEnv(gym.Env):
    def __init__(self, context: Dict[str, Any]):
        self.max_depth: int = context['max_depth']
        self.num_qubits: int = context['num_qubits']
        self.instruction_callbacks: List[InstructionCallback] = context['instruction_callbacks']
        self.transformation_rules: List[TransformationRule] = context['transformation_rules']
        self.continuous_optimization: ContinuousOptimizationFunction = context['continuous_optimization']
        self.u: Optional[QuantumCircuit] = context.get('u', None)

        if self.u is not None and self.u.num_qubits > self.num_qubits:
            raise ValueError(f'Target circuit has {self.u.num_qubits} qubits but the observation space '
                             f'only has {self.num_qubits} qubits')

        self.rng = np.random.default_rng()
        self.param_gen = ParameterGenerator()
        self.basis_gates = [callback.instruction.name for callback in self.instruction_callbacks]

        num_actions = self.max_depth * self.num_qubits * len(self.transformation_rules)
        # Use Box for observations instead of MultiBinary due to bugs with RLlib
        self.observation_space: spaces.Tuple = spaces.Tuple((
            spaces.Box(0.0, 1.0, shape=(num_actions,)),
            spaces.Box(0, 1, shape=(self.max_depth, self.num_qubits, len(self.instruction_callbacks))),
        ))
        self.action_space: spaces.Discrete = spaces.Discrete(num_actions)
        self.valid_actions: np.ndarray = np.zeros(num_actions)

        self.current_circuit = QuantumCircuit(self.num_qubits)
        self.current_cost = 1.0
        self.target_circuit = QuantumCircuit(self.num_qubits)
        self.epoch = 0

    @classmethod
    def from_args(
        cls,
        max_depth: int,
        num_qubits: int,
        instruction_callbacks: List[InstructionCallback],
        transformation_rules: List[TransformationRule],
        continuous_optimization: ContinuousOptimizationFunction,
        u: Optional[QuantumCircuit] = None,
    ) -> 'TransformationCircuitEnv':
        return cls({
            'max_depth': max_depth,
            'num_qubits': num_qubits,
            'instruction_callbacks': instruction_callbacks,
            'transformation_rules': transformation_rules,
            'continuous_optimization': continuous_optimization,
            'u': u,
        })

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        if not self.valid_actions[action]:
            layer, qubit, rule = self._parse_action(action)
            rule_type = self.transformation_rules[rule].__class__.__name__
            raise ValueError(f'An invalid action was selected: {rule_type} on layer {layer} and qubit {qubit}')

        layer, qubit, rule = np.unravel_index(
            action,
            shape=(self.max_depth, self.num_qubits, len(self.transformation_rules)),
        )

        next_circuit = self.transformation_rules[rule].perform(self, layer, qubit)

        self.current_circuit = next_circuit
        self.epoch += 1

        reward = self._reward(next_circuit)
        terminated = self.epoch == 32

        obs = self.current_observation()
        self.valid_actions = obs[0]

        return obs, reward, terminated, False, {}

    def reset(self, *, seed=None, options=None) -> Tuple[Observation, Dict[str, Any]]:
        if self.u is not None:
            # Evaluation
            self.current_circuit = self.u.copy()
        else:
            # Training
            self.current_circuit = self._generate_random_circuit()

        self.target_circuit = self.current_circuit.copy()
        self.epoch = 0

        self.current_cost = 0.0
        self.current_circuit = self.param_gen.parameterize_circuit(self.current_circuit)

        obs = self.current_observation()
        self.valid_actions = obs[0]

        return obs, {}

    def action_mask(self) -> np.ndarray:
        actions = [self._parse_action(a) for a in range(self.action_space.n)]

        return np.array([
            self.transformation_rules[rule].is_valid(self, layer, qubit)
            for layer, qubit, rule in actions
        ], dtype=self.observation_space.spaces[0].dtype)

    def indices_from_op_node(self, qc: QuantumCircuit, op_node: DAGOpNode) -> Tuple[int, int]:
        """
        Returns the (observation) instruction and qubit indices corresponding to a specific ``DAGOpNode`` in a
        ``QuantumCircuit``.

        :raises ValueError: if the instruction represented by the ``DAGOpNode`` is not in the native instruction set.
        """
        qubit = min(qc.find_bit(q)[0] for q in op_node.qargs)
        qubits = tuple(qc.find_bit(q).index for q in op_node.qargs)

        try:
            instruction_idx = index_with_key(
                self.instruction_callbacks,
                lambda c: c.equals_instruction(op_node.op, qubits, qubit)
            )
        except ValueError as ex:
            raise ValueError('Circuit is incompatible with target instruction set') from ex

        return instruction_idx, qubit

    @staticmethod
    def op_node_at(qc: QuantumCircuit, layer: DAGCircuit, qubit: int) -> Optional[DAGOpNode]:
        target_node = None

        for op_node in layer.op_nodes(include_directives=False):
            if qubit in (qc.find_bit(q)[0] for q in op_node.qargs):
                target_node = op_node
                break

        return target_node

    def current_observation(self) -> Observation:
        return self._circuit_to_obs(self.current_circuit)

    def _generate_random_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        while qc.depth() < self.max_depth:
            try:
                ic = self.rng.choice(self.instruction_callbacks)
                ic(self.param_gen, qc, self.rng.integers(self.num_qubits))
            except CircuitError:
                pass

        qc = transpile(qc, basis_gates=self.basis_gates)
        qc.assign_parameters(self.rng.uniform(-pi, pi, qc.num_parameters), inplace=True)

        return qc

    def _parse_action(self, action: Action) -> Tuple[int, int, int]:
        # noinspection PyTypeChecker
        return np.unravel_index(
            action,
            shape=(self.max_depth, self.num_qubits, len(self.transformation_rules)),
        )

    def _reward(self, next_circuit: QuantumCircuit) -> float:
        _, next_cost = self.continuous_optimization(self.target_circuit, next_circuit)

        depth_next = next_circuit.depth()
        depth_current = self.current_circuit.depth()
        depth_target = self.target_circuit.depth()

        incremental_weight = 1.0e-4
        incremental_cost_diff = self.current_cost - next_cost
        incremental_depth_diff = (depth_current - depth_next) / depth_target
        incremental_reward = incremental_weight * (incremental_depth_diff + incremental_cost_diff)

        depth_diff = (depth_target - depth_next) / depth_target
        depth_exponent = 0.7
        cost_exponent = 5.0

        if depth_diff > 0.0:
            discovery_reward = depth_diff ** depth_exponent * next_cost ** cost_exponent
        else:
            discovery_reward = 0.0

        return incremental_reward + discovery_reward

    def _circuit_to_obs(self, qc: QuantumCircuit) -> Observation:
        if qc.depth() > self.max_depth:
            raise ValueError(f'Circuit depth must not exceed {self.max_depth}')

        orig_obs = self.observation_space.spaces[1]
        orig_obs = np.zeros(shape=orig_obs.shape, dtype=orig_obs.dtype)
        dag = circuit_to_dag(qc)

        for layer_idx, layer in enumerate(dag.layers()):
            for op_node in layer['graph'].op_nodes(include_directives=False):
                instruction_idx, qubit = self.indices_from_op_node(qc, op_node)
                orig_obs[layer_idx, qubit, instruction_idx] = 1

        return self.action_mask(), orig_obs

    def _obs_to_circuit(self, obs: Observation) -> QuantumCircuit:
        qc = QuantumCircuit(self.u.num_qubits)

        for layer in obs[1]:
            for first_qubit, instructions in enumerate(layer):
                indices = np.flatnonzero(instructions)
                if indices.size > 0:
                    self.instruction_callbacks[indices[0]](self.param_gen, qc, first_qubit)

        return qc


def main():
    from qiskit.circuit.library import CXGate, SXGate, RZGate
    from qiskit.circuit import Parameter
    from ray.rllib.algorithms.ppo import PPOConfig
    from rich import print
    from action_mask_model import TorchActionMaskModel

    from gradient_based import gradient_based_hst_weighted
    from tqdm.rich import trange

    u = QuantumCircuit(2)
    u.cz(0, 1)
    u = transpile(u, basis_gates=['cx', 'sx', 'rz'])

    max_depth = 8
    num_qubits = 2

    sx = SXGate()
    rz = RZGate(Parameter('x'))
    cx = CXGate()

    instruction_callbacks = [
        InstructionCallback(sx),
        InstructionCallback(rz),
        InstructionCallback(cx, nn_qubit_callback),
        InstructionCallback(cx, nn_qubit_callback_reversed),
    ]

    transformation_rules = [
        SwapInstructions(),
        ShiftQubit(down=False),
        ShiftQubit(down=True),
        RemoveInstruction(),
        *(AddInstruction(cb) for cb in instruction_callbacks),
    ]

    config = (
        PPOConfig()
        .training(
            model={
                'custom_model': TorchActionMaskModel,
            },
            train_batch_size=32,
            sgd_minibatch_size=4,
        )
        .debugging(log_level='ERROR')
        .environment(
            TransformationCircuitEnv,
            env_config={
                'max_depth': max_depth,
                'num_qubits': num_qubits,
                'instruction_callbacks': instruction_callbacks,
                'transformation_rules': transformation_rules,
                'continuous_optimization': gradient_based_hst_weighted,
            },
            disable_env_checking=True,
        )
        .fault_tolerance(
            recreate_failed_workers=True,
            restart_failed_sub_environments=True,
        )
        .framework('torch')
        .rollouts(
            num_rollout_workers=2,
        )
    )

    algo = config.build()
    # algo.restore('tce/checkpoints/checkpoint_000091')

    for i in trange(100):
        algo.train()
        if i % 10 == 0:
            checkpoint_dir = algo.save('tce/checkpoints')
            print(f'Iteration {i}: saved checkpoint in {checkpoint_dir}')

    env = TransformationCircuitEnv.from_args(max_depth, num_qubits, instruction_callbacks, transformation_rules,
                                             gradient_based_hst_weighted, u)
    obs, _ = env.reset()
    terminated = False

    while not terminated:
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, terminated, *_ = env.step(action)
        print(f'Epoch {env.epoch}: {reward}')

    print(env.current_circuit)
    print(env.current_cost)


if __name__ == '__main__':
    main()
