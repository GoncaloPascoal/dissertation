
import random
from abc import ABC, abstractmethod

import numpy as np
from qiskit import QuantumCircuit


class CircuitGenerator(ABC):
    def __init__(self, num_qubits: int):
        if num_qubits < 2:
            raise ValueError(f'`num_qubits` must be greater or equal than 2, got {num_qubits}')

        self.num_qubits = num_qubits

    @abstractmethod
    def generate(self) -> QuantumCircuit:
        raise NotImplementedError


class Random(CircuitGenerator):
    def __init__(self, num_qubits: int, num_gates: int):
        super().__init__(num_qubits)

        if num_gates <= 0:
            raise ValueError(f'`num_gates` must be positive, got {num_gates}')

        self.num_gates = num_gates

    def generate(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        for _ in range(self.num_gates):
            qubits = np.random.choice(self.num_qubits, 2, replace=False)
            qc.cx(*qubits)

        return qc


class LayeredRandom(CircuitGenerator):
    def __init__(self, num_qubits: int, num_layers: int = 1, density: float = 1.0):
        super().__init__(num_qubits)

        if num_layers <= 0:
            raise ValueError(f'`num_layers` must be positive, got {num_layers}')

        if not (0.0 <= density <= 1.0):
            raise ValueError(f'`density` must be a value between 0 and 1, got {density}')

        self.num_layers = num_layers
        self.density = density

        self.cnots_per_layer = int(density * num_qubits // 2)

    def generate(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        qubits = list(range(self.num_qubits))

        for _ in range(self.num_layers):
            selected = random.sample(qubits, 2 * self.cnots_per_layer)
            for i in range(0, len(selected), 2):
                qc.cx(*selected[i:i + 2])

        return qc
