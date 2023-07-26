
from abc import ABC, abstractmethod
from typing import Optional, Self

import numpy as np
from qiskit import QuantumCircuit

from revlib import files_in_dir


class CircuitGenerator(ABC):
    def __init__(self, *, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(self) -> QuantumCircuit:
        raise NotImplementedError


class RandomCircuitGenerator(CircuitGenerator):
    def __init__(self, num_qubits: int, num_gates: int, *, seed: Optional[int] = None):
        super().__init__(seed=seed)

        if num_qubits < 2:
            raise ValueError(f'Number of qubits must be greater or equal than 2, got {num_qubits}')
        if num_gates <= 0:
            raise ValueError(f'Number of gates must be positive, got {num_gates}')

        self.num_qubits = num_qubits
        self.num_gates = num_gates

    def generate(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        for _ in range(self.num_gates):
            qubits = self.rng.choice(self.num_qubits, 2, replace=False)
            qc.cx(*qubits)

        return qc


class LayeredCircuitGenerator(CircuitGenerator):
    def __init__(self, num_qubits: int, num_layers: int = 1, density: float = 1.0, *, seed: Optional[int] = None):
        super().__init__(seed=seed)

        if num_qubits < 2:
            raise ValueError(f'Number of qubits must be greater or equal than 2, got {num_qubits}')
        if num_layers <= 0:
            raise ValueError(f'Number of layers must be positive, got {num_layers}')
        if not (0.0 <= density <= 1.0):
            raise ValueError(f'Density must be a value between 0 and 1, got {density}')

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.density = density

        self.cnots_per_layer = int(density * num_qubits // 2)

    def generate(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        qubits = list(range(self.num_qubits))

        for _ in range(self.num_layers):
            selected = self.rng.choice(qubits, 2 * self.cnots_per_layer, replace=False)
            for i in range(0, len(selected), 2):
                qc.cx(*selected[i:i + 2])

        return qc


class DatasetCircuitGenerator(CircuitGenerator):
    def __init__(self, dataset: list[QuantumCircuit], *, random: bool = False, seed: Optional[int] = None):
        super().__init__(seed=seed)

        if not dataset:
            raise ValueError('Dataset cannot be empty')

        self.dataset = dataset
        self.random = random
        self.idx = 0

    @classmethod
    def from_dir(cls, path: str, *, random: bool = False, seed: Optional[int] = None) -> Self:
        dataset = [QuantumCircuit.from_qasm_file(file_path) for file_path in files_in_dir(path)]
        return cls(dataset, random=random, seed=seed)

    def generate(self) -> QuantumCircuit:
        if self.random:
            qc = self.rng.choice(self.dataset)
        else:
            qc = self.dataset[self.idx]
            self.idx = (self.idx + 1) % len(self.dataset)

        return qc
