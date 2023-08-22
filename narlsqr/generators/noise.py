
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Callable
from numbers import Number
from typing import Optional, Self

import numpy as np
from nptyping import NDArray
from qiskit.providers.models import BackendProperties
from qiskit.providers.models.backendproperties import Gate
from scipy.stats import gaussian_kde


def _get_error_rates_from_backend_properties(properties: BackendProperties) -> list[float]:
    gates: list[Gate] = properties.gates

    cnot_gates = sorted(
        (g for g in gates if g.gate == 'cx' and g.qubits[0] < g.qubits[1]),
        key=lambda g: g.qubits,
    )

    return [properties.gate_property(cnot.gate, cnot.qubits, 'gate_error')[0] for cnot in cnot_gates]

class NoiseGenerator(ABC):
    """
    Allows configuration of randomly generated gate error rates. Intended to be used when training noise-aware
    reinforcement learning models.

    :param num_edges: Number of two-qubit error rates to generate.
    :param min_error_rate: Minimum error rate.
    :param seed: Seed for the random number generator.
    """

    def __init__(self, num_edges: int, *, min_error_rate: float = 1e-3, seed: Optional[int] = None):
        if num_edges <= 0:
            raise ValueError(f'Number of edges must be positive, got {num_edges}')
        if min_error_rate < 0.0:
            raise ValueError(f'Minimum error rate must not be negative, got {min_error_rate}')

        self.num_edges = num_edges
        self.min_error_rate = min_error_rate
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def _generate(self) -> NDArray:
        raise NotImplementedError

    def generate(self) -> NDArray:
        return self._generate().clip(self.min_error_rate, 1.0)

    def seed(self, seed: Optional[int] = None):
        """Reseed the random number generator used to generate gate error rates."""
        self.rng = np.random.default_rng(seed)

class UniformNoiseGenerator(NoiseGenerator):
    """
    Generates gate error rates according to a normal distribution (clamped between 0 and 1).

    :param mean: Mean gate error rate.
    :param std: Standard deviation of gate error rates.
    """

    def __init__(
        self,
        num_edges: int,
        mean: float,
        std: float,
        *,
        min_error_rate: float = 1e-3,
        seed: Optional[int] = None,
    ):
        super().__init__(num_edges, min_error_rate=min_error_rate, seed=seed)

        self.mean = mean
        self.std = std

    @classmethod
    def from_samples(cls, num_edges: int, samples: Iterable[float], **kwargs) -> Self:
        """
        Create a uniform noise generator using the mean and standard deviation of a set of samples.
        """
        samples = np.array(samples, copy=False)
        return cls(num_edges, samples.mean(), samples.std(), **kwargs)

    @classmethod
    def from_backend_properties(cls, properties: BackendProperties, **kwargs) -> Self:
        """
        Create a uniform noise generator from a ``BackendProperties`` object containing device calibration data.
        """
        error_rates = _get_error_rates_from_backend_properties(properties)
        return cls.from_samples(len(error_rates), error_rates, **kwargs)

    @classmethod
    def from_calibration_file(cls, path: str, **kwargs) -> Self:
        """
        Create a noise generator from a JSON file containing device calibration data.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_backend_properties(BackendProperties.from_dict(data), **kwargs)

    def _generate(self) -> NDArray:
        return self.rng.normal(self.mean, self.std, self.num_edges)

class KdeNoiseGenerator(NoiseGenerator):
    """
    Applies Gaussian `kernel density estimation <https://en.wikipedia.org/wiki/Kernel_density_estimation>`_ to
    a set of gate error rate samples and generates error rates according to the resulting probability distribution.

    :param samples: Gate error rate samples.
    :param bw_method: Method used to calculate the estimator bandwidth (passed to ``scipy.stats.gaussian_kde``, see
        `SciPy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html>`_
        for more information).
    """

    def __init__(
        self,
        num_edges: int,
        samples: Iterable[float],
        *,
        min_error_rate: float = 1e-3,
        seed: Optional[int] = None,
        bw_method: Optional[str | Number | Callable] = None
    ):
        super().__init__(num_edges, min_error_rate=min_error_rate, seed=seed)

        self.kde = gaussian_kde(samples, bw_method=bw_method)

    @classmethod
    def from_backend_properties(cls, properties: BackendProperties, **kwargs) -> Self:
        """
        Create a KDE noise generator from a ``BackendProperties`` object containing device calibration data.
        """
        error_rates = _get_error_rates_from_backend_properties(properties)
        return cls(len(error_rates), error_rates, **kwargs)

    @classmethod
    def from_calibration_file(cls, path: str, **kwargs) -> Self:
        """
        Create a KDE noise generator from a JSON file containing device calibration data.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_backend_properties(BackendProperties.from_dict(data), **kwargs)

    def _generate(self) -> NDArray:
        return np.squeeze(self.kde.resample(self.num_edges, self.rng))
