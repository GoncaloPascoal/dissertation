
from abc import ABC, abstractmethod
from collections.abc import Iterable, Callable
from dataclasses import dataclass, field
from math import e
from numbers import Number
from typing import Optional, Self

import numpy as np
from nptyping import NDArray
from scipy.stats import gaussian_kde


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """
    Configuration values for controlling rewards in noise-aware routing environments.

    :ivar log_base: Base used to calculate log reliabilities from gate error rates.
    :ivar min_log_reliability: Greatest penalty that can be issued from scheduling a gate, to prevent infinite rewards.
                               Cannot be positive.
    :ivar added_gate_reward: Flat value that will be added to the reward associated with each two-qubit gate from the
                             original circuit. Cannot be negative.
    """
    log_base: float = field(default=e, kw_only=True)
    min_log_reliability: float = field(default=-100.0, kw_only=True)
    added_gate_reward: float = field(default=0.02, kw_only=True)

    def __post_init__(self):
        if self.log_base <= 1.0:
            raise ValueError(f'Logarithm base must be greater than 1, got {self.log_base}')
        if self.min_log_reliability > 0.0:
            raise ValueError(f'Minimum log reliability cannot be positive, got {self.min_log_reliability}')
        if self.added_gate_reward < 0.0:
            raise ValueError(f'Added gate reward cannot be negative, got {self.added_gate_reward}')

    def calculate_log_reliabilities(self, error_rates: NDArray) -> NDArray:
        if np.any((error_rates < 0.0) | (error_rates > 1.0)):
            raise ValueError('Got invalid values for error rates')

        return np.where(
            error_rates < 1.0,
            np.emath.logn(self.log_base, 1.0 - error_rates),
            -np.inf
        ).clip(self.min_log_reliability)


class NoiseGenerator(ABC):
    """
    Allows configuration of randomly generated gate error rates. Intended to be used when training noise-aware
    reinforcement learning models.
    """
    def __init__(self, *, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate_error_rates(self, n: int) -> NDArray:
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None):
        """Reseed the random number generator used to generate gate error rates."""
        self.rng = np.random.default_rng(seed)

class UniformNoiseGenerator(NoiseGenerator):
    """
    Generates gate error rates according to a normal distribution (clamped between 0 and 1).

    :param mean: Mean gate error rate.
    :param std: Standard deviation of gate error rates.
    """

    def __init__(self, mean: float, std: float, *, seed: Optional[int] = None):
        super().__init__(seed=seed)

        self.mean = mean
        self.std = std

    @classmethod
    def from_samples(cls, samples: Iterable[float]) -> Self:
        """
        Create a noise generator using the mean and standard deviation of a set of samples.
        """
        samples = np.array(samples, copy=False)
        return cls(samples.mean(), samples.std())

    def generate_error_rates(self, n: int) -> NDArray:
        return np.random.normal(self.mean, self.std, n).clip(0.0, 1.0)

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
        samples: Iterable[float],
        *,
        seed: Optional[int] = None,
        bw_method: Optional[str | Number | Callable] = None
    ):
        super().__init__(seed=seed)

        self.kde = gaussian_kde(samples, bw_method=bw_method)

    def generate_error_rates(self, n: int) -> NDArray:
        return np.squeeze(self.kde.resample(n)).clip(0.0, 1.0)
