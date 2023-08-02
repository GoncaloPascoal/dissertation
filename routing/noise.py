
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import e

import numpy as np
from nptyping import NDArray


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

    :param recalibration_interval: Error rates will be regenerated after routing this many circuits.
    """

    def __init__(self, *, recalibration_interval: int = 16):
        if recalibration_interval <= 0:
            raise ValueError(f'Recalibration interval must be positive, got {recalibration_interval}')

        self.recalibration_interval = recalibration_interval

    @abstractmethod
    def generate_error_rates(self, n: int) -> NDArray:
        raise NotImplementedError


class UniformNoiseGenerator(NoiseGenerator):
    """
    Generates gate error rates according to a normal distribution (clamped between 0 and 1).

    :ivar mean: Mean gate error rate.
    :ivar std: Standard deviation of gate error rates.
    """
    mean: float
    std: float
    recalibration_interval: int = field(default=16, kw_only=True)

    def __init__(self, mean: float, std: float, *, recalibration_interval: int = 16):
        super().__init__(recalibration_interval=recalibration_interval)

        self.mean = mean
        self.std = std

    def generate_error_rates(self, n: int) -> NDArray:
        return np.random.normal(self.mean, self.std, n).clip(0.0, 1.0)

