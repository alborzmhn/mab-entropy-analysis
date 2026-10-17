from __future__ import annotations

from collections.abc import Sequence

import numpy as np


# === File-local configuration ===
# Valid lower bound for Bernoulli arm means.
MIN_BERNOULLI_MEAN = 0.0
# Valid upper bound for Bernoulli arm means.
MAX_BERNOULLI_MEAN = 1.0


class BernoulliBandit:
    """Stationary stochastic Bernoulli multi-armed bandit."""

    def __init__(self, means: Sequence[float], seed: int | None = None) -> None:
        if len(means) == 0:
            raise ValueError("means must contain at least one arm.")

        means_array = np.asarray(means, dtype=float)

        if means_array.ndim != 1:
            raise ValueError("means must be one-dimensional.")

        if np.any(
            (means_array < MIN_BERNOULLI_MEAN)
            | (means_array > MAX_BERNOULLI_MEAN)
        ):
            raise ValueError("All Bernoulli means must lie in [0, 1].")

        self._means = means_array
        self._rng = np.random.default_rng(seed)

    @property
    def means(self) -> np.ndarray:
        return self._means.copy()

    @property
    def n_arms(self) -> int:
        return int(self._means.size)

    @property
    def optimal_arm(self) -> int:
        return int(np.argmax(self._means))

    @property
    def optimal_mean(self) -> float:
        return float(self._means[self.optimal_arm])

    def arm_mean(self, arm: int) -> float:
        self._validate_arm(arm)
        return float(self._means[arm])

    def pull(self, arm: int) -> int:
        self._validate_arm(arm)
        return int(self._rng.random() < self._means[arm])

    def _validate_arm(self, arm: int) -> None:
        if not isinstance(arm, (int, np.integer)):
            raise TypeError("arm must be an integer index.")
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(
                f"arm index {arm} is out of range for {self.n_arms} arms."
            )
