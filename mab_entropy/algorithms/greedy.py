from __future__ import annotations

import numpy as np

from .base import BanditAlgorithm


# === File-local configuration ===
# Initial empirical value assigned to every unobserved arm.
INITIAL_VALUE_ESTIMATE = 0.0


class Greedy(BanditAlgorithm):
    """Pure greedy with randomized tie-breaking and no forced initialization."""

    def __init__(self, n_arms: int, *, seed: int | None = None) -> None:
        super().__init__(n_arms=n_arms)
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)
        self._rng = np.random.default_rng(self._seed)

    @property
    def value_estimates(self) -> np.ndarray:
        values = np.full(
            self.n_arms,
            INITIAL_VALUE_ESTIMATE,
            dtype=float,
        )
        observed = self.counts > 0
        values[observed] = self.reward_sums[observed] / self.counts[observed]
        return values

    def action_probabilities(self) -> np.ndarray:
        values = self.value_estimates
        maximizers = np.flatnonzero(np.isclose(values, np.max(values)))
        probabilities = np.zeros(self.n_arms, dtype=float)
        probabilities[maximizers] = 1.0 / maximizers.size
        return probabilities

    def select_arm(self) -> int:
        probabilities = self.action_probabilities()
        return int(self._rng.choice(self.n_arms, p=probabilities))

    def update(self, arm: int, reward: float) -> None:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(f"arm index {arm} is out of range.")
        if reward < 0.0 or reward > 1.0:
            raise ValueError("For this phase, reward must lie in [0, 1].")

        self.counts[arm] += 1
        self.reward_sums[arm] += float(reward)
