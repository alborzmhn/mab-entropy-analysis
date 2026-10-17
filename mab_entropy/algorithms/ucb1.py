from __future__ import annotations

import math

import numpy as np

from .base import BanditAlgorithm

# === File-local configuration ===
# Minimum time value used before taking the logarithm.
MIN_LOG_TIME = 1


class UCB1(BanditAlgorithm):
    """Deterministic UCB1 with deterministic initialization and tie-breaking."""

    def __init__(
        self,
        n_arms: int,
        *,
        exploration_coefficient: float,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_arms=n_arms)

        if exploration_coefficient <= 0.0:
            raise ValueError("exploration_coefficient must be positive.")

        self.exploration_coefficient = float(exploration_coefficient)
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)
        self.total_pulls = 0

    @property
    def value_estimates(self) -> np.ndarray:
        values = np.zeros(self.n_arms, dtype=float)
        observed = self.counts > 0
        values[observed] = (
            self.reward_sums[observed] / self.counts[observed]
        )
        return values

    def action_probabilities(self) -> np.ndarray:
        probabilities = np.zeros(self.n_arms, dtype=float)

        # Deterministic initialization:
        # pull unseen arms in fixed index order.
        unseen = np.flatnonzero(self.counts == 0)

        if unseen.size > 0:
            selected_arm = int(unseen[0])
            probabilities[selected_arm] = 1.0
            return probabilities

        t = max(MIN_LOG_TIME, self.total_pulls)

        bonus = np.sqrt(
            self.exploration_coefficient
            * math.log(t)
            / self.counts
        )

        scores = self.value_estimates + bonus

        # np.argmax deterministically chooses the first maximum.
        selected_arm = int(np.argmax(scores))

        probabilities[selected_arm] = 1.0

        return probabilities

    def select_arm(self) -> int:
        probabilities = self.action_probabilities()
        return int(np.argmax(probabilities))

    def update(self, arm: int, reward: float) -> None:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(f"arm index {arm} is out of range.")

        if reward < 0.0 or reward > 1.0:
            raise ValueError("reward must lie in [0, 1].")

        self.counts[arm] += 1
        self.reward_sums[arm] += float(reward)
        self.total_pulls += 1