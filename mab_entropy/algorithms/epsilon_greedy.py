from __future__ import annotations

import math

import numpy as np

from .base import BanditAlgorithm


# === File-local configuration ===
# Exploration schedules implemented by this module.
SUPPORTED_SCHEDULES = {"constant", "inverse_sqrt"}


class EpsilonGreedy(BanditAlgorithm):
    """Epsilon-greedy with constant or inverse-square-root exploration."""

    def __init__(
        self,
        n_arms: int,
        *,
        epsilon: float,
        schedule: str,
        min_epsilon: float,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_arms=n_arms)

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1].")
        if not 0.0 <= min_epsilon <= 1.0:
            raise ValueError("min_epsilon must lie in [0, 1].")
        if schedule not in SUPPORTED_SCHEDULES:
            raise ValueError(
                "schedule must be either 'constant' or 'inverse_sqrt'."
            )

        self.epsilon = float(epsilon)
        self.schedule = schedule
        self.min_epsilon = float(min_epsilon)
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)
        self.total_pulls = 0
        self._rng = np.random.default_rng(self._seed)

    @property
    def value_estimates(self) -> np.ndarray:
        values = np.zeros(self.n_arms, dtype=float)
        observed = self.counts > 0
        values[observed] = self.reward_sums[observed] / self.counts[observed]
        return values

    @property
    def current_epsilon(self) -> float:
        if self.schedule == "constant":
            return self.epsilon

        t = self.total_pulls + 1
        return max(self.min_epsilon, self.epsilon / math.sqrt(t))

    def action_probabilities(self) -> np.ndarray:
        epsilon_t = self.current_epsilon
        probabilities = np.full(
            self.n_arms,
            epsilon_t / self.n_arms,
            dtype=float,
        )

        values = self.value_estimates
        maximizers = np.flatnonzero(np.isclose(values, np.max(values)))
        probabilities[maximizers] += (1.0 - epsilon_t) / maximizers.size
        return probabilities

    def select_arm(self) -> int:
        probabilities = self.action_probabilities()
        return int(self._rng.choice(self.n_arms, p=probabilities))

    def update(self, arm: int, reward: float) -> None:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(f"arm index {arm} is out of range.")
        if reward < 0.0 or reward > 1.0:
            raise ValueError("reward must lie in [0, 1].")

        self.counts[arm] += 1
        self.reward_sums[arm] += float(reward)
        self.total_pulls += 1
