from __future__ import annotations

import numpy as np

from .base import BanditAlgorithm


# === File-local configuration ===
# Initial log-weight assigned to every arm.
INITIAL_LOG_WEIGHT = 0.0


class EXP3(BanditAlgorithm):
    """EXP3 for rewards in [0, 1], implemented with log-weights."""

    def __init__(
        self,
        n_arms: int,
        *,
        gamma: float,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_arms=n_arms)

        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1].")

        self.gamma = float(gamma)
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self.log_weights = np.full(
            self.n_arms,
            INITIAL_LOG_WEIGHT,
            dtype=float,
        )
        self._rng = np.random.default_rng(self._seed)
        self._last_probabilities: np.ndarray | None = None
        self._last_selected_arm: int | None = None

    def action_probabilities(self) -> np.ndarray:
        shifted = self.log_weights - np.max(self.log_weights)
        weights = np.exp(shifted)
        normalized = weights / weights.sum()
        return (1.0 - self.gamma) * normalized + self.gamma / self.n_arms

    def select_arm(self) -> int:
        probabilities = self.action_probabilities()
        arm = int(self._rng.choice(self.n_arms, p=probabilities))
        self._last_probabilities = probabilities
        self._last_selected_arm = arm
        return arm

    def update(self, arm: int, reward: float) -> None:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(f"arm index {arm} is out of range.")
        if reward < 0.0 or reward > 1.0:
            raise ValueError("reward must lie in [0, 1].")
        if self._last_probabilities is None or self._last_selected_arm is None:
            raise RuntimeError("select_arm must be called before update.")
        if arm != self._last_selected_arm:
            raise ValueError("update arm must match the last selected arm.")

        probability = float(self._last_probabilities[arm])
        estimated_reward = float(reward) / probability
        self.log_weights[arm] += self.gamma * estimated_reward / self.n_arms

        self._last_probabilities = None
        self._last_selected_arm = None
