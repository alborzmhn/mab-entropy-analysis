from __future__ import annotations

import numpy as np

from .base import BanditAlgorithm


# === File-local configuration ===
# Number of independent RNG streams: one for actions and one for policy probing.
RNG_STREAM_COUNT = 2


class ThompsonSampling(BanditAlgorithm):
    """Beta-Bernoulli Thompson Sampling with a separate policy-probe RNG."""

    def __init__(
        self,
        n_arms: int,
        *,
        alpha_prior: float,
        beta_prior: float,
        policy_probability_samples: int,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_arms=n_arms)

        if alpha_prior <= 0.0 or beta_prior <= 0.0:
            raise ValueError("Beta prior parameters must be positive.")
        if policy_probability_samples <= 0:
            raise ValueError("policy_probability_samples must be positive.")

        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)
        self.policy_probability_samples = int(policy_probability_samples)
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self.alpha = np.full(self.n_arms, self.alpha_prior, dtype=float)
        self.beta = np.full(self.n_arms, self.beta_prior, dtype=float)

        seed_sequence = np.random.SeedSequence(self._seed)
        action_sequence, probe_sequence = seed_sequence.spawn(RNG_STREAM_COUNT)
        self._rng = np.random.default_rng(action_sequence)
        self._policy_rng = np.random.default_rng(probe_sequence)

    def action_probabilities(self) -> np.ndarray:
        samples = self._policy_rng.beta(
            self.alpha,
            self.beta,
            size=(self.policy_probability_samples, self.n_arms),
        )
        winners = np.argmax(samples, axis=1)
        counts = np.bincount(winners, minlength=self.n_arms)
        return counts.astype(float) / self.policy_probability_samples

    def select_arm(self) -> int:
        samples = self._rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(f"arm index {arm} is out of range.")
        if reward not in (0, 1, 0.0, 1.0):
            raise ValueError(
                "Beta-Bernoulli Thompson Sampling requires binary rewards."
            )

        self.alpha[arm] += float(reward)
        self.beta[arm] += 1.0 - float(reward)
