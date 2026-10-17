from __future__ import annotations

import numpy as np


class BetaBernoulliObserver:
    """Common Bayesian evaluator for Bernoulli-bandit histories.

    The observer is external to the bandit algorithm. It receives exactly the
    same action/reward history, maintains an independent Beta posterior for
    each arm, and estimates the posterior probability that each arm is the
    optimal arm.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        alpha_prior: float,
        beta_prior: float,
        optimal_arm_probability_samples: int,
        seed: int | None = None,
    ) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be positive.")
        if alpha_prior <= 0.0 or beta_prior <= 0.0:
            raise ValueError("Beta prior parameters must be positive.")
        if optimal_arm_probability_samples <= 0:
            raise ValueError(
                "optimal_arm_probability_samples must be positive."
            )

        self.n_arms = int(n_arms)
        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)
        self.optimal_arm_probability_samples = int(
            optimal_arm_probability_samples
        )
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self.alpha = np.full(
            self.n_arms,
            self.alpha_prior,
            dtype=float,
        )
        self.beta = np.full(
            self.n_arms,
            self.beta_prior,
            dtype=float,
        )
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)
        self._rng = np.random.default_rng(self._seed)

    @property
    def posterior_means(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    @property
    def empirical_means(self) -> np.ndarray:
        means = np.full(self.n_arms, np.nan, dtype=float)
        observed = self.counts > 0
        means[observed] = (
            self.reward_sums[observed]
            / self.counts[observed]
        )
        return means

    def optimal_arm_probabilities(self) -> np.ndarray:
        """Estimate q_t(a)=P(A*=a | F_t) by posterior Monte Carlo."""
        samples = self._rng.beta(
            self.alpha,
            self.beta,
            size=(self.optimal_arm_probability_samples, self.n_arms),
        )
        winners = np.argmax(samples, axis=1)
        counts = np.bincount(winners, minlength=self.n_arms)
        return counts.astype(float) / self.optimal_arm_probability_samples

    def update(self, arm: int, reward: float) -> None:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(f"arm index {arm} is out of range.")
        if reward not in (0, 1, 0.0, 1.0):
            raise ValueError(
                "Beta-Bernoulli observer requires binary rewards."
            )

        reward = float(reward)
        self.counts[arm] += 1
        self.reward_sums[arm] += reward
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward
