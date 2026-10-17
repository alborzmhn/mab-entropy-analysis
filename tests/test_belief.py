import numpy as np

from mab_entropy.metrics.belief import (
    jensen_shannon_divergence,
    normalized_jensen_shannon_divergence,
    posterior_correctness_probability,
)
from mab_entropy.metrics.entropy import normalized_shannon_entropy
from mab_entropy.observers.beta_bernoulli import BetaBernoulliObserver


def test_js_divergence_is_zero_for_identical_distributions() -> None:
    p = np.array([0.2, 0.3, 0.5])
    assert np.isclose(jensen_shannon_divergence(p, p), 0.0)
    assert np.isclose(normalized_jensen_shannon_divergence(p, p), 0.0)


def test_normalized_js_is_one_for_disjoint_point_masses() -> None:
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert np.isclose(normalized_jensen_shannon_divergence(p, q), 1.0)


def test_posterior_correctness_reads_true_arm_mass() -> None:
    q = np.array([0.1, 0.7, 0.2])
    assert np.isclose(
        posterior_correctness_probability(q, true_optimal_arm=1),
        0.7,
    )


def test_common_observer_prior_is_approximately_uniform_over_best_arm() -> None:
    observer = BetaBernoulliObserver(
        n_arms=4,
        alpha_prior=1.0,
        beta_prior=1.0,
        optimal_arm_probability_samples=20_000,
        seed=123,
    )
    q = observer.optimal_arm_probabilities()

    assert np.allclose(q, np.full(4, 0.25), atol=0.02)
    assert normalized_shannon_entropy(q) > 0.99


def test_repeated_successes_raise_belief_in_that_arm() -> None:
    observer = BetaBernoulliObserver(
        n_arms=2,
        alpha_prior=1.0,
        beta_prior=1.0,
        optimal_arm_probability_samples=20_000,
        seed=123,
    )

    prior_q = observer.optimal_arm_probabilities()
    for _ in range(12):
        observer.update(0, 1)
    posterior_q = observer.optimal_arm_probabilities()

    assert posterior_q[0] > prior_q[0]
    assert posterior_q[0] > 0.9
