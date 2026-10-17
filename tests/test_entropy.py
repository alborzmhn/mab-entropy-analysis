import numpy as np

from mab_entropy.metrics.entropy import (
    empirical_action_distribution,
    entropy_half_life,
    normalized_shannon_entropy,
    shannon_entropy,
)


def test_entropy_of_deterministic_distribution_is_zero() -> None:
    assert np.isclose(shannon_entropy(np.array([1.0, 0.0, 0.0])), 0.0)


def test_entropy_of_uniform_distribution_is_log_k() -> None:
    probabilities = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.isclose(shannon_entropy(probabilities), np.log(4))
    assert np.isclose(normalized_shannon_entropy(probabilities), 1.0)


def test_empirical_action_distribution() -> None:
    selected = np.array([0, 0, 1, 2, 2, 2])
    distribution = empirical_action_distribution(selected, n_arms=3)
    assert np.allclose(distribution, [2 / 6, 1 / 6, 3 / 6])


def test_entropy_half_life() -> None:
    curve = np.array([1.0, 0.8, 0.49, 0.4])
    assert entropy_half_life(curve, fraction=0.5) == 3


def test_entropy_half_life_none_when_not_reached() -> None:
    curve = np.array([1.0, 0.9, 0.7, 0.6])
    assert entropy_half_life(curve, fraction=0.5) is None
