import numpy as np

from mab_entropy.algorithms.epsilon_greedy import EpsilonGreedy
from mab_entropy.algorithms.exp3 import EXP3
from mab_entropy.algorithms.greedy import Greedy
from mab_entropy.algorithms.thompson_sampling import ThompsonSampling
from mab_entropy.algorithms.ucb1 import UCB1


def assert_valid_probabilities(probabilities: np.ndarray) -> None:
    assert probabilities.shape[0] > 0
    assert np.all(probabilities >= 0.0)
    assert np.isclose(probabilities.sum(), 1.0)


def test_greedy_has_no_forced_initialization_and_is_seeded() -> None:
    a1 = Greedy(n_arms=4, seed=123)
    a2 = Greedy(n_arms=4, seed=123)
    assert a1.select_arm() == a2.select_arm()


def test_greedy_initial_policy_is_uniform_due_to_tie() -> None:
    algo = Greedy(n_arms=4, seed=123)
    assert np.allclose(algo.action_probabilities(), np.full(4, 0.25))


def test_greedy_becomes_deterministic_after_unique_positive_maximum() -> None:
    algo = Greedy(n_arms=3, seed=123)
    algo.update(1, 1.0)
    assert np.allclose(algo.action_probabilities(), [0.0, 1.0, 0.0])


def test_epsilon_greedy_policy_matches_standard_definition() -> None:
    algo = EpsilonGreedy(
        n_arms=5,
        epsilon=0.1,
        schedule="constant",
        min_epsilon=0.0,
        seed=1,
    )
    algo.update(2, 1.0)

    probabilities = algo.action_probabilities()
    expected = np.full(5, 0.02)
    expected[2] = 0.92
    assert np.allclose(probabilities, expected)


def test_decaying_epsilon_decreases() -> None:
    algo = EpsilonGreedy(
        n_arms=2,
        epsilon=1.0,
        schedule="inverse_sqrt",
        min_epsilon=0.01,
        seed=1,
    )

    first = algo.current_epsilon
    for _ in range(20):
        algo.update(1, 1.0)
    later = algo.current_epsilon

    assert later < first
    assert later >= 0.01


def test_ucb1_policy_is_deterministic_over_unseen_arms() -> None:
    algo = UCB1(n_arms=4, exploration_coefficient=2.0, seed=7)

    # Deterministic initialization always selects the first unseen arm.
    assert np.allclose(algo.action_probabilities(), [1.0, 0.0, 0.0, 0.0])

    algo.update(0, 0.0)
    assert np.allclose(algo.action_probabilities(), [0.0, 1.0, 0.0, 0.0])

    algo.update(1, 0.0)
    assert np.allclose(algo.action_probabilities(), [0.0, 0.0, 1.0, 0.0])


def test_ucb1_samples_every_unseen_arm_once() -> None:
    algo = UCB1(n_arms=4, exploration_coefficient=2.0, seed=7)
    selected = []
    for _ in range(4):
        arm = algo.select_arm()
        selected.append(arm)
        algo.update(arm, 0.0)

    assert sorted(selected) == [0, 1, 2, 3]


def test_thompson_updates_beta_posterior() -> None:
    algo = ThompsonSampling(
        n_arms=2,
        alpha_prior=1.0,
        beta_prior=1.0,
        policy_probability_samples=500,
        seed=123,
    )

    alpha_before = algo.alpha.copy()
    beta_before = algo.beta.copy()

    algo.update(0, 1)
    algo.update(1, 0)

    assert algo.alpha[0] == alpha_before[0] + 1
    assert algo.beta[1] == beta_before[1] + 1


def test_thompson_policy_probabilities_are_valid_and_probe_does_not_break_actions() -> None:
    algo = ThompsonSampling(
        n_arms=3,
        alpha_prior=1.0,
        beta_prior=1.0,
        policy_probability_samples=1000,
        seed=123,
    )
    probabilities = algo.action_probabilities()
    assert_valid_probabilities(probabilities)
    assert np.allclose(probabilities, np.full(3, 1 / 3), atol=0.06)

    arm = algo.select_arm()
    assert 0 <= arm < 3


def test_exp3_probabilities_are_valid() -> None:
    algo = EXP3(n_arms=4, gamma=0.1, seed=123)

    probs = algo.action_probabilities()
    assert_valid_probabilities(probs)

    arm = algo.select_arm()
    algo.update(arm, 1.0)

    probs_after = algo.action_probabilities()
    assert_valid_probabilities(probs_after)
