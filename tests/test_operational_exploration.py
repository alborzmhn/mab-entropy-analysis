import numpy as np

from mab_entropy.algorithms.exp3 import EXP3
from mab_entropy.algorithms.greedy import Greedy
from mab_entropy.algorithms.thompson_sampling import ThompsonSampling
from mab_entropy.experiments.runner import (
    _operational_exploration_indicator,
    _operational_reference_values,
)


def test_greedy_action_on_current_maximizer_is_exploitation() -> None:
    algorithm = Greedy(n_arms=3, seed=0)
    algorithm.counts[:] = [1, 1, 1]
    algorithm.reward_sums[:] = [0.0, 1.0, 0.0]
    reference = _operational_reference_values(algorithm)

    assert _operational_exploration_indicator(
        algorithm=algorithm,
        selected_arm=1,
        reference_values=reference,
    ) == 0.0


def test_thompson_reference_uses_beta_posterior_means() -> None:
    algorithm = ThompsonSampling(
        n_arms=2,
        alpha_prior=1.0,
        beta_prior=1.0,
        policy_probability_samples=32,
        seed=0,
    )
    algorithm.alpha[:] = [3.0, 1.0]
    algorithm.beta[:] = [1.0, 3.0]

    reference = _operational_reference_values(algorithm)
    assert np.allclose(reference, [0.75, 0.25])


def test_exp3_is_intentionally_unsupported() -> None:
    algorithm = EXP3(n_arms=3, gamma=0.1, seed=0)
    assert _operational_reference_values(algorithm) is None
