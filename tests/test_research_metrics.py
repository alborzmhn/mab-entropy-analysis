import numpy as np

from mab_entropy.metrics.research import (
    asymptotic_entropy_from_misidentification,
    entropy_dissipation_rate,
    entropy_reduction_curve,
    pareto_efficient_mask,
    policy_kl_step_divergences,
    proposal_entropy_from_misidentification,
    regret_per_entropy_reduction,
)


def test_entropy_reduction_and_dissipation() -> None:
    entropy = np.array([1.0, 0.8, 0.5, 0.6])

    assert np.allclose(
        entropy_reduction_curve(entropy),
        [0.0, 0.2, 0.5, 0.4],
    )
    assert np.allclose(
        entropy_dissipation_rate(entropy),
        [0.0, 0.2, 0.3, -0.1],
    )



def test_entropy_reduction_can_use_post_warmup_reference() -> None:
    entropy = np.array([0.0, 0.0, 1.6, 1.2, 0.8])
    reduction = entropy_reduction_curve(entropy, reference_index=2)

    assert np.isnan(reduction[0])
    assert np.isnan(reduction[1])
    assert np.allclose(reduction[2:], [0.0, 0.4, 0.8])

def test_regret_per_entropy_reduction_is_nan_before_reduction() -> None:
    regret = np.array([0.2, 0.4, 0.6])
    reduction = np.array([0.0, 0.2, 0.3])
    ratio = regret_per_entropy_reduction(regret, reduction)

    assert np.isnan(ratio[0])
    assert np.allclose(ratio[1:], [2.0, 2.0])


def test_proposal_entropy_formula_for_two_arms_reduces_to_binary_entropy() -> None:
    p = np.array([0.1, 0.25, 0.5])
    predicted = proposal_entropy_from_misidentification(p, n_arms=2)
    expected = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    assert np.allclose(predicted, expected)


def test_asymptotic_entropy_handles_zero() -> None:
    p = np.array([0.0, 0.1])
    values = asymptotic_entropy_from_misidentification(p)
    assert values[0] == 0.0
    assert np.isclose(values[1], 0.1 * np.log(10.0))


def test_policy_kl_reports_infinite_exact_but_finite_smoothed_support_change() -> None:
    policies = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    exact, smoothed = policy_kl_step_divergences(
        policies,
        probability_floor=1e-9,
    )

    assert exact[0] == 0.0
    assert np.isinf(exact[1])
    assert smoothed[0] == 0.0
    assert np.isfinite(smoothed[1])
    assert smoothed[1] > 0.0


def test_pareto_frontier_min_regret_max_entropy_reduction() -> None:
    regret = np.array([10.0, 8.0, 12.0, 7.0])
    reduction = np.array([0.5, 0.5, 0.9, 0.3])
    mask = pareto_efficient_mask(regret, reduction)

    # Point 0 is dominated by point 1. The other three trade off regret and
    # entropy reduction and are therefore non-dominated.
    assert np.array_equal(mask, [False, True, True, True])
