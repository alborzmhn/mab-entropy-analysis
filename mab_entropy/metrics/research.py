from __future__ import annotations

import numpy as np

from .belief import kl_divergence
from .entropy import validate_probability_vector


# === File-local configuration ===
NUMERICAL_EPSILON = 1e-12


def entropy_reduction_curve(
    entropy_curve: np.ndarray,
    *,
    reference_index: int = 0,
) -> np.ndarray:
    """Return entropy reduction relative to a chosen reference time step.

    The default reproduces the original definition, Delta H_t = H_1 - H_t.
    A later reference is useful for algorithms such as UCB1 whose first K
    actions are deterministic forced initialization rather than genuine UCB
    decisions. Values before the reference are returned as NaN so they are not
    accidentally interpreted as part of the reduction phase.
    """
    entropy_curve = np.asarray(entropy_curve, dtype=float)
    if entropy_curve.ndim != 1 or entropy_curve.size == 0:
        raise ValueError("entropy_curve must be a non-empty 1D array.")
    if reference_index < 0 or reference_index >= entropy_curve.size:
        raise ValueError("reference_index must index entropy_curve.")

    result = np.full_like(entropy_curve, np.nan, dtype=float)
    result[reference_index:] = (
        float(entropy_curve[reference_index])
        - entropy_curve[reference_index:]
    )
    return result


def entropy_dissipation_rate(entropy_curve: np.ndarray) -> np.ndarray:
    """Discrete entropy dissipation H_{t-1} - H_t, with zero at t=1.

    Positive values mean entropy decreased at the current step. Negative values
    indicate entropy reheating/increase relative to the previous step.
    """
    entropy_curve = np.asarray(entropy_curve, dtype=float)
    if entropy_curve.ndim != 1 or entropy_curve.size == 0:
        raise ValueError("entropy_curve must be a non-empty 1D array.")

    result = np.zeros_like(entropy_curve, dtype=float)
    if entropy_curve.size > 1:
        result[1:] = entropy_curve[:-1] - entropy_curve[1:]
    return result


def regret_per_entropy_reduction(
    cumulative_regret: np.ndarray,
    entropy_reduction: np.ndarray,
    *,
    minimum_reduction: float = NUMERICAL_EPSILON,
) -> np.ndarray:
    """Operationalize the proposal's regret/entropy-reduction coefficient.

    Values are NaN while entropy reduction is too small to form a stable ratio.
    Under the pseudo-regret interpretation, LOWER values are preferable because
    they mean less regret was accumulated per nat of action-entropy reduction.
    """
    cumulative_regret = np.asarray(cumulative_regret, dtype=float)
    entropy_reduction = np.asarray(entropy_reduction, dtype=float)

    if cumulative_regret.shape != entropy_reduction.shape:
        raise ValueError("cumulative_regret and entropy_reduction must match.")
    if cumulative_regret.ndim != 1 or cumulative_regret.size == 0:
        raise ValueError("inputs must be non-empty 1D arrays.")
    if minimum_reduction <= 0.0:
        raise ValueError("minimum_reduction must be positive.")

    result = np.full_like(cumulative_regret, np.nan, dtype=float)
    valid = entropy_reduction > minimum_reduction
    result[valid] = cumulative_regret[valid] / entropy_reduction[valid]
    return result


def binary_entropy(probability: np.ndarray | float) -> np.ndarray:
    """Binary entropy h(p) in nats, evaluated elementwise."""
    p = np.asarray(probability, dtype=float)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probability values must lie in [0, 1].")

    result = np.zeros_like(p, dtype=float)
    interior = (p > 0.0) & (p < 1.0)
    q = p[interior]
    result[interior] = -q * np.log(q) - (1.0 - q) * np.log(1.0 - q)
    return result


def proposal_entropy_from_misidentification(
    misidentification_probability: np.ndarray,
    *,
    n_arms: int,
) -> np.ndarray:
    """Compute h(p_t) + p_t log(K-1) from the proposal.

    Equality with action entropy is exact when the total suboptimal-action mass
    p_t is spread uniformly over the K-1 suboptimal arms. In general this is a
    useful proposal-motivated reference curve rather than an identity.
    """
    p = np.asarray(misidentification_probability, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(
            "misidentification_probability must be a non-empty 1D array."
        )
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("misidentification probabilities must lie in [0, 1].")
    if n_arms <= 1:
        return np.zeros_like(p, dtype=float)

    return binary_entropy(p) + p * float(np.log(n_arms - 1))


def asymptotic_entropy_from_misidentification(
    misidentification_probability: np.ndarray,
) -> np.ndarray:
    """Compute p_t log(1/p_t), the proposal's leading asymptotic signature."""
    p = np.asarray(misidentification_probability, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(
            "misidentification_probability must be a non-empty 1D array."
        )
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("misidentification probabilities must lie in [0, 1].")

    result = np.zeros_like(p, dtype=float)
    positive = p > 0.0
    result[positive] = p[positive] * np.log(1.0 / p[positive])
    return result


def regularize_probability_vector(
    probabilities: np.ndarray,
    *,
    floor: float,
) -> np.ndarray:
    """Add a small probability floor and renormalize for finite KL diagnostics."""
    probabilities = validate_probability_vector(probabilities)
    if floor <= 0.0:
        raise ValueError("floor must be positive.")
    if floor * probabilities.size >= 1.0:
        raise ValueError("floor is too large for the number of arms.")

    regularized = np.maximum(probabilities, floor)
    return regularized / regularized.sum()


def policy_kl_step_divergences(
    policy_probabilities: np.ndarray,
    *,
    probability_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact and smoothed D_KL(pi_t || pi_{t-1}) step trajectories.

    Input shape is (T, K). The first value is zero because no previous policy
    exists. Exact KL may be infinity when support changes. The smoothed version
    applies a probability floor so it remains finite and plot-friendly.
    """
    policies = np.asarray(policy_probabilities, dtype=float)
    if policies.ndim != 2 or policies.shape[0] == 0 or policies.shape[1] == 0:
        raise ValueError("policy_probabilities must have shape (T, K).")

    exact = np.zeros(policies.shape[0], dtype=float)
    smoothed = np.zeros(policies.shape[0], dtype=float)

    for t in range(1, policies.shape[0]):
        current = validate_probability_vector(policies[t])
        previous = validate_probability_vector(policies[t - 1])

        exact[t] = kl_divergence(current, previous)
        current_s = regularize_probability_vector(
            current,
            floor=probability_floor,
        )
        previous_s = regularize_probability_vector(
            previous,
            floor=probability_floor,
        )
        smoothed[t] = kl_divergence(current_s, previous_s)

    return exact, smoothed


def pareto_efficient_mask(
    regret: np.ndarray,
    entropy_reduction: np.ndarray,
) -> np.ndarray:
    """Return non-dominated points for min regret / max entropy reduction.

    Point i is dominated if another point has no greater regret and no smaller
    entropy reduction, with at least one strict improvement.
    """
    regret = np.asarray(regret, dtype=float)
    entropy_reduction = np.asarray(entropy_reduction, dtype=float)

    if regret.shape != entropy_reduction.shape:
        raise ValueError("regret and entropy_reduction must have matching shape.")
    if regret.ndim != 1:
        raise ValueError("regret and entropy_reduction must be 1D arrays.")
    if np.any(~np.isfinite(regret)) or np.any(~np.isfinite(entropy_reduction)):
        raise ValueError("Pareto inputs must contain only finite values.")

    n = regret.size
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        dominates_i = (
            (regret <= regret[i])
            & (entropy_reduction >= entropy_reduction[i])
            & ((regret < regret[i]) | (entropy_reduction > entropy_reduction[i]))
        )
        if np.any(dominates_i):
            efficient[i] = False

    return efficient
