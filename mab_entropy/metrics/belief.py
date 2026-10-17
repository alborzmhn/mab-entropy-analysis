from __future__ import annotations

import numpy as np

from .entropy import validate_probability_vector


# === File-local configuration ===
# Natural logarithms are used throughout, so the maximum JS divergence is ln(2).
MAX_JS_DIVERGENCE_NATS = float(np.log(2.0))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) in nats for discrete probability vectors."""
    p = validate_probability_vector(p)
    q = validate_probability_vector(q)

    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape.")

    positive = p > 0.0
    if np.any(q[positive] <= 0.0):
        return float("inf")

    return float(np.sum(p[positive] * np.log(p[positive] / q[positive])))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric Jensen-Shannon divergence in nats."""
    p = validate_probability_vector(p)
    q = validate_probability_vector(q)

    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape.")

    midpoint = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def normalized_jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """Jensen-Shannon divergence normalized to [0, 1]."""
    return jensen_shannon_divergence(p, q) / MAX_JS_DIVERGENCE_NATS


def posterior_correctness_probability(
    optimal_arm_probabilities: np.ndarray,
    *,
    true_optimal_arm: int,
) -> float:
    """Posterior mass assigned to the known true optimal arm."""
    probabilities = validate_probability_vector(optimal_arm_probabilities)
    if true_optimal_arm < 0 or true_optimal_arm >= probabilities.size:
        raise IndexError("true_optimal_arm is out of range.")
    return float(probabilities[true_optimal_arm])
