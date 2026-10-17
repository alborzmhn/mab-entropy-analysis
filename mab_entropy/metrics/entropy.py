from __future__ import annotations

import numpy as np


# === File-local configuration ===
# Numerical tolerance for tiny negative probability values from floating point.
NEGATIVE_PROBABILITY_TOLERANCE = 1e-12
# Numerical tolerance used when checking that probabilities sum to one.
PROBABILITY_SUM_TOLERANCE = 1e-8
# Numerical tolerance used when comparing entropy to its half-life threshold.
ENTROPY_THRESHOLD_TOLERANCE = 1e-12


def validate_probability_vector(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional array.")
    if probabilities.size == 0:
        raise ValueError("probabilities cannot be empty.")
    if np.any(probabilities < -NEGATIVE_PROBABILITY_TOLERANCE):
        raise ValueError("probabilities cannot contain negative values.")

    probabilities = np.clip(probabilities, 0.0, None)
    total = float(probabilities.sum())

    if not np.isclose(total, 1.0, atol=PROBABILITY_SUM_TOLERANCE):
        raise ValueError(
            f"probabilities must sum to 1; received sum={total}."
        )

    return probabilities / total


def shannon_entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy in nats, using the natural logarithm."""
    probabilities = validate_probability_vector(probabilities)
    positive = probabilities > 0.0
    p = probabilities[positive]
    return float(-np.sum(p * np.log(p)))


def normalized_shannon_entropy(probabilities: np.ndarray) -> float:
    """Entropy divided by log(K), so the result lies in [0, 1] for K>1."""
    probabilities = validate_probability_vector(probabilities)
    k = probabilities.size

    if k == 1:
        return 0.0

    return shannon_entropy(probabilities) / float(np.log(k))


def empirical_action_distribution(
    selected_arms: np.ndarray,
    *,
    n_arms: int,
) -> np.ndarray:
    selected_arms = np.asarray(selected_arms, dtype=int)

    if selected_arms.ndim != 1:
        raise ValueError("selected_arms must be one-dimensional.")
    if selected_arms.size == 0:
        raise ValueError("selected_arms cannot be empty.")
    if np.any(selected_arms < 0) or np.any(selected_arms >= n_arms):
        raise ValueError("selected_arms contains an out-of-range arm index.")

    counts = np.bincount(selected_arms, minlength=n_arms)
    return counts.astype(float) / selected_arms.size


def entropy_half_life(
    entropy_curve: np.ndarray,
    *,
    fraction: float,
) -> int | None:
    """Return the first 1-based timestep at or below fraction * H_1."""
    entropy_curve = np.asarray(entropy_curve, dtype=float)

    if entropy_curve.ndim != 1 or entropy_curve.size == 0:
        raise ValueError("entropy_curve must be a non-empty 1D array.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")

    threshold = fraction * float(entropy_curve[0])
    indices = np.flatnonzero(
        entropy_curve <= threshold + ENTROPY_THRESHOLD_TOLERANCE
    )

    if indices.size == 0:
        return None

    return int(indices[0] + 1)
