def instantaneous_pseudo_regret(
    optimal_mean: float,
    selected_arm_mean: float,
) -> float:
    """Return mu_star - mu_selected_arm."""
    regret = float(optimal_mean) - float(selected_arm_mean)

    if regret < -1e-12:
        raise ValueError(
            "selected_arm_mean cannot exceed optimal_mean."
        )

    return max(0.0, regret)
