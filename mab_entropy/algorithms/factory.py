from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import BanditAlgorithm
from .epsilon_greedy import EpsilonGreedy
from .exp3 import EXP3
from .greedy import Greedy
from .thompson_sampling import ThompsonSampling
from .ucb1 import UCB1


# === File-local configuration ===
# Canonical algorithm names accepted in config.yaml.
GREEDY_NAME = "greedy"
EPSILON_GREEDY_NAME = "epsilon_greedy"
UCB1_NAME = "ucb1"
THOMPSON_SAMPLING_NAME = "thompson_sampling"
EXP3_NAME = "exp3"


def _require(config: Mapping[str, Any], key: str, algorithm_name: str) -> Any:
    """Return a required experiment parameter with a clear error message."""
    if key not in config:
        raise KeyError(
            f"Missing required config parameter '{key}' for algorithm "
            f"'{algorithm_name}'. Define it in config.yaml."
        )
    return config[key]


def build_algorithm(
    config: Mapping[str, Any],
    *,
    n_arms: int,
    seed: int | None,
) -> BanditAlgorithm:
    name = str(config["name"]).lower()

    if name == GREEDY_NAME:
        return Greedy(n_arms=n_arms, seed=seed)

    if name == EPSILON_GREEDY_NAME:
        return EpsilonGreedy(
            n_arms=n_arms,
            epsilon=float(_require(config, "epsilon", name)),
            schedule=str(_require(config, "schedule", name)),
            min_epsilon=float(_require(config, "min_epsilon", name)),
            seed=seed,
        )

    if name == UCB1_NAME:
        return UCB1(
            n_arms=n_arms,
            exploration_coefficient=float(
                _require(config, "exploration_coefficient", name)
            ),
            seed=seed,
        )

    if name == THOMPSON_SAMPLING_NAME:
        return ThompsonSampling(
            n_arms=n_arms,
            alpha_prior=float(_require(config, "alpha_prior", name)),
            beta_prior=float(_require(config, "beta_prior", name)),
            policy_probability_samples=int(
                _require(config, "policy_probability_samples", name)
            ),
            seed=seed,
        )

    if name == EXP3_NAME:
        return EXP3(
            n_arms=n_arms,
            gamma=float(_require(config, "gamma", name)),
            seed=seed,
        )

    raise ValueError(f"Unknown algorithm name: {name}")
