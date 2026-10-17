from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BanditAlgorithm(ABC):
    """Common interface for all bandit algorithms."""

    def __init__(self, n_arms: int) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be positive.")
        self.n_arms = int(n_arms)

    @abstractmethod
    def action_probabilities(self) -> np.ndarray:
        """Return P(A_t=a | current history) before selecting the next arm."""
        raise NotImplementedError

    @abstractmethod
    def select_arm(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
