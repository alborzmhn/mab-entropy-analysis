from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# === File-local configuration ===
# Default figure size for per-algorithm regret plots.
REGRET_FIGSIZE = (9, 5)
# Default figure size for cross-algorithm comparison plots.
COMPARISON_FIGSIZE = (10, 6)
# Resolution used when saving PNG figures.
PLOT_DPI = 160
# Transparency of the standard-deviation band.
STD_BAND_ALPHA = 0.20
# Transparency of plot grid lines.
GRID_ALPHA = 0.25


def save_regret_plot(
    *,
    timesteps: np.ndarray,
    mean_cumulative_regret: np.ndarray,
    std_cumulative_regret: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lower = mean_cumulative_regret - std_cumulative_regret
    upper = mean_cumulative_regret + std_cumulative_regret

    fig, ax = plt.subplots(figsize=REGRET_FIGSIZE)
    ax.plot(
        timesteps,
        mean_cumulative_regret,
        label="Mean cumulative pseudo-regret",
    )
    ax.fill_between(
        timesteps,
        lower,
        upper,
        alpha=STD_BAND_ALPHA,
        label="±1 std",
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative pseudo-regret")
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_comparison_regret_plot(
    *,
    timesteps: np.ndarray,
    mean_curves: Mapping[str, np.ndarray],
    output_path: str | Path,
    title: str = "Mean Cumulative Pseudo-Regret Comparison",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    for label, values in mean_curves.items():
        ax.plot(timesteps, values, label=label)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean cumulative pseudo-regret")
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)
