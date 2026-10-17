from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# === File-local configuration ===
# Default figure size for per-algorithm entropy plots.
ENTROPY_FIGSIZE = (9, 5)
# Default figure size for cross-algorithm comparison plots.
COMPARISON_FIGSIZE = (10, 6)
# Resolution used when saving PNG figures.
PLOT_DPI = 160
# Transparency of plot grid lines.
GRID_ALPHA = 0.25


def save_entropy_decomposition_plot(
    *,
    timesteps: np.ndarray,
    marginal_entropy: np.ndarray,
    mean_conditional_entropy: np.ndarray,
    history_action_mutual_information: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=ENTROPY_FIGSIZE)
    ax.plot(timesteps, marginal_entropy, label="Marginal action entropy")
    ax.plot(
        timesteps,
        mean_conditional_entropy,
        label="Mean conditional policy entropy",
    )
    ax.plot(
        timesteps,
        history_action_mutual_information,
        label="History-action mutual information",
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Entropy / mutual information (nats)")
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_normalized_entropy_plot(
    *,
    timesteps: np.ndarray,
    normalized_marginal_entropy: np.ndarray,
    mean_normalized_conditional_entropy: np.ndarray,
    normalized_history_action_mutual_information: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=ENTROPY_FIGSIZE)
    ax.plot(
        timesteps,
        normalized_marginal_entropy,
        label="Normalized marginal action entropy",
    )
    ax.plot(
        timesteps,
        mean_normalized_conditional_entropy,
        label="Normalized mean conditional policy entropy",
    )
    ax.plot(
        timesteps,
        normalized_history_action_mutual_information,
        label="Normalized history-action mutual information",
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized entropy / information (0–1)")
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_marginal_entropy_comparison_plot(
    *,
    timesteps: np.ndarray,
    curves: dict[str, np.ndarray],
    output_path: str | Path,
    title: str = "Normalized Marginal Action Entropy Comparison",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    for label, values in curves.items():
        ax.plot(timesteps, values, label=label)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized marginal action entropy")
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)
