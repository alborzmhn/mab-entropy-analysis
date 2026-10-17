from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# === File-local configuration ===
# Default size for per-algorithm belief plots.
BELIEF_FIGSIZE = (9, 5)
# Default size for cross-algorithm belief comparisons.
COMPARISON_FIGSIZE = (10, 6)
# Resolution used for PNG files.
PLOT_DPI = 160
# Transparency of plot grid lines.
GRID_ALPHA = 0.25


def save_belief_analysis_plot(
    *,
    timesteps: np.ndarray,
    normalized_belief_entropy: np.ndarray,
    mean_correctness: np.ndarray,
    belief_map_accuracy: np.ndarray,
    confident_but_wrong_probability: np.ndarray,
    belief_half_life: int | None,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot belief uncertainty/correctness and mark half-life commitment quality."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=BELIEF_FIGSIZE)
    ax.plot(
        timesteps,
        normalized_belief_entropy,
        label="Normalized belief entropy",
    )
    ax.plot(
        timesteps,
        mean_correctness,
        label="Mean posterior mass on true optimal arm",
    )
    ax.plot(
        timesteps,
        belief_map_accuracy,
        label="Posterior-MAP best-arm accuracy",
    )
    ax.plot(
        timesteps,
        confident_but_wrong_probability,
        label="Confident-but-wrong probability",
    )

    if belief_half_life is not None:
        idx = int(np.argmin(np.abs(timesteps - belief_half_life)))
        t_half = timesteps[idx]
        c_half = float(mean_correctness[idx])
        map_half = float(belief_map_accuracy[idx])
        wrong_half = float(confident_but_wrong_probability[idx])
        ax.axvline(
            t_half, linestyle="--", linewidth=1.5,
            label=f"Belief entropy half-life (t={int(t_half)})",
        )
        note = (
            f"At half-life:\n"
            f"posterior correctness = {c_half:.3f}\n"
            f"MAP accuracy = {map_half:.3f}\n"
            f"wrong-confidence = {wrong_half:.3f}"
        )
        ax.annotate(
            note, xy=(t_half, 0.5), xycoords="data",
            xytext=(0.60, 0.32), textcoords="axes fraction",
            bbox=dict(boxstyle="round", alpha=0.85),
            arrowprops=dict(arrowstyle="->"),
        )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized entropy / probability (0–1)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_policy_belief_comparison_plot(
    *,
    timesteps: np.ndarray,
    normalized_conditional_policy_entropy: np.ndarray,
    normalized_belief_entropy: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Compare action randomization with uncertainty about the optimal arm."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=BELIEF_FIGSIZE)
    ax.plot(
        timesteps,
        normalized_conditional_policy_entropy,
        label="Normalized conditional policy entropy",
    )
    ax.plot(
        timesteps,
        normalized_belief_entropy,
        label="Normalized best-arm belief entropy",
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized entropy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_policy_belief_mismatch_plot(
    *,
    timesteps: np.ndarray,
    normalized_js_divergence: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot Jensen-Shannon mismatch between policy and best-arm belief."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=BELIEF_FIGSIZE)
    ax.plot(
        timesteps,
        normalized_js_divergence,
        label="Normalized policy-belief JS divergence",
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized Jensen-Shannon divergence")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_belief_regret_phase_plot(
    *,
    normalized_belief_entropy: np.ndarray,
    mean_cumulative_regret: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot cumulative regret against normalized best-arm belief entropy."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=BELIEF_FIGSIZE)
    ax.plot(normalized_belief_entropy, mean_cumulative_regret)
    ax.set_xlabel("Normalized best-arm belief entropy")
    ax.set_ylabel("Mean cumulative pseudo-regret")
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def _save_comparison_plot(
    *,
    timesteps: np.ndarray,
    curves: dict[str, np.ndarray],
    output_path: str | Path,
    title: str,
    ylabel: str,
    unit_interval: bool,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    for label, values in curves.items():
        ax.plot(timesteps, values, label=label)

    ax.set_xlabel("Time step")
    ax.set_ylabel(ylabel)
    if unit_interval:
        ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_belief_entropy_comparison_plot(
    *,
    timesteps: np.ndarray,
    curves: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    _save_comparison_plot(
        timesteps=timesteps,
        curves=curves,
        output_path=output_path,
        title="Best-Arm Belief Entropy Comparison",
        ylabel="Normalized mean belief entropy",
        unit_interval=True,
    )


def save_correctness_comparison_plot(
    *,
    timesteps: np.ndarray,
    curves: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    _save_comparison_plot(
        timesteps=timesteps,
        curves=curves,
        output_path=output_path,
        title="Posterior Correctness Comparison",
        ylabel="Mean posterior mass on true optimal arm",
        unit_interval=True,
    )


def save_policy_belief_mismatch_comparison_plot(
    *,
    timesteps: np.ndarray,
    curves: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    _save_comparison_plot(
        timesteps=timesteps,
        curves=curves,
        output_path=output_path,
        title="Policy-Belief Mismatch Comparison",
        ylabel="Normalized Jensen-Shannon divergence",
        unit_interval=True,
    )


def save_policy_analysis_plot(
    *,
    timesteps: np.ndarray,
    normalized_policy_entropy: np.ndarray,
    mean_optimal_action_probability: np.ndarray,
    policy_map_accuracy: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot policy concentration together with optimal-arm correctness."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=BELIEF_FIGSIZE)
    ax.plot(
        timesteps,
        normalized_policy_entropy,
        label="Normalized conditional policy entropy",
    )
    ax.plot(
        timesteps,
        mean_optimal_action_probability,
        label="Mean policy mass on true optimal arm",
    )
    ax.plot(
        timesteps,
        policy_map_accuracy,
        label="Policy-MAP optimal-arm accuracy",
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized entropy / probability (0–1)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_marginal_action_analysis_plot(
    *,
    timesteps: np.ndarray,
    normalized_marginal_entropy: np.ndarray,
    marginal_optimal_action_probability: np.ndarray,
    empirical_optimal_action_probability: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot across-run action concentration and optimal-arm correctness."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=BELIEF_FIGSIZE)
    ax.plot(
        timesteps,
        normalized_marginal_entropy,
        label="Normalized marginal action entropy",
    )
    ax.plot(
        timesteps,
        marginal_optimal_action_probability,
        label="Across-run probability of true optimal arm",
    )
    ax.plot(
        timesteps,
        empirical_optimal_action_probability,
        label="Empirical optimal-action rate",
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized entropy / probability (0–1)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_half_life_commitment_quality_plot(
    *,
    timesteps: np.ndarray,
    normalized_belief_entropy: np.ndarray,
    mean_correctness: np.ndarray,
    confident_but_wrong_probability: np.ndarray,
    belief_half_life: int | None,
    output_path: str | Path,
    title: str,
) -> None:
    """Pair belief half-life with correctness and wrong-confidence over time."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=BELIEF_FIGSIZE)
    ax.plot(timesteps, normalized_belief_entropy, label="Normalized belief entropy")
    ax.plot(timesteps, mean_correctness, label="Mean posterior correctness")
    ax.plot(
        timesteps,
        confident_but_wrong_probability,
        label="Confident-but-wrong probability",
    )

    if belief_half_life is not None:
        ax.axvline(
            belief_half_life,
            linestyle="--",
            linewidth=1.5,
            label=f"Belief entropy half-life (t={belief_half_life})",
        )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized entropy / probability (0–1)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_entropy_half_life_comparison_plot(
    *,
    half_lives: dict[str, tuple[int | None, int | None, int | None]],
    output_path: str | Path,
) -> None:
    """Compare marginal, conditional-policy, and belief half-lives by algorithm."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(half_lives)
    x = np.arange(len(labels), dtype=float)
    width = 0.25

    marginal = np.array([
        np.nan if half_lives[label][0] is None else half_lives[label][0]
        for label in labels
    ], dtype=float)
    policy = np.array([
        np.nan if half_lives[label][1] is None else half_lives[label][1]
        for label in labels
    ], dtype=float)
    belief = np.array([
        np.nan if half_lives[label][2] is None else half_lives[label][2]
        for label in labels
    ], dtype=float)

    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    ax.bar(x - width, marginal, width, label="Marginal action entropy")
    ax.bar(x, policy, width, label="Conditional policy entropy")
    ax.bar(x + width, belief, width, label="Belief entropy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Half-life time step")
    ax.set_title("Entropy Half-Life Comparison")
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_belief_half_life_quality_comparison_plot(
    *,
    points: dict[str, tuple[int | None, float, float]],
    output_path: str | Path,
) -> None:
    """Compare correctness and wrong-confidence at each algorithm's belief half-life."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    plotted = False
    for label, (half_life, correctness, wrong_probability) in points.items():
        if half_life is None or not np.isfinite(correctness):
            continue
        size = 60.0 + 900.0 * max(0.0, float(wrong_probability))
        ax.scatter(float(half_life), float(correctness), s=size, alpha=0.75)
        ax.annotate(label, (float(half_life), float(correctness)), xytext=(5, 5), textcoords="offset points")
        plotted = True

    ax.set_xlabel("Belief entropy half-life (time step)")
    ax.set_ylabel("Mean posterior correctness at half-life")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Belief Half-Life Commitment Quality")
    ax.grid(True, alpha=GRID_ALPHA)
    if plotted:
        ax.text(
            0.02,
            0.02,
            "Larger markers = higher confident-but-wrong probability at half-life",
            transform=ax.transAxes,
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)
