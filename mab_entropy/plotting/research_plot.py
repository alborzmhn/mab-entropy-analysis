from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# === File-local configuration ===
RESEARCH_FIGSIZE = (9, 6)
COMPARISON_FIGSIZE = (10, 6)
PLOT_DPI = 160
GRID_ALPHA = 0.25
STD_BAND_ALPHA = 0.20
MARKER_SIZE = 34


def _prepare_output(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_entropy_efficiency_plot(
    *,
    timesteps: np.ndarray,
    entropy_reduction: np.ndarray,
    regret_per_entropy_reduction: np.ndarray,
    minimum_entropy_reduction: float,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot proposal entropy reduction and regret cost per nat."""
    output_path = _prepare_output(output_path)

    fig, axes = plt.subplots(2, 1, figsize=RESEARCH_FIGSIZE, sharex=True)
    axes[0].plot(timesteps, entropy_reduction)
    axes[0].set_ylabel("Marginal entropy reduction, ΔH (nats)")
    axes[0].grid(True, alpha=GRID_ALPHA)

    axes[1].plot(timesteps, regret_per_entropy_reduction)
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Cumulative regret / ΔH")
    axes[1].grid(True, alpha=GRID_ALPHA)
    axes[1].text(
        0.99,
        0.97,
        f"Ratio omitted while ΔH ≤ {minimum_entropy_reduction:g} nats\n"
        "to avoid unstable near-zero denominators.",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "alpha": 0.15},
    )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_entropy_dissipation_plot(
    *,
    timesteps: np.ndarray,
    dissipation_rate: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot discrete H_{t-1}-H_t; negative values indicate reheating."""
    output_path = _prepare_output(output_path)

    fig, ax = plt.subplots(figsize=RESEARCH_FIGSIZE)
    ax.plot(timesteps, dissipation_rate, label="Entropy dissipation rate")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Time step")
    ax.set_ylabel("H(t-1) - H(t) (nats/step)")
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_action_entropy_regret_phase_plot(
    *,
    normalized_action_entropy: np.ndarray,
    mean_cumulative_regret: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot the proposal phase trajectory (H_t, R_t)."""
    output_path = _prepare_output(output_path)

    fig, ax = plt.subplots(figsize=RESEARCH_FIGSIZE)
    ax.plot(normalized_action_entropy, mean_cumulative_regret)
    ax.scatter(
        [normalized_action_entropy[0]],
        [mean_cumulative_regret[0]],
        s=MARKER_SIZE,
        label="Start (t=1)",
    )
    ax.scatter(
        [normalized_action_entropy[-1]],
        [mean_cumulative_regret[-1]],
        s=MARKER_SIZE,
        label="End (t=T)",
    )
    ax.set_xlabel("Normalized marginal action entropy")
    ax.set_ylabel("Mean cumulative pseudo-regret")
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_policy_geometry_plot(
    *,
    timesteps: np.ndarray,
    mean_smoothed_step_kl: np.ndarray,
    std_smoothed_step_kl: np.ndarray,
    cumulative_path_length: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot finite KL movement and cumulative information-geometric path."""
    output_path = _prepare_output(output_path)

    lower = np.maximum(0.0, mean_smoothed_step_kl - std_smoothed_step_kl)
    upper = mean_smoothed_step_kl + std_smoothed_step_kl

    fig, axes = plt.subplots(2, 1, figsize=RESEARCH_FIGSIZE, sharex=True)
    axes[0].plot(timesteps, mean_smoothed_step_kl, label="Mean smoothed KL step")
    axes[0].fill_between(
        timesteps,
        lower,
        upper,
        alpha=STD_BAND_ALPHA,
        label="±1 std across runs",
    )
    axes[0].set_ylabel("KL step (nats)")
    axes[0].grid(True, alpha=GRID_ALPHA)
    axes[0].legend()

    axes[1].plot(timesteps, cumulative_path_length, label="Cumulative KL path length")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Path length (nats)")
    axes[1].grid(True, alpha=GRID_ALPHA)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_entropy_misidentification_theory_plot(
    *,
    timesteps: np.ndarray,
    observed_action_entropy: np.ndarray,
    proposal_predicted_entropy: np.ndarray,
    asymptotic_predicted_entropy: np.ndarray,
    misidentification_probability: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Compare observed entropy with both proposal-motivated p_t curves."""
    output_path = _prepare_output(output_path)

    fig, axes = plt.subplots(2, 1, figsize=RESEARCH_FIGSIZE, sharex=True)
    axes[0].plot(timesteps, observed_action_entropy, label="Observed H(A_t)")
    axes[0].plot(
        timesteps,
        proposal_predicted_entropy,
        label="h(p_t) + p_t log(K-1)",
    )
    axes[0].plot(
        timesteps,
        asymptotic_predicted_entropy,
        label="p_t log(1/p_t)",
    )
    axes[0].set_ylabel("Entropy (nats)")
    axes[0].grid(True, alpha=GRID_ALPHA)
    axes[0].legend()

    axes[1].plot(timesteps, misidentification_probability, label="p_t")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("P(A_t != a*)")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=GRID_ALPHA)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_efficiency_comparison_plot(
    *,
    timesteps: np.ndarray,
    curves: dict[str, np.ndarray],
    minimum_entropy_reduction: float,
    output_path: str | Path,
) -> None:
    output_path = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    for label, curve in curves.items():
        ax.plot(timesteps, curve, label=label)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Regret / entropy reduction")
    ax.set_title("Marginal Entropy Reduction Efficiency Comparison (lower is better)")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.text(
        0.99,
        0.97,
        f"Ratio omitted while ΔH ≤ {minimum_entropy_reduction:g} nats",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "alpha": 0.15},
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_dissipation_comparison_plot(
    *,
    timesteps: np.ndarray,
    curves: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    output_path = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    for label, curve in curves.items():
        ax.plot(timesteps, curve, label=label)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized entropy dissipation / step")
    ax.set_title("Action-Entropy Dissipation Comparison")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_action_entropy_regret_phase_comparison_plot(
    *,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: str | Path,
) -> None:
    output_path = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    for label, (entropy_curve, regret_curve) in curves.items():
        ax.plot(entropy_curve, regret_curve, label=label)
    ax.set_xlabel("Normalized marginal action entropy")
    ax.set_ylabel("Mean cumulative pseudo-regret")
    ax.set_title("Action-Entropy / Regret Phase Diagram")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_policy_kl_path_comparison_plot(
    *,
    timesteps: np.ndarray,
    curves: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    output_path = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    for label, curve in curves.items():
        ax.plot(timesteps, curve, label=label)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean cumulative smoothed KL path length (nats)")
    ax.set_title("Information-Geometric Policy Path Comparison")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_pareto_frontier_plot(
    *,
    labels: list[str],
    regret: np.ndarray,
    entropy_reduction: np.ndarray,
    efficient_mask: np.ndarray,
    output_path: str | Path,
    title: str = "Final Entropy-Regret Pareto Frontier",
) -> None:
    """Plot final algorithm points for min regret / max entropy reduction."""
    output_path = _prepare_output(output_path)
    regret = np.asarray(regret, dtype=float)
    entropy_reduction = np.asarray(entropy_reduction, dtype=float)
    efficient_mask = np.asarray(efficient_mask, dtype=bool)

    fig, ax = plt.subplots(figsize=COMPARISON_FIGSIZE)
    ax.scatter(entropy_reduction, regret, s=MARKER_SIZE)

    for x, y, label in zip(entropy_reduction, regret, labels, strict=True):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")

    if np.any(efficient_mask):
        frontier_x = entropy_reduction[efficient_mask]
        frontier_y = regret[efficient_mask]
        order = np.argsort(frontier_x)
        ax.plot(
            frontier_x[order],
            frontier_y[order],
            marker="o",
            label="Pareto-efficient algorithms",
        )

    ax.set_xlabel("Final action-entropy reduction (nats; higher is better)")
    ax.set_ylabel("Final mean cumulative pseudo-regret (lower is better)")
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_operational_exploration_plot(
    *,
    timesteps: np.ndarray,
    exploration_rate: np.ndarray,
    exploitation_rate: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot operational exploration and exploitation for one algorithm."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps, exploration_rate, label="Operational exploration rate")
    ax.plot(timesteps, exploitation_rate, label="Operational exploitation rate")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_operational_exploration_comparison_plot(
    *,
    timesteps: np.ndarray,
    curves: dict[str, np.ndarray],
    output_path: str | Path,
    title: str = "Operational Exploration Rate Comparison",
) -> None:
    """Compare operational exploration across supported algorithm configs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, curve in curves.items():
        ax.plot(timesteps, curve, label=label)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Operational exploration rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_explicit_entropy_regret_phase_plot(
    *,
    entropy: np.ndarray,
    mean_cumulative_regret: np.ndarray,
    output_path: str | Path,
    title: str,
    xlabel: str,
) -> None:
    """Save an explicit entropy/regret phase curve for proposal section 1.4.1."""
    output_path = _prepare_output(output_path)

    fig, ax = plt.subplots(figsize=RESEARCH_FIGSIZE)
    ax.plot(entropy, mean_cumulative_regret)
    ax.scatter(
        [entropy[0]],
        [mean_cumulative_regret[0]],
        s=MARKER_SIZE,
        label="Start (t=1)",
    )
    ax.scatter(
        [entropy[-1]],
        [mean_cumulative_regret[-1]],
        s=MARKER_SIZE,
        label="End (t=T)",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean cumulative pseudo-regret")
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_policy_entropy_regret_phase_plot(
    *,
    normalized_policy_entropy: np.ndarray,
    mean_cumulative_regret: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot (H_t^pi, R_t): conditional policy entropy versus regret."""
    _save_explicit_entropy_regret_phase_plot(
        entropy=normalized_policy_entropy,
        mean_cumulative_regret=mean_cumulative_regret,
        output_path=output_path,
        title=title,
        xlabel="Normalized conditional policy entropy, Hπ / log K",
    )


def save_belief_entropy_regret_phase_plot(
    *,
    normalized_belief_entropy: np.ndarray,
    mean_cumulative_regret: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot (H_t^B, R_t): best-arm belief entropy versus regret."""
    _save_explicit_entropy_regret_phase_plot(
        entropy=normalized_belief_entropy,
        mean_cumulative_regret=mean_cumulative_regret,
        output_path=output_path,
        title=title,
        xlabel="Normalized best-arm belief entropy, HB / log K",
    )


def save_marginal_entropy_regret_phase_plot(
    *,
    normalized_marginal_entropy: np.ndarray,
    mean_cumulative_regret: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot (H_t^m, R_t): across-run marginal action entropy versus regret."""
    _save_explicit_entropy_regret_phase_plot(
        entropy=normalized_marginal_entropy,
        mean_cumulative_regret=mean_cumulative_regret,
        output_path=output_path,
        title=title,
        xlabel="Normalized marginal action entropy, Hm / log K",
    )
