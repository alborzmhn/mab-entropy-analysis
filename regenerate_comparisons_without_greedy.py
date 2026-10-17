from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

from mab_entropy.metrics.research import pareto_efficient_mask
from mab_entropy.plotting.belief_plot import (
    save_belief_entropy_comparison_plot,
    save_entropy_half_life_comparison_plot,
    save_belief_half_life_quality_comparison_plot,
    save_correctness_comparison_plot,
    save_policy_belief_mismatch_comparison_plot,
)
from mab_entropy.plotting.entropy_plot import save_marginal_entropy_comparison_plot
from mab_entropy.plotting.regret_plot import save_comparison_regret_plot
from mab_entropy.plotting.research_plot import (
    save_action_entropy_regret_phase_comparison_plot,
    save_dissipation_comparison_plot,
    save_efficiency_comparison_plot,
    save_pareto_frontier_plot,
    save_policy_kl_path_comparison_plot,
    save_operational_exploration_comparison_plot,
)

CONFIG_PATH = Path("config.yaml")

# Destination requested by the user.
WITHOUT_GREEDY_DIR_NAME = "without_greedy_plot"
WITH_GREEDY_DIR_NAME = "plots"

# Same filenames as main.py
REGRET_COMPARISON_PLOT_FILENAME = "comparison_cumulative_regret.png"
MARGINAL_ENTROPY_COMPARISON_PLOT_FILENAME = "comparison_marginal_entropy_normalized.png"
BELIEF_ENTROPY_COMPARISON_PLOT_FILENAME = "comparison_belief_entropy_normalized.png"
CORRECTNESS_COMPARISON_PLOT_FILENAME = "comparison_belief_correctness.png"
MISMATCH_COMPARISON_PLOT_FILENAME = "comparison_policy_belief_js_divergence.png"
ENTROPY_HALF_LIFE_COMPARISON_PLOT_FILENAME = "comparison_entropy_half_lives.png"
BELIEF_HALF_LIFE_QUALITY_COMPARISON_PLOT_FILENAME = "comparison_belief_half_life_commitment_quality.png"
EFFICIENCY_COMPARISON_PLOT_FILENAME = "comparison_marginal_entropy_reduction_efficiency.png"
DISSIPATION_COMPARISON_PLOT_FILENAME = "comparison_marginal_entropy_dissipation.png"
ACTION_ENTROPY_REGRET_PHASE_COMPARISON_PLOT_FILENAME = "comparison_phase_marginal_entropy_vs_regret.png"
POLICY_KL_PATH_COMPARISON_PLOT_FILENAME = "comparison_policy_kl_path.png"
OPERATIONAL_EXPLORATION_COMPARISON_PLOT_FILENAME = "comparison_operational_exploration.png"
PARETO_FRONTIER_PLOT_FILENAME = "pareto_entropy_reduction_vs_regret.png"
CUMULATIVE_MARGINAL_ENTROPY_PLOT_FILENAME = "comparison_cumulative_marginal_entropy.png"
CUMULATIVE_BELIEF_ENTROPY_PLOT_FILENAME = "comparison_cumulative_belief_entropy.png"
CUMULATIVE_POLICY_ENTROPY_PLOT_FILENAME = "comparison_cumulative_policy_entropy.png"
BELIEF_ENTROPY_REDUCTION_PLOT_FILENAME = "comparison_belief_entropy_reduction.png"
BELIEF_ENTROPY_DISSIPATION_PLOT_FILENAME = "comparison_belief_entropy_dissipation.png"
BELIEF_ENTROPY_EFFICIENCY_PLOT_FILENAME = "comparison_belief_entropy_reduction_efficiency.png"
BELIEF_PARETO_FRONTIER_PLOT_FILENAME = "pareto_belief_entropy_reduction_vs_regret.png"

SUMMARY_FILENAME = "summary.csv"
FINAL_COMPARISON_FILENAME = "final_algorithm_comparison.csv"


def slugify(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower())
    return slug.strip("_")


def load_csv(path: Path) -> dict[str, np.ndarray]:
    """Read a numeric CSV as a dict of numpy arrays."""
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    out: dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        values = []
        numeric = True
        for row in rows:
            raw = row[key]
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                numeric = False
                break
        if numeric:
            out[key] = np.asarray(values, dtype=float)
    return out


def pick(data: dict[str, np.ndarray], *names: str) -> np.ndarray:
    """Pick the first existing column name, with a useful error if none exists."""
    for name in names:
        if name in data:
            return data[name]
    raise KeyError(
        "None of these columns were found: "
        + ", ".join(names)
        + "\nAvailable columns: "
        + ", ".join(sorted(data))
    )


def read_final_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    x = float(value)
    if not np.isfinite(x):
        return None
    return int(round(x))



def save_cumulative_entropy_pair_plot(
    timesteps: np.ndarray,
    raw_curves: dict[str, np.ndarray],
    normalized_curves: dict[str, np.ndarray],
    output_path: Path,
    entropy_name: str,
) -> None:
    """Save raw and normalized cumulative entropy in one PNG (two panels)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for label, curve in raw_curves.items():
        axes[0].plot(timesteps, curve, label=label)
    axes[0].set_title(f"Cumulative {entropy_name} (nats)")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Cumulative entropy (nats)")
    axes[0].grid(True, alpha=0.25)

    for label, curve in normalized_curves.items():
        axes[1].plot(timesteps, curve, label=label)
    axes[1].set_title(f"Cumulative Normalized {entropy_name}")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Cumulative normalized entropy")
    axes[1].grid(True, alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, min(3, len(labels))), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{entropy_name} — Cumulative Entropy Comparison", y=1.08)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_entropy_change_pair_plot(
    timesteps: np.ndarray,
    raw_curves: dict[str, np.ndarray],
    normalized_curves: dict[str, np.ndarray],
    output_path: Path,
    entropy_name: str,
    quantity_name: str,
    raw_ylabel: str,
    normalized_ylabel: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for label, curve in raw_curves.items():
        axes[0].plot(timesteps, curve, label=label)
    axes[0].set_title(f"{entropy_name} {quantity_name} (nats)")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel(raw_ylabel)
    axes[0].grid(True, alpha=0.25)
    for label, curve in normalized_curves.items():
        axes[1].plot(timesteps, curve, label=label)
    axes[1].set_title(f"Normalized {entropy_name} {quantity_name}")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel(normalized_ylabel)
    axes[1].grid(True, alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, min(3, len(labels))), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{entropy_name} - {quantity_name} Comparison", y=1.08)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_generic_curve_comparison(
    timesteps: np.ndarray, curves: dict[str, np.ndarray], output_path: Path, title: str, ylabel: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, curve in curves.items():
        ax.plot(timesteps, curve, label=label)
    ax.set_xlabel("Time step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_belief_pareto_plot(
    labels: list[str], regret: np.ndarray, belief_entropy_reduction: np.ndarray,
    efficient_mask: np.ndarray, output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(belief_entropy_reduction, regret, s=40)
    for x, y, label in zip(belief_entropy_reduction, regret, labels, strict=True):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")
    if np.any(efficient_mask):
        x = belief_entropy_reduction[efficient_mask]
        y = regret[efficient_mask]
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker="o", label="Pareto-efficient algorithms")
    ax.set_xlabel("Final belief-entropy reduction (nats; higher is better)")
    ax.set_ylabel("Final mean cumulative pseudo-regret (lower is better)")
    ax.set_title("Final Belief-Entropy / Regret Pareto Frontier")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_comparisons(
    *,
    config: dict,
    included_configs: list[dict],
    algorithms_dir: Path,
    comparisons_data_dir: Path,
    destination: Path,
    minimum_entropy_reduction: float,
) -> None:
    """Regenerate comparison plots only from existing CSVs; never rerun algorithms."""

def _generate_once(*, exclude_greedy: bool, destination_name: str) -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_cfg = config["experiment"]
    research_cfg = config.get("research_metrics", {})

    output_dir = Path(exp_cfg["output_dir"]) / str(exp_cfg["experiment_name"])
    algorithms_dir = output_dir / "algorithms"
    comparisons_dir = output_dir / "comparisons"
    comparisons_data_dir = comparisons_dir / "data"
    destination = comparisons_dir / destination_name
    destination.mkdir(parents=True, exist_ok=True)

    minimum_entropy_reduction = float(
        research_cfg.get("entropy_efficiency_minimum_reduction", 1e-12)
    )

    # Important: exclude only the pure Greedy config, NOT epsilon-greedy.
    included_configs = [
        cfg for cfg in config["algorithms"]
        if (not exclude_greedy) or str(cfg.get("name", "")).strip().lower() != "greedy"
    ]

    if not included_configs:
        raise ValueError("No algorithm configurations found in config.yaml")

    regret_curves: dict[str, np.ndarray] = {}
    marginal_entropy_curves: dict[str, np.ndarray] = {}
    belief_entropy_curves: dict[str, np.ndarray] = {}
    correctness_curves: dict[str, np.ndarray] = {}
    mismatch_curves: dict[str, np.ndarray] = {}
    efficiency_curves: dict[str, np.ndarray] = {}
    dissipation_curves: dict[str, np.ndarray] = {}
    phase_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    policy_kl_path_curves: dict[str, np.ndarray] = {}
    operational_exploration_curves: dict[str, np.ndarray] = {}
    cumulative_marginal_raw_curves: dict[str, np.ndarray] = {}
    cumulative_marginal_normalized_curves: dict[str, np.ndarray] = {}
    cumulative_belief_raw_curves: dict[str, np.ndarray] = {}
    cumulative_belief_normalized_curves: dict[str, np.ndarray] = {}
    cumulative_policy_raw_curves: dict[str, np.ndarray] = {}
    cumulative_policy_normalized_curves: dict[str, np.ndarray] = {}
    belief_reduction_raw_curves: dict[str, np.ndarray] = {}
    belief_reduction_normalized_curves: dict[str, np.ndarray] = {}
    belief_dissipation_raw_curves: dict[str, np.ndarray] = {}
    belief_dissipation_normalized_curves: dict[str, np.ndarray] = {}
    belief_efficiency_curves: dict[str, np.ndarray] = {}
    final_belief_reduction_by_label: dict[str, float] = {}
    final_regret_by_label: dict[str, float] = {}
    common_timesteps: np.ndarray | None = None

    for cfg in included_configs:
        label = str(cfg.get("label", cfg["name"]))
        summary_path = algorithms_dir / slugify(label) / "data" / SUMMARY_FILENAME
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing existing result file for {label}: {summary_path}\n"
                "This script does not rerun experiments; it only reads old results."
            )

        d = load_csv(summary_path)

        timesteps = pick(d, "timestep", "time_step", "timesteps", "t")
        if common_timesteps is None:
            common_timesteps = timesteps
        elif len(timesteps) != len(common_timesteps) or not np.allclose(timesteps, common_timesteps):
            raise ValueError(f"Timesteps for {label} do not match the other algorithms.")

        regret = pick(d, "mean_cumulative_regret", "mean_cumulative_pseudo_regret")
        marginal_entropy = pick(d, "normalized_marginal_action_entropy")
        belief_entropy = pick(d, "mean_normalized_belief_entropy")
        raw_marginal_entropy = pick(d, "marginal_action_entropy")
        raw_belief_entropy = pick(d, "mean_belief_entropy")
        cumulative_marginal_raw_curves[label] = (
            d["cumulative_marginal_action_entropy"]
            if "cumulative_marginal_action_entropy" in d
            else np.cumsum(raw_marginal_entropy)
        )
        cumulative_belief_raw_curves[label] = (
            d["cumulative_mean_belief_entropy"]
            if "cumulative_mean_belief_entropy" in d
            else np.cumsum(raw_belief_entropy)
        )
        cumulative_marginal_normalized_curves[label] = np.cumsum(marginal_entropy)
        cumulative_belief_normalized_curves[label] = np.cumsum(belief_entropy)

        raw_policy_entropy = pick(d, "mean_conditional_policy_entropy")
        normalized_policy_entropy = pick(d, "mean_normalized_conditional_policy_entropy")
        cumulative_policy_raw_curves[label] = (
            d["cumulative_mean_conditional_policy_entropy"]
            if "cumulative_mean_conditional_policy_entropy" in d
            else np.cumsum(raw_policy_entropy)
        )
        cumulative_policy_normalized_curves[label] = np.cumsum(normalized_policy_entropy)

        belief_reduction_raw = raw_belief_entropy[0] - raw_belief_entropy
        belief_reduction_normalized = belief_entropy[0] - belief_entropy
        belief_dissipation_raw = np.zeros_like(raw_belief_entropy)
        belief_dissipation_raw[1:] = raw_belief_entropy[:-1] - raw_belief_entropy[1:]
        belief_dissipation_normalized = np.zeros_like(belief_entropy)
        belief_dissipation_normalized[1:] = belief_entropy[:-1] - belief_entropy[1:]
        regret_since_reference = regret - regret[0]
        belief_efficiency = np.full_like(belief_reduction_raw, np.nan, dtype=float)
        valid = belief_reduction_raw > minimum_entropy_reduction
        belief_efficiency[valid] = regret_since_reference[valid] / belief_reduction_raw[valid]
        belief_reduction_raw_curves[label] = belief_reduction_raw
        belief_reduction_normalized_curves[label] = belief_reduction_normalized
        belief_dissipation_raw_curves[label] = belief_dissipation_raw
        belief_dissipation_normalized_curves[label] = belief_dissipation_normalized
        belief_efficiency_curves[label] = belief_efficiency
        final_belief_reduction_by_label[label] = float(belief_reduction_raw[-1])
        final_regret_by_label[label] = float(regret[-1])

        correctness = pick(d, "mean_belief_correctness")
        mismatch = pick(d, "mean_normalized_policy_belief_js_divergence")
        efficiency = pick(d, "regret_per_entropy_reduction")
        dissipation = pick(d, "normalized_marginal_entropy_dissipation_rate")
        kl_path = pick(d, "cumulative_mean_smoothed_policy_kl_path_length")

        regret_curves[label] = regret
        marginal_entropy_curves[label] = marginal_entropy
        belief_entropy_curves[label] = belief_entropy
        correctness_curves[label] = correctness
        mismatch_curves[label] = mismatch
        efficiency_curves[label] = efficiency
        dissipation_curves[label] = dissipation
        phase_curves[label] = (marginal_entropy, regret)
        policy_kl_path_curves[label] = kl_path

        # EXP3 (and any unsupported algorithm) may not have this curve.
        for col in (
            "mean_operational_exploration_rate",
            "operational_exploration_rate",
        ):
            if col in d and np.any(np.isfinite(d[col])):
                operational_exploration_curves[label] = d[col]
                break

    assert common_timesteps is not None

    # Time-series/group plots: identical plot functions, just with Greedy absent.
    save_comparison_regret_plot(
        timesteps=common_timesteps,
        mean_curves=regret_curves,
        output_path=destination / REGRET_COMPARISON_PLOT_FILENAME,
    )
    save_marginal_entropy_comparison_plot(
        timesteps=common_timesteps,
        curves=marginal_entropy_curves,
        output_path=destination / MARGINAL_ENTROPY_COMPARISON_PLOT_FILENAME,
    )
    save_belief_entropy_comparison_plot(
        timesteps=common_timesteps,
        curves=belief_entropy_curves,
        output_path=destination / BELIEF_ENTROPY_COMPARISON_PLOT_FILENAME,
    )
    save_cumulative_entropy_pair_plot(
        timesteps=common_timesteps,
        raw_curves=cumulative_marginal_raw_curves,
        normalized_curves=cumulative_marginal_normalized_curves,
        output_path=destination / CUMULATIVE_MARGINAL_ENTROPY_PLOT_FILENAME,
        entropy_name="Marginal Action Entropy",
    )
    save_cumulative_entropy_pair_plot(
        timesteps=common_timesteps,
        raw_curves=cumulative_belief_raw_curves,
        normalized_curves=cumulative_belief_normalized_curves,
        output_path=destination / CUMULATIVE_BELIEF_ENTROPY_PLOT_FILENAME,
        entropy_name="Belief Entropy",
    )
    save_cumulative_entropy_pair_plot(
        timesteps=common_timesteps,
        raw_curves=cumulative_policy_raw_curves,
        normalized_curves=cumulative_policy_normalized_curves,
        output_path=destination / CUMULATIVE_POLICY_ENTROPY_PLOT_FILENAME,
        entropy_name="Conditional Policy Entropy",
    )
    save_entropy_change_pair_plot(
        common_timesteps, belief_reduction_raw_curves, belief_reduction_normalized_curves,
        destination / BELIEF_ENTROPY_REDUCTION_PLOT_FILENAME,
        "Belief Entropy", "Reduction", "Belief entropy reduction (nats)",
        "Normalized belief entropy reduction",
    )
    save_entropy_change_pair_plot(
        common_timesteps, belief_dissipation_raw_curves, belief_dissipation_normalized_curves,
        destination / BELIEF_ENTROPY_DISSIPATION_PLOT_FILENAME,
        "Belief Entropy", "Dissipation", "H_B(t-1) - H_B(t) (nats/step)",
        "Normalized belief entropy dissipation / step",
    )
    save_generic_curve_comparison(
        common_timesteps, belief_efficiency_curves,
        destination / BELIEF_ENTROPY_EFFICIENCY_PLOT_FILENAME,
        "Belief-Entropy Reduction Efficiency Comparison (lower is better)",
        "Regret / belief-entropy reduction",
    )
    save_correctness_comparison_plot(
        timesteps=common_timesteps,
        curves=correctness_curves,
        output_path=destination / CORRECTNESS_COMPARISON_PLOT_FILENAME,
    )
    save_policy_belief_mismatch_comparison_plot(
        timesteps=common_timesteps,
        curves=mismatch_curves,
        output_path=destination / MISMATCH_COMPARISON_PLOT_FILENAME,
    )
    save_efficiency_comparison_plot(
        timesteps=common_timesteps,
        curves=efficiency_curves,
        minimum_entropy_reduction=minimum_entropy_reduction,
        output_path=destination / EFFICIENCY_COMPARISON_PLOT_FILENAME,
    )
    save_dissipation_comparison_plot(
        timesteps=common_timesteps,
        curves=dissipation_curves,
        output_path=destination / DISSIPATION_COMPARISON_PLOT_FILENAME,
    )
    save_action_entropy_regret_phase_comparison_plot(
        curves=phase_curves,
        output_path=destination / ACTION_ENTROPY_REGRET_PHASE_COMPARISON_PLOT_FILENAME,
    )
    save_policy_kl_path_comparison_plot(
        timesteps=common_timesteps,
        curves=policy_kl_path_curves,
        output_path=destination / POLICY_KL_PATH_COMPARISON_PLOT_FILENAME,
    )

    if operational_exploration_curves:
        save_operational_exploration_comparison_plot(
            timesteps=common_timesteps,
            curves=operational_exploration_curves,
            output_path=destination / OPERATIONAL_EXPLORATION_COMPARISON_PLOT_FILENAME,
        )

    # Scalar/final-horizon comparison plots come from the already-generated comparison CSV.
    final_path = comparisons_data_dir / FINAL_COMPARISON_FILENAME
    if not final_path.exists():
        raise FileNotFoundError(f"Missing existing comparison file: {final_path}")

    all_final_rows = read_final_rows(final_path)
    greedy_labels = {
        str(cfg.get("label", cfg["name"]))
        for cfg in config["algorithms"]
        if str(cfg.get("name", "")).strip().lower() == "greedy"
    }
    final_rows = (
        [row for row in all_final_rows if row.get("algorithm") not in greedy_labels]
        if exclude_greedy else all_final_rows
    )

    entropy_half_lives: dict[str, tuple[int | None, int | None, int | None]] = {}
    belief_half_life_quality: dict[str, tuple[int | None, float, float]] = {}

    for row in final_rows:
        label = str(row["algorithm"])
        entropy_half_lives[label] = (
            to_optional_int(row.get("marginal_entropy_half_life")),
            to_optional_int(row.get("conditional_entropy_half_life")),
            to_optional_int(row.get("belief_entropy_half_life")),
        )
        belief_half_life_quality[label] = (
            to_optional_int(row.get("belief_entropy_half_life")),
            float(row.get("belief_half_life_correctness", "nan")),
            float(row.get("belief_half_life_confident_but_wrong_probability", "nan")),
        )

    save_entropy_half_life_comparison_plot(
        half_lives=entropy_half_lives,
        output_path=destination / ENTROPY_HALF_LIFE_COMPARISON_PLOT_FILENAME,
    )
    save_belief_half_life_quality_comparison_plot(
        points=belief_half_life_quality,
        output_path=destination / BELIEF_HALF_LIFE_QUALITY_COMPARISON_PLOT_FILENAME,
    )

    pareto_regret = np.asarray(
        [float(row["final_mean_cumulative_pseudo_regret"]) for row in final_rows],
        dtype=float,
    )
    pareto_entropy_reduction = np.asarray(
        [float(row["final_marginal_entropy_reduction"]) for row in final_rows],
        dtype=float,
    )
    pareto_mask = pareto_efficient_mask(pareto_regret, pareto_entropy_reduction)
    save_pareto_frontier_plot(
        labels=[str(row["algorithm"]) for row in final_rows],
        regret=pareto_regret,
        entropy_reduction=pareto_entropy_reduction,
        efficient_mask=pareto_mask,
        output_path=destination / PARETO_FRONTIER_PLOT_FILENAME,
    )

    belief_labels = [str(row["algorithm"]) for row in final_rows]
    belief_regret = np.asarray([final_regret_by_label[label] for label in belief_labels], dtype=float)
    belief_reduction = np.asarray([final_belief_reduction_by_label[label] for label in belief_labels], dtype=float)
    belief_pareto_mask = pareto_efficient_mask(belief_regret, belief_reduction)
    save_belief_pareto_plot(
        labels=belief_labels,
        regret=belief_regret,
        belief_entropy_reduction=belief_reduction,
        efficient_mask=belief_pareto_mask,
        output_path=destination / BELIEF_PARETO_FRONTIER_PLOT_FILENAME,
    )

    print(f"Generated comparison plots in: {destination.resolve()}")


def main() -> None:
    print("Regenerating comparison plots from existing CSV files only...")
    _generate_once(exclude_greedy=False, destination_name=WITH_GREEDY_DIR_NAME)
    _generate_once(exclude_greedy=True, destination_name=WITHOUT_GREEDY_DIR_NAME)
    print("Done. No experiments were rerun.")


if __name__ == "__main__":
    main()