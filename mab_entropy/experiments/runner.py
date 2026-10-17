from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from mab_entropy.algorithms.base import BanditAlgorithm
from mab_entropy.environments.bernoulli_bandit import BernoulliBandit
from mab_entropy.metrics.belief import (
    jensen_shannon_divergence,
    normalized_jensen_shannon_divergence,
    posterior_correctness_probability,
)
from mab_entropy.metrics.entropy import (
    empirical_action_distribution,
    entropy_half_life,
    normalized_shannon_entropy,
    shannon_entropy,
)
from mab_entropy.metrics.regret import instantaneous_pseudo_regret
from mab_entropy.metrics.research import (
    asymptotic_entropy_from_misidentification,
    entropy_dissipation_rate,
    entropy_reduction_curve,
    policy_kl_step_divergences,
    proposal_entropy_from_misidentification,
    regret_per_entropy_reduction,
)
from mab_entropy.observers.beta_bernoulli import BetaBernoulliObserver


AlgorithmFactory = Callable[[int, int], BanditAlgorithm]
ObserverFactory = Callable[[int, int], BetaBernoulliObserver]


# === File-local configuration ===
# Child RNG streams per run: environment, algorithm, and external observer.
RNG_STREAMS_PER_RUN = 3
# Population standard deviation is used across independent runs.
STD_DDOF = 0
# Smallest valid time-step index exposed in saved results.
FIRST_TIMESTEP = 1


@dataclass(frozen=True)
class RunRecord:
    run_id: int
    t: int
    selected_arm: int
    reward: int
    selected_arm_mean: float
    optimal_arm: int
    optimal_mean: float
    instantaneous_pseudo_regret: float
    cumulative_pseudo_regret: float

    # Policy layer: P(A_t=a | F_t).
    policy_probabilities: tuple[float, ...]
    conditional_policy_entropy: float
    normalized_conditional_policy_entropy: float

    # Operational exploration relative to the algorithm's current greedy estimate.
    operational_exploration: float | None
    operational_exploitation: float | None

    # Evidence layer exposed by the common observer before the action at t.
    observer_pull_counts: tuple[int, ...]
    observer_empirical_means: tuple[float, ...]
    observer_posterior_means: tuple[float, ...]

    # Belief layer: q_t(a)=P(A*=a | F_t) under the common observer.
    optimal_arm_belief_probabilities: tuple[float, ...]
    belief_entropy: float
    normalized_belief_entropy: float
    belief_correctness: float
    belief_map_is_correct: bool

    # Alignment between policy and belief.
    policy_belief_js_divergence: float
    normalized_policy_belief_js_divergence: float
    confident_but_wrong: bool


@dataclass
class MultiRunResult:
    records: list[RunRecord]
    timesteps: np.ndarray

    # Performance.
    mean_cumulative_regret: np.ndarray
    std_cumulative_regret: np.ndarray

    # Policy/action entropy decomposition.
    mean_conditional_policy_entropy: np.ndarray
    std_conditional_policy_entropy: np.ndarray
    mean_normalized_conditional_policy_entropy: np.ndarray
    marginal_action_probabilities: np.ndarray
    empirical_selected_action_probabilities: np.ndarray
    marginal_action_entropy: np.ndarray
    normalized_marginal_action_entropy: np.ndarray
    history_action_mutual_information: np.ndarray
    normalized_history_action_mutual_information: np.ndarray
    cumulative_marginal_action_entropy: np.ndarray
    cumulative_mean_conditional_policy_entropy: np.ndarray
    cumulative_history_action_mutual_information: np.ndarray
    marginal_entropy_half_life: int | None
    conditional_entropy_half_life: int | None

    # Common-observer evidence/belief analysis.
    mean_observer_pull_counts: np.ndarray
    mean_observer_posterior_means: np.ndarray
    mean_optimal_arm_belief_probabilities: np.ndarray
    mean_belief_entropy: np.ndarray
    std_belief_entropy: np.ndarray
    mean_normalized_belief_entropy: np.ndarray
    mean_belief_correctness: np.ndarray
    std_belief_correctness: np.ndarray
    belief_map_accuracy: np.ndarray
    confident_but_wrong_probability: np.ndarray
    mean_policy_belief_js_divergence: np.ndarray
    std_policy_belief_js_divergence: np.ndarray
    mean_normalized_policy_belief_js_divergence: np.ndarray
    cumulative_mean_belief_entropy: np.ndarray
    belief_entropy_half_life: int | None
    belief_half_life_correctness: float
    belief_half_life_map_accuracy: float
    belief_half_life_confident_but_wrong_probability: float

    # Direct action-quality diagnostics.
    marginal_optimal_action_probability: np.ndarray
    empirical_optimal_action_probability: np.ndarray

    # Policy and marginal correctness diagnostics relative to the true optimum.
    mean_policy_optimal_action_probability: np.ndarray
    policy_map_accuracy: np.ndarray
    marginal_map_is_correct: np.ndarray

    # Operational exploration/exploitation. NaN for unsupported algorithms (EXP3).
    mean_operational_exploration_rate: np.ndarray
    mean_operational_exploitation_rate: np.ndarray
    operational_exploration_supported: bool

    # Proposal research metrics: entropy reduction and dissipation.
    marginal_entropy_reduction: np.ndarray
    normalized_marginal_entropy_reduction: np.ndarray
    marginal_entropy_dissipation_rate: np.ndarray
    normalized_marginal_entropy_dissipation_rate: np.ndarray
    regret_per_entropy_reduction: np.ndarray

    # Proposal theoretical link H_t versus p_t=P(A_t != a*).
    action_misidentification_probability: np.ndarray
    proposal_predicted_action_entropy: np.ndarray
    asymptotic_predicted_action_entropy: np.ndarray
    proposal_entropy_absolute_error: np.ndarray
    asymptotic_entropy_absolute_error: np.ndarray
    proposal_entropy_rmse: float
    asymptotic_entropy_rmse: float

    # Information-geometric policy path diagnostics.
    mean_policy_step_kl_divergence: np.ndarray
    cumulative_mean_policy_kl_path_length: np.ndarray
    infinite_policy_step_kl_fraction: np.ndarray
    mean_smoothed_policy_step_kl_divergence: np.ndarray
    std_smoothed_policy_step_kl_divergence: np.ndarray
    cumulative_mean_smoothed_policy_kl_path_length: np.ndarray
    final_mean_smoothed_policy_kl_path_length: float
    mean_smoothed_policy_step_kl_temporal_std: float

    optimal_arm: int
    optimal_mean: float

    def save_records_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.records:
            raise ValueError("No records are available to save.")

        fieldnames = list(asdict(self.records[0]).keys())

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))

    def save_summary_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        n_arms = self.marginal_action_probabilities.shape[1]
        marginal_probability_headers = [
            f"marginal_action_probability_arm_{arm}"
            for arm in range(n_arms)
        ]
        empirical_probability_headers = [
            f"empirical_selected_action_probability_arm_{arm}"
            for arm in range(n_arms)
        ]
        belief_probability_headers = [
            f"mean_optimal_arm_belief_probability_arm_{arm}"
            for arm in range(n_arms)
        ]
        observer_count_headers = [
            f"mean_observer_pull_count_arm_{arm}"
            for arm in range(n_arms)
        ]
        observer_posterior_mean_headers = [
            f"mean_observer_posterior_mean_arm_{arm}"
            for arm in range(n_arms)
        ]

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t",
                "mean_cumulative_pseudo_regret",
                "std_cumulative_pseudo_regret",
                "mean_conditional_policy_entropy",
                "std_conditional_policy_entropy",
                "mean_normalized_conditional_policy_entropy",
                "marginal_action_entropy",
                "normalized_marginal_action_entropy",
                "history_action_mutual_information",
                "normalized_history_action_mutual_information",
                "cumulative_mean_conditional_policy_entropy",
                "cumulative_marginal_action_entropy",
                "cumulative_history_action_mutual_information",
                "mean_belief_entropy",
                "std_belief_entropy",
                "mean_normalized_belief_entropy",
                "mean_belief_correctness",
                "std_belief_correctness",
                "belief_map_accuracy",
                "confident_but_wrong_probability",
                "mean_policy_belief_js_divergence",
                "std_policy_belief_js_divergence",
                "mean_normalized_policy_belief_js_divergence",
                "cumulative_mean_belief_entropy",
                "marginal_optimal_action_probability",
                "empirical_optimal_action_probability",
                "mean_policy_optimal_action_probability",
                "policy_map_accuracy",
                "marginal_map_is_correct",
                "mean_operational_exploration_rate",
                "mean_operational_exploitation_rate",
                "marginal_entropy_reduction",
                "normalized_marginal_entropy_reduction",
                "marginal_entropy_dissipation_rate",
                "normalized_marginal_entropy_dissipation_rate",
                "regret_per_entropy_reduction",
                "action_misidentification_probability",
                "proposal_predicted_action_entropy",
                "asymptotic_predicted_action_entropy",
                "proposal_entropy_absolute_error",
                "asymptotic_entropy_absolute_error",
                "mean_policy_step_kl_divergence",
                "cumulative_mean_policy_kl_path_length",
                "infinite_policy_step_kl_fraction",
                "mean_smoothed_policy_step_kl_divergence",
                "std_smoothed_policy_step_kl_divergence",
                "cumulative_mean_smoothed_policy_kl_path_length",
                *marginal_probability_headers,
                *empirical_probability_headers,
                *belief_probability_headers,
                *observer_count_headers,
                *observer_posterior_mean_headers,
            ])

            for i, t in enumerate(self.timesteps):
                writer.writerow([
                    int(t),
                    float(self.mean_cumulative_regret[i]),
                    float(self.std_cumulative_regret[i]),
                    float(self.mean_conditional_policy_entropy[i]),
                    float(self.std_conditional_policy_entropy[i]),
                    float(self.mean_normalized_conditional_policy_entropy[i]),
                    float(self.marginal_action_entropy[i]),
                    float(self.normalized_marginal_action_entropy[i]),
                    float(self.history_action_mutual_information[i]),
                    float(self.normalized_history_action_mutual_information[i]),
                    float(self.cumulative_mean_conditional_policy_entropy[i]),
                    float(self.cumulative_marginal_action_entropy[i]),
                    float(self.cumulative_history_action_mutual_information[i]),
                    float(self.mean_belief_entropy[i]),
                    float(self.std_belief_entropy[i]),
                    float(self.mean_normalized_belief_entropy[i]),
                    float(self.mean_belief_correctness[i]),
                    float(self.std_belief_correctness[i]),
                    float(self.belief_map_accuracy[i]),
                    float(self.confident_but_wrong_probability[i]),
                    float(self.mean_policy_belief_js_divergence[i]),
                    float(self.std_policy_belief_js_divergence[i]),
                    float(self.mean_normalized_policy_belief_js_divergence[i]),
                    float(self.cumulative_mean_belief_entropy[i]),
                    float(self.marginal_optimal_action_probability[i]),
                    float(self.empirical_optimal_action_probability[i]),
                    float(self.mean_policy_optimal_action_probability[i]),
                    float(self.policy_map_accuracy[i]),
                    float(self.marginal_map_is_correct[i]),
                    float(self.mean_operational_exploration_rate[i]),
                    float(self.mean_operational_exploitation_rate[i]),
                    float(self.marginal_entropy_reduction[i]),
                    float(self.normalized_marginal_entropy_reduction[i]),
                    float(self.marginal_entropy_dissipation_rate[i]),
                    float(self.normalized_marginal_entropy_dissipation_rate[i]),
                    float(self.regret_per_entropy_reduction[i]),
                    float(self.action_misidentification_probability[i]),
                    float(self.proposal_predicted_action_entropy[i]),
                    float(self.asymptotic_predicted_action_entropy[i]),
                    float(self.proposal_entropy_absolute_error[i]),
                    float(self.asymptotic_entropy_absolute_error[i]),
                    float(self.mean_policy_step_kl_divergence[i]),
                    float(self.cumulative_mean_policy_kl_path_length[i]),
                    float(self.infinite_policy_step_kl_fraction[i]),
                    float(self.mean_smoothed_policy_step_kl_divergence[i]),
                    float(self.std_smoothed_policy_step_kl_divergence[i]),
                    float(self.cumulative_mean_smoothed_policy_kl_path_length[i]),
                    *[
                        float(x)
                        for x in self.marginal_action_probabilities[i]
                    ],
                    *[
                        float(x)
                        for x in self.empirical_selected_action_probabilities[i]
                    ],
                    *[
                        float(x)
                        for x in self.mean_optimal_arm_belief_probabilities[i]
                    ],
                    *[
                        float(x)
                        for x in self.mean_observer_pull_counts[i]
                    ],
                    *[
                        float(x)
                        for x in self.mean_observer_posterior_means[i]
                    ],
                ])


def _operational_reference_values(algorithm: BanditAlgorithm) -> np.ndarray | None:
    """Return the algorithm's current mean-reward estimate for exploration analysis.

    Greedy, epsilon-greedy, and UCB expose empirical value estimates directly.
    Thompson Sampling uses its Beta posterior means. EXP3 is intentionally
    unsupported because its internal decision state is adversarial-bandit weights
    rather than an estimate of stationary arm means.
    """
    values = getattr(algorithm, "value_estimates", None)
    if values is not None:
        return np.asarray(values, dtype=float).copy()

    alpha = getattr(algorithm, "alpha", None)
    beta = getattr(algorithm, "beta", None)
    if alpha is not None and beta is not None:
        alpha_array = np.asarray(alpha, dtype=float)
        beta_array = np.asarray(beta, dtype=float)
        return alpha_array / (alpha_array + beta_array)

    return None


def _operational_exploration_indicator(
    *,
    algorithm: BanditAlgorithm,
    selected_arm: int,
    reference_values: np.ndarray | None,
) -> float | None:
    """Classify whether the selected action departs from current greedy belief.

    Ties are treated as equally greedy, so randomized tie-breaking in Greedy is
    not mislabeled as exploration. For UCB's deterministic forced initialization,
    selecting a previously unseen arm after at least one arm has been observed is
    treated as exploration even if empirical means are tied at zero.
    """
    if reference_values is None:
        return None

    counts = getattr(algorithm, "counts", None)
    if algorithm.__class__.__name__ == "UCB1" and counts is not None:
        counts_array = np.asarray(counts)
        if np.any(counts_array > 0) and counts_array[selected_arm] == 0:
            return 1.0

    maximizers = np.flatnonzero(
        np.isclose(reference_values, np.max(reference_values))
    )
    return float(selected_arm not in set(int(x) for x in maximizers))


def run_single(
    *,
    means: list[float],
    horizon: int,
    run_id: int,
    env_seed: int,
    algorithm_seed: int,
    observer_seed: int,
    algorithm_factory: AlgorithmFactory,
    observer_factory: ObserverFactory,
    confident_entropy_threshold_normalized: float,
    wrong_correctness_threshold: float,
) -> list[RunRecord]:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    env = BernoulliBandit(means=means, seed=env_seed)
    algorithm = algorithm_factory(env.n_arms, algorithm_seed)
    observer = observer_factory(env.n_arms, observer_seed)

    cumulative_regret = 0.0
    records: list[RunRecord] = []

    for t in range(1, horizon + 1):
        # All analysis quantities are measured before A_t is sampled, so they
        # condition on exactly the history F_t available at decision time.
        policy_probabilities = algorithm.action_probabilities()
        policy_entropy = shannon_entropy(policy_probabilities)
        normalized_policy_entropy = normalized_shannon_entropy(
            policy_probabilities
        )
        operational_reference_values = _operational_reference_values(algorithm)

        observer_pull_counts = observer.counts.copy()
        observer_empirical_means = observer.empirical_means
        observer_posterior_means = observer.posterior_means

        belief_probabilities = observer.optimal_arm_probabilities()
        belief_entropy = shannon_entropy(belief_probabilities)
        normalized_belief_entropy = normalized_shannon_entropy(
            belief_probabilities
        )
        belief_correctness = posterior_correctness_probability(
            belief_probabilities,
            true_optimal_arm=env.optimal_arm,
        )
        belief_map_is_correct = bool(
            int(np.argmax(belief_probabilities)) == env.optimal_arm
        )

        policy_belief_js = jensen_shannon_divergence(
            policy_probabilities,
            belief_probabilities,
        )
        normalized_policy_belief_js = normalized_jensen_shannon_divergence(
            policy_probabilities,
            belief_probabilities,
        )
        confident_but_wrong = bool(
            normalized_belief_entropy
            < confident_entropy_threshold_normalized
            and belief_correctness < wrong_correctness_threshold
        )

        arm = algorithm.select_arm()
        operational_exploration = _operational_exploration_indicator(
            algorithm=algorithm,
            selected_arm=arm,
            reference_values=operational_reference_values,
        )
        operational_exploitation = (
            None
            if operational_exploration is None
            else 1.0 - operational_exploration
        )
        reward = env.pull(arm)
        algorithm.update(arm, reward)
        observer.update(arm, reward)

        selected_mean = env.arm_mean(arm)
        instant_regret = instantaneous_pseudo_regret(
            optimal_mean=env.optimal_mean,
            selected_arm_mean=selected_mean,
        )
        cumulative_regret += instant_regret

        records.append(
            RunRecord(
                run_id=run_id,
                t=t,
                selected_arm=arm,
                reward=reward,
                selected_arm_mean=selected_mean,
                optimal_arm=env.optimal_arm,
                optimal_mean=env.optimal_mean,
                instantaneous_pseudo_regret=instant_regret,
                cumulative_pseudo_regret=cumulative_regret,
                policy_probabilities=tuple(
                    float(x) for x in policy_probabilities
                ),
                conditional_policy_entropy=policy_entropy,
                normalized_conditional_policy_entropy=(
                    normalized_policy_entropy
                ),
                operational_exploration=operational_exploration,
                operational_exploitation=operational_exploitation,
                observer_pull_counts=tuple(
                    int(x) for x in observer_pull_counts
                ),
                observer_empirical_means=tuple(
                    float(x) for x in observer_empirical_means
                ),
                observer_posterior_means=tuple(
                    float(x) for x in observer_posterior_means
                ),
                optimal_arm_belief_probabilities=tuple(
                    float(x) for x in belief_probabilities
                ),
                belief_entropy=belief_entropy,
                normalized_belief_entropy=normalized_belief_entropy,
                belief_correctness=belief_correctness,
                belief_map_is_correct=belief_map_is_correct,
                policy_belief_js_divergence=policy_belief_js,
                normalized_policy_belief_js_divergence=(
                    normalized_policy_belief_js
                ),
                confident_but_wrong=confident_but_wrong,
            )
        )

    return records


def run_many(
    *,
    means: list[float],
    horizon: int,
    n_runs: int,
    base_seed: int,
    algorithm_factory: AlgorithmFactory,
    observer_factory: ObserverFactory,
    half_life_fraction: float,
    confident_entropy_threshold_normalized: float,
    wrong_correctness_threshold: float,
    policy_kl_probability_floor: float = 1e-12,
    entropy_efficiency_minimum_reduction: float = 1e-12,
    entropy_reduction_reference_index: int = 0,
) -> MultiRunResult:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if n_runs <= 0:
        raise ValueError("n_runs must be positive.")
    if not 0.0 <= confident_entropy_threshold_normalized <= 1.0:
        raise ValueError(
            "confident_entropy_threshold_normalized must lie in [0, 1]."
        )
    if not 0.0 <= wrong_correctness_threshold <= 1.0:
        raise ValueError("wrong_correctness_threshold must lie in [0, 1].")
    if policy_kl_probability_floor <= 0.0:
        raise ValueError("policy_kl_probability_floor must be positive.")
    if entropy_efficiency_minimum_reduction <= 0.0:
        raise ValueError(
            "entropy_efficiency_minimum_reduction must be positive."
        )
    if entropy_reduction_reference_index < 0 or entropy_reduction_reference_index >= horizon:
        raise ValueError(
            "entropy_reduction_reference_index must lie in [0, horizon)."
        )

    probe_env = BernoulliBandit(means=means, seed=base_seed)
    all_records: list[RunRecord] = []

    regret_matrix = np.zeros((n_runs, horizon), dtype=float)
    conditional_entropy_matrix = np.zeros((n_runs, horizon), dtype=float)
    normalized_conditional_entropy_matrix = np.zeros(
        (n_runs, horizon),
        dtype=float,
    )
    operational_exploration_matrix = np.full(
        (n_runs, horizon), np.nan, dtype=float
    )
    selected_arm_matrix = np.zeros((n_runs, horizon), dtype=np.int64)
    policy_probability_tensor = np.zeros(
        (n_runs, horizon, probe_env.n_arms),
        dtype=float,
    )

    observer_count_tensor = np.zeros(
        (n_runs, horizon, probe_env.n_arms),
        dtype=float,
    )
    observer_posterior_mean_tensor = np.zeros(
        (n_runs, horizon, probe_env.n_arms),
        dtype=float,
    )
    belief_probability_tensor = np.zeros(
        (n_runs, horizon, probe_env.n_arms),
        dtype=float,
    )
    belief_entropy_matrix = np.zeros((n_runs, horizon), dtype=float)
    normalized_belief_entropy_matrix = np.zeros(
        (n_runs, horizon),
        dtype=float,
    )
    belief_correctness_matrix = np.zeros((n_runs, horizon), dtype=float)
    belief_map_correct_matrix = np.zeros((n_runs, horizon), dtype=float)
    confident_wrong_matrix = np.zeros((n_runs, horizon), dtype=float)
    policy_belief_js_matrix = np.zeros((n_runs, horizon), dtype=float)
    normalized_policy_belief_js_matrix = np.zeros(
        (n_runs, horizon),
        dtype=float,
    )

    master_sequence = np.random.SeedSequence(base_seed)
    run_sequences = master_sequence.spawn(n_runs)

    for run_id, run_sequence in enumerate(run_sequences):
        (
            env_sequence,
            algorithm_sequence,
            observer_sequence,
        ) = run_sequence.spawn(RNG_STREAMS_PER_RUN)

        env_seed = int(env_sequence.generate_state(1, dtype=np.uint32)[0])
        algorithm_seed = int(
            algorithm_sequence.generate_state(1, dtype=np.uint32)[0]
        )
        observer_seed = int(
            observer_sequence.generate_state(1, dtype=np.uint32)[0]
        )

        records = run_single(
            means=means,
            horizon=horizon,
            run_id=run_id,
            env_seed=env_seed,
            algorithm_seed=algorithm_seed,
            observer_seed=observer_seed,
            algorithm_factory=algorithm_factory,
            observer_factory=observer_factory,
            confident_entropy_threshold_normalized=(
                confident_entropy_threshold_normalized
            ),
            wrong_correctness_threshold=wrong_correctness_threshold,
        )

        all_records.extend(records)
        regret_matrix[run_id, :] = [
            record.cumulative_pseudo_regret for record in records
        ]
        conditional_entropy_matrix[run_id, :] = [
            record.conditional_policy_entropy for record in records
        ]
        normalized_conditional_entropy_matrix[run_id, :] = [
            record.normalized_conditional_policy_entropy
            for record in records
        ]
        operational_exploration_matrix[run_id, :] = [
            np.nan
            if record.operational_exploration is None
            else float(record.operational_exploration)
            for record in records
        ]
        selected_arm_matrix[run_id, :] = [
            record.selected_arm for record in records
        ]
        policy_probability_tensor[run_id, :, :] = np.asarray(
            [record.policy_probabilities for record in records],
            dtype=float,
        )

        observer_count_tensor[run_id, :, :] = np.asarray(
            [record.observer_pull_counts for record in records],
            dtype=float,
        )
        observer_posterior_mean_tensor[run_id, :, :] = np.asarray(
            [record.observer_posterior_means for record in records],
            dtype=float,
        )
        belief_probability_tensor[run_id, :, :] = np.asarray(
            [record.optimal_arm_belief_probabilities for record in records],
            dtype=float,
        )
        belief_entropy_matrix[run_id, :] = [
            record.belief_entropy for record in records
        ]
        normalized_belief_entropy_matrix[run_id, :] = [
            record.normalized_belief_entropy for record in records
        ]
        belief_correctness_matrix[run_id, :] = [
            record.belief_correctness for record in records
        ]
        belief_map_correct_matrix[run_id, :] = [
            float(record.belief_map_is_correct) for record in records
        ]
        confident_wrong_matrix[run_id, :] = [
            float(record.confident_but_wrong) for record in records
        ]
        policy_belief_js_matrix[run_id, :] = [
            record.policy_belief_js_divergence for record in records
        ]
        normalized_policy_belief_js_matrix[run_id, :] = [
            record.normalized_policy_belief_js_divergence
            for record in records
        ]

    # P(A_t=a) = E_{F_t}[P(A_t=a | F_t)].
    marginal_probabilities = policy_probability_tensor.mean(axis=0)
    empirical_selected_probabilities = np.zeros(
        (horizon, probe_env.n_arms),
        dtype=float,
    )
    marginal_entropy = np.zeros(horizon, dtype=float)
    normalized_marginal_entropy = np.zeros(horizon, dtype=float)

    for t_index in range(horizon):
        marginal_distribution = marginal_probabilities[t_index]
        empirical_selected_probabilities[t_index] = (
            empirical_action_distribution(
                selected_arm_matrix[:, t_index],
                n_arms=probe_env.n_arms,
            )
        )
        marginal_entropy[t_index] = shannon_entropy(marginal_distribution)
        normalized_marginal_entropy[t_index] = normalized_shannon_entropy(
            marginal_distribution
        )

    mean_conditional_entropy = conditional_entropy_matrix.mean(axis=0)
    std_conditional_entropy = conditional_entropy_matrix.std(
        axis=0,
        ddof=STD_DDOF,
    )
    mean_normalized_conditional_entropy = (
        normalized_conditional_entropy_matrix.mean(axis=0)
    )

    operational_exploration_supported = not np.all(
        np.isnan(operational_exploration_matrix)
    )
    if operational_exploration_supported:
        mean_operational_exploration_rate = np.nanmean(
            operational_exploration_matrix, axis=0
        )
        mean_operational_exploitation_rate = (
            1.0 - mean_operational_exploration_rate
        )
    else:
        mean_operational_exploration_rate = np.full(horizon, np.nan)
        mean_operational_exploitation_rate = np.full(horizon, np.nan)

    history_action_mutual_information = np.maximum(
        0.0,
        marginal_entropy - mean_conditional_entropy,
    )
    normalized_history_action_mutual_information = np.maximum(
        0.0,
        normalized_marginal_entropy - mean_normalized_conditional_entropy,
    )

    # --- Proposal metrics: entropy reduction, dissipation, and efficiency. ---
    marginal_entropy_reduction = entropy_reduction_curve(
        marginal_entropy,
        reference_index=entropy_reduction_reference_index,
    )
    normalized_marginal_entropy_reduction = entropy_reduction_curve(
        normalized_marginal_entropy,
        reference_index=entropy_reduction_reference_index,
    )
    marginal_entropy_dissipation = entropy_dissipation_rate(marginal_entropy)
    normalized_marginal_entropy_dissipation = entropy_dissipation_rate(
        normalized_marginal_entropy
    )

    mean_cumulative_regret = regret_matrix.mean(axis=0)
    # Measure regret cost from the same reference used for entropy reduction.
    # This avoids charging UCB1's forced warm-up regret to the post-warm-up
    # entropy-reduction phase.
    reference_regret = float(mean_cumulative_regret[entropy_reduction_reference_index])
    regret_since_reference = mean_cumulative_regret - reference_regret
    regret_since_reference[:entropy_reduction_reference_index] = np.nan
    proposal_efficiency = regret_per_entropy_reduction(
        regret_since_reference,
        marginal_entropy_reduction,
        minimum_reduction=entropy_efficiency_minimum_reduction,
    )

    # --- Proposal theoretical link: H_t versus p_t=P(A_t != a*). ---
    marginal_optimal_action_probability = (
        marginal_probabilities[:, probe_env.optimal_arm]
    )

    # Policy-layer correctness: compute within each run/history first, then
    # average across runs. This is distinct from posterior correctness.
    policy_optimal_action_probability_matrix = (
        policy_probability_tensor[:, :, probe_env.optimal_arm]
    )
    mean_policy_optimal_action_probability = (
        policy_optimal_action_probability_matrix.mean(axis=0)
    )
    policy_map_accuracy = np.mean(
        np.argmax(policy_probability_tensor, axis=2)
        == probe_env.optimal_arm,
        axis=0,
    )

    # Marginal-layer MAP correctness: does the most probable action across
    # histories/runs coincide with the true optimal arm at this time step?
    marginal_map_is_correct = (
        np.argmax(marginal_probabilities, axis=1)
        == probe_env.optimal_arm
    ).astype(float)

    action_misidentification_probability = np.clip(
        1.0 - marginal_optimal_action_probability,
        0.0,
        1.0,
    )
    proposal_predicted_action_entropy = (
        proposal_entropy_from_misidentification(
            action_misidentification_probability,
            n_arms=probe_env.n_arms,
        )
    )
    asymptotic_predicted_action_entropy = (
        asymptotic_entropy_from_misidentification(
            action_misidentification_probability
        )
    )
    proposal_entropy_absolute_error = np.abs(
        marginal_entropy - proposal_predicted_action_entropy
    )
    asymptotic_entropy_absolute_error = np.abs(
        marginal_entropy - asymptotic_predicted_action_entropy
    )
    proposal_entropy_rmse = float(
        np.sqrt(np.mean((marginal_entropy - proposal_predicted_action_entropy) ** 2))
    )
    asymptotic_entropy_rmse = float(
        np.sqrt(np.mean((marginal_entropy - asymptotic_predicted_action_entropy) ** 2))
    )

    # --- Information geometry: per-run KL path of conditional policies. ---
    exact_policy_kl_matrix = np.zeros((n_runs, horizon), dtype=float)
    smoothed_policy_kl_matrix = np.zeros((n_runs, horizon), dtype=float)
    for run_id in range(n_runs):
        exact_kl, smoothed_kl = policy_kl_step_divergences(
            policy_probability_tensor[run_id],
            probability_floor=policy_kl_probability_floor,
        )
        exact_policy_kl_matrix[run_id] = exact_kl
        smoothed_policy_kl_matrix[run_id] = smoothed_kl

    with np.errstate(invalid="ignore"):
        mean_policy_step_kl = np.mean(exact_policy_kl_matrix, axis=0)
    cumulative_mean_policy_kl_path_length = np.cumsum(mean_policy_step_kl)
    infinite_policy_step_kl_fraction = np.mean(
        np.isinf(exact_policy_kl_matrix),
        axis=0,
    )
    mean_smoothed_policy_step_kl = smoothed_policy_kl_matrix.mean(axis=0)
    std_smoothed_policy_step_kl = smoothed_policy_kl_matrix.std(
        axis=0,
        ddof=STD_DDOF,
    )
    cumulative_mean_smoothed_policy_kl_path_length = np.cumsum(
        mean_smoothed_policy_step_kl
    )
    final_mean_smoothed_policy_kl_path_length = float(
        cumulative_mean_smoothed_policy_kl_path_length[-1]
    )
    # A compact smoothness statistic: average across runs of the temporal
    # standard deviation of finite, smoothed KL step sizes. Lower means the
    # geometric movement is more even over time.
    mean_smoothed_policy_step_kl_temporal_std = float(
        np.mean(np.std(smoothed_policy_kl_matrix, axis=1, ddof=STD_DDOF))
    )

    mean_belief_entropy = belief_entropy_matrix.mean(axis=0)
    mean_normalized_belief_entropy = (
        normalized_belief_entropy_matrix.mean(axis=0)
    )

    # Marginal entropy half-life should describe the post-initialization
    # concentration dynamics. UCB1 deterministically forces one pull of each
    # arm, so H^m can start at zero and make an H_1/2 half-life meaningless.
    # main.py already passes entropy_reduction_reference_index=K for UCB1 and
    # 0 for the other algorithms; reuse that same reference here.
    marginal_half_life_curve = marginal_entropy[
        entropy_reduction_reference_index:
    ]
    marginal_half_life_local = entropy_half_life(
        marginal_half_life_curve,
        fraction=half_life_fraction,
    )
    marginal_half_life = (
        None
        if marginal_half_life_local is None
        else marginal_half_life_local + entropy_reduction_reference_index
    )

    belief_half_life = entropy_half_life(
        mean_belief_entropy,
        fraction=half_life_fraction,
    )
    if belief_half_life is None:
        belief_half_life_correctness = float("nan")
        belief_half_life_map_accuracy = float("nan")
        belief_half_life_confident_but_wrong_probability = float("nan")
    else:
        half_life_index = belief_half_life - FIRST_TIMESTEP
        belief_half_life_correctness = float(
            belief_correctness_matrix.mean(axis=0)[half_life_index]
        )
        belief_half_life_map_accuracy = float(
            belief_map_correct_matrix.mean(axis=0)[half_life_index]
        )
        belief_half_life_confident_but_wrong_probability = float(
            confident_wrong_matrix.mean(axis=0)[half_life_index]
        )

    return MultiRunResult(
        records=all_records,
        timesteps=np.arange(FIRST_TIMESTEP, horizon + 1),
        mean_cumulative_regret=mean_cumulative_regret,
        std_cumulative_regret=regret_matrix.std(axis=0, ddof=STD_DDOF),
        mean_conditional_policy_entropy=mean_conditional_entropy,
        std_conditional_policy_entropy=std_conditional_entropy,
        mean_normalized_conditional_policy_entropy=(
            mean_normalized_conditional_entropy
        ),
        marginal_action_probabilities=marginal_probabilities,
        empirical_selected_action_probabilities=(
            empirical_selected_probabilities
        ),
        marginal_action_entropy=marginal_entropy,
        normalized_marginal_action_entropy=normalized_marginal_entropy,
        history_action_mutual_information=(
            history_action_mutual_information
        ),
        normalized_history_action_mutual_information=(
            normalized_history_action_mutual_information
        ),
        cumulative_marginal_action_entropy=np.cumsum(marginal_entropy),
        cumulative_mean_conditional_policy_entropy=np.cumsum(
            mean_conditional_entropy
        ),
        cumulative_history_action_mutual_information=np.cumsum(
            history_action_mutual_information
        ),
        marginal_entropy_half_life=marginal_half_life,
        conditional_entropy_half_life=entropy_half_life(
            mean_conditional_entropy,
            fraction=half_life_fraction,
        ),
        mean_observer_pull_counts=observer_count_tensor.mean(axis=0),
        mean_observer_posterior_means=(
            observer_posterior_mean_tensor.mean(axis=0)
        ),
        mean_optimal_arm_belief_probabilities=(
            belief_probability_tensor.mean(axis=0)
        ),
        mean_belief_entropy=mean_belief_entropy,
        std_belief_entropy=belief_entropy_matrix.std(
            axis=0,
            ddof=STD_DDOF,
        ),
        mean_normalized_belief_entropy=(
            mean_normalized_belief_entropy
        ),
        mean_belief_correctness=belief_correctness_matrix.mean(axis=0),
        std_belief_correctness=belief_correctness_matrix.std(
            axis=0,
            ddof=STD_DDOF,
        ),
        belief_map_accuracy=belief_map_correct_matrix.mean(axis=0),
        confident_but_wrong_probability=confident_wrong_matrix.mean(axis=0),
        mean_policy_belief_js_divergence=policy_belief_js_matrix.mean(axis=0),
        std_policy_belief_js_divergence=policy_belief_js_matrix.std(
            axis=0,
            ddof=STD_DDOF,
        ),
        mean_normalized_policy_belief_js_divergence=(
            normalized_policy_belief_js_matrix.mean(axis=0)
        ),
        cumulative_mean_belief_entropy=np.cumsum(mean_belief_entropy),
        belief_entropy_half_life=belief_half_life,
        belief_half_life_correctness=belief_half_life_correctness,
        belief_half_life_map_accuracy=belief_half_life_map_accuracy,
        belief_half_life_confident_but_wrong_probability=(
            belief_half_life_confident_but_wrong_probability
        ),
        marginal_optimal_action_probability=(
            marginal_optimal_action_probability
        ),
        empirical_optimal_action_probability=(
            empirical_selected_probabilities[:, probe_env.optimal_arm]
        ),
        mean_policy_optimal_action_probability=(
            mean_policy_optimal_action_probability
        ),
        policy_map_accuracy=policy_map_accuracy,
        marginal_map_is_correct=marginal_map_is_correct,
        mean_operational_exploration_rate=(
            mean_operational_exploration_rate
        ),
        mean_operational_exploitation_rate=(
            mean_operational_exploitation_rate
        ),
        operational_exploration_supported=(
            operational_exploration_supported
        ),
        marginal_entropy_reduction=marginal_entropy_reduction,
        normalized_marginal_entropy_reduction=(
            normalized_marginal_entropy_reduction
        ),
        marginal_entropy_dissipation_rate=marginal_entropy_dissipation,
        normalized_marginal_entropy_dissipation_rate=(
            normalized_marginal_entropy_dissipation
        ),
        regret_per_entropy_reduction=proposal_efficiency,
        action_misidentification_probability=(
            action_misidentification_probability
        ),
        proposal_predicted_action_entropy=(
            proposal_predicted_action_entropy
        ),
        asymptotic_predicted_action_entropy=(
            asymptotic_predicted_action_entropy
        ),
        proposal_entropy_absolute_error=proposal_entropy_absolute_error,
        asymptotic_entropy_absolute_error=asymptotic_entropy_absolute_error,
        proposal_entropy_rmse=proposal_entropy_rmse,
        asymptotic_entropy_rmse=asymptotic_entropy_rmse,
        mean_policy_step_kl_divergence=mean_policy_step_kl,
        cumulative_mean_policy_kl_path_length=(
            cumulative_mean_policy_kl_path_length
        ),
        infinite_policy_step_kl_fraction=(
            infinite_policy_step_kl_fraction
        ),
        mean_smoothed_policy_step_kl_divergence=(
            mean_smoothed_policy_step_kl
        ),
        std_smoothed_policy_step_kl_divergence=(
            std_smoothed_policy_step_kl
        ),
        cumulative_mean_smoothed_policy_kl_path_length=(
            cumulative_mean_smoothed_policy_kl_path_length
        ),
        final_mean_smoothed_policy_kl_path_length=(
            final_mean_smoothed_policy_kl_path_length
        ),
        mean_smoothed_policy_step_kl_temporal_std=(
            mean_smoothed_policy_step_kl_temporal_std
        ),
        optimal_arm=probe_env.optimal_arm,
        optimal_mean=probe_env.optimal_mean,
    )