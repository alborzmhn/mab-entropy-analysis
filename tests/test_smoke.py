import numpy as np

from mab_entropy.algorithms.factory import build_algorithm
from mab_entropy.algorithms.greedy import Greedy
from mab_entropy.environments.bernoulli_bandit import BernoulliBandit
from mab_entropy.experiments.runner import run_many, run_single
from mab_entropy.observers.beta_bernoulli import BetaBernoulliObserver


def observer_factory(n_arms: int, seed: int) -> BetaBernoulliObserver:
    return BetaBernoulliObserver(
        n_arms=n_arms,
        alpha_prior=1.0,
        beta_prior=1.0,
        optimal_arm_probability_samples=128,
        seed=seed,
    )


def test_bandit_properties() -> None:
    env = BernoulliBandit([0.2, 0.4, 0.8], seed=123)
    assert env.n_arms == 3
    assert env.optimal_arm == 2
    assert env.optimal_mean == 0.8


def test_greedy_starts_immediately_from_greedy_rule() -> None:
    algo = Greedy(n_arms=3, seed=11)

    first = algo.select_arm()
    assert 0 <= first < 3
    algo.update(first, 1)

    assert algo.select_arm() == first


def test_single_run_contains_policy_and_belief_metrics() -> None:
    records = run_single(
        means=[0.2, 0.4, 0.8],
        horizon=20,
        run_id=0,
        env_seed=7,
        algorithm_seed=8,
        observer_seed=9,
        algorithm_factory=lambda n_arms, seed: Greedy(n_arms, seed=seed),
        observer_factory=observer_factory,
        confident_entropy_threshold_normalized=0.2,
        wrong_correctness_threshold=0.2,
    )

    assert len(records) == 20
    assert np.isclose(records[0].conditional_policy_entropy, np.log(3))
    assert np.isclose(records[0].normalized_conditional_policy_entropy, 1.0)
    assert len(records[0].optimal_arm_belief_probabilities) == 3
    assert 0.0 <= records[0].normalized_belief_entropy <= 1.0
    assert 0.0 <= records[0].belief_correctness <= 1.0
    assert 0.0 <= records[0].normalized_policy_belief_js_divergence <= 1.0


def test_many_runs_compute_action_and_belief_metrics() -> None:
    result = run_many(
        means=[0.2, 0.4, 0.8],
        horizon=30,
        n_runs=40,
        base_seed=42,
        half_life_fraction=0.5,
        confident_entropy_threshold_normalized=0.2,
        wrong_correctness_threshold=0.2,
        algorithm_factory=lambda n_arms, seed: Greedy(n_arms, seed=seed),
        observer_factory=observer_factory,
    )

    assert result.marginal_action_entropy.shape == (30,)
    assert result.mean_conditional_policy_entropy.shape == (30,)
    assert result.history_action_mutual_information.shape == (30,)
    assert result.marginal_action_probabilities.shape == (30, 3)
    assert result.empirical_selected_action_probabilities.shape == (30, 3)

    assert result.mean_belief_entropy.shape == (30,)
    assert result.mean_normalized_belief_entropy.shape == (30,)
    assert result.mean_belief_correctness.shape == (30,)
    assert result.belief_map_accuracy.shape == (30,)
    assert result.confident_but_wrong_probability.shape == (30,)
    assert result.mean_policy_belief_js_divergence.shape == (30,)
    assert result.mean_optimal_arm_belief_probabilities.shape == (30, 3)

    assert result.marginal_entropy_reduction.shape == (30,)
    assert result.marginal_entropy_dissipation_rate.shape == (30,)
    assert result.regret_per_entropy_reduction.shape == (30,)
    assert result.action_misidentification_probability.shape == (30,)
    assert result.proposal_predicted_action_entropy.shape == (30,)
    assert result.asymptotic_predicted_action_entropy.shape == (30,)
    assert result.mean_policy_step_kl_divergence.shape == (30,)
    assert result.cumulative_mean_policy_kl_path_length.shape == (30,)
    assert result.mean_smoothed_policy_step_kl_divergence.shape == (30,)
    assert result.cumulative_mean_smoothed_policy_kl_path_length.shape == (30,)
    assert np.isfinite(result.proposal_entropy_rmse)
    assert np.isfinite(result.asymptotic_entropy_rmse)
    assert np.isfinite(result.final_mean_smoothed_policy_kl_path_length)

    assert np.allclose(
        result.history_action_mutual_information,
        np.maximum(
            0.0,
            result.marginal_action_entropy - result.mean_conditional_policy_entropy,
        ),
    )
    assert np.allclose(
        result.marginal_action_probabilities.sum(axis=1),
        1.0,
    )
    assert np.allclose(
        result.mean_optimal_arm_belief_probabilities.sum(axis=1),
        1.0,
    )
    assert np.all((result.mean_belief_correctness >= 0.0))
    assert np.all((result.mean_belief_correctness <= 1.0))
    assert np.all((result.confident_but_wrong_probability >= 0.0))
    assert np.all((result.confident_but_wrong_probability <= 1.0))


def test_all_algorithm_configs_run() -> None:
    configs = [
        {"name": "greedy"},
        {
            "name": "epsilon_greedy",
            "epsilon": 0.1,
            "schedule": "constant",
            "min_epsilon": 0.0,
        },
        {
            "name": "epsilon_greedy",
            "epsilon": 1.0,
            "schedule": "inverse_sqrt",
            "min_epsilon": 0.01,
        },
        {"name": "ucb1", "exploration_coefficient": 2.0},
        {
            "name": "thompson_sampling",
            "alpha_prior": 1.0,
            "beta_prior": 1.0,
            "policy_probability_samples": 64,
        },
        {"name": "exp3", "gamma": 0.1},
    ]

    for cfg in configs:
        result = run_many(
            means=[0.2, 0.4, 0.8],
            horizon=20,
            n_runs=8,
            base_seed=42,
            half_life_fraction=0.5,
            confident_entropy_threshold_normalized=0.2,
            wrong_correctness_threshold=0.2,
            algorithm_factory=lambda n_arms, seed, cfg=cfg: (
                build_algorithm(cfg, n_arms=n_arms, seed=seed)
            ),
            observer_factory=observer_factory,
        )

        assert len(result.records) == 160
        assert result.timesteps.shape == (20,)
        assert result.mean_cumulative_regret.shape == (20,)
        assert result.std_cumulative_regret.shape == (20,)
        assert result.marginal_action_entropy.shape == (20,)
        assert result.mean_conditional_policy_entropy.shape == (20,)
        assert result.mean_belief_entropy.shape == (20,)
        assert result.mean_belief_correctness.shape == (20,)
        assert result.marginal_entropy_reduction.shape == (20,)
        assert result.marginal_entropy_dissipation_rate.shape == (20,)
        assert result.proposal_predicted_action_entropy.shape == (20,)
        assert result.mean_smoothed_policy_step_kl_divergence.shape == (20,)
        assert np.all(
            np.diff(result.mean_cumulative_regret) >= -1e-12
        )
