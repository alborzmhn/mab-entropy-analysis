# MAB Entropy Research Codebase

This project compares stochastic multi-armed bandit algorithms through regret,
action entropy, best-arm belief entropy, policy-belief alignment, and the
research metrics proposed in the accompanying entropy-based MAB proposal.

## Implemented algorithms

- Greedy
- Epsilon-Greedy with constant epsilon
- Epsilon-Greedy with inverse-square-root epsilon decay
- UCB1
- Thompson Sampling
- EXP3

## Environment

The current baseline is a stationary Bernoulli bandit. The true arm means are
configured in `config.yaml`.

## Existing measurement layers

### 1. Evidence

A common Beta-Bernoulli observer receives the same action/reward history
produced by each algorithm. Before every decision it stores:

- number of observations per arm,
- empirical reward means,
- posterior means.

The observer does not affect the algorithm's decisions.

### 2. Belief about the optimal arm

The common observer estimates

`q_t(a) = P(A* = a | F_t)`

by posterior Monte Carlo. From this distribution the code records:

- best-arm belief entropy,
- normalized belief entropy,
- posterior correctness: mass on the true optimal arm,
- MAP best-arm accuracy,
- confident-but-wrong probability.

### 3. Policy

Each algorithm exposes

`pi_t(a) = P(A_t = a | F_t)`.

The code records conditional policy entropy for every run and averages it over
histories/runs.

### 4. Action across runs

The marginal action distribution is estimated by averaging the per-run policy
vectors:

`P(A_t=a) = E_Ft[P(A_t=a | F_t)]`.

The code also stores the empirical histogram of actually sampled actions as a
diagnostic.

### 5. Action entropy decomposition

The code evaluates

`H(A_t) = H(A_t | F_t) + I(A_t ; F_t)`.

The mutual-information term measures history-induced action variability. It is
not the same as information gained about the optimal arm.

### 6. Policy-belief mismatch

For each run and time step the code computes Jensen-Shannon divergence between
`pi_t` and `q_t`. A normalized version lies in `[0, 1]`.

For exact probability matching, Thompson Sampling has `pi_t = q_t` when its
prior/model matches the observer. In finite Monte Carlo estimation the measured
mismatch is expected to be small rather than exactly zero.

---

# Proposal research metrics now implemented

The following six proposal-driven analyses have been added.

## 1. Entropy reduction and entropy-reduction efficiency

For marginal action entropy `H_t`, the code computes

`DeltaH_t = H_1 - H_t`.

The proposal coefficient is operationalized using cumulative pseudo-regret:

`eta_t = cumulative_pseudo_regret_t / DeltaH_t`.

The saved field is named `regret_per_entropy_reduction` to avoid ambiguity.
Under this interpretation, **lower is better**: the algorithm accumulated less
regret per nat of action-entropy reduction.

The ratio is stored as `NaN` while entropy reduction is numerically too small.
The threshold is configurable through
`research_metrics.entropy_efficiency_minimum_reduction`.

## 2. Entropy dissipation rate

The discrete dissipation rate is

`D_t = H_{t-1} - H_t`,

with `D_1 = 0`.

Interpretation:

- positive: entropy decreased,
- zero: no change,
- negative: entropy increased/reheated.

Both raw and normalized marginal-action-entropy dissipation curves are saved.

## 3. Action-entropy / regret phase diagram

For every algorithm, the project now plots the trajectory

`(H_t, R_t)`

using normalized marginal action entropy and mean cumulative pseudo-regret.
A cross-algorithm phase diagram is also generated so the trajectories can be
compared as algorithmic fingerprints.

## 4. Entropy-regret Pareto frontier

At the common final horizon `T`, algorithms are compared with two objectives:

- minimize final mean cumulative pseudo-regret,
- maximize final action-entropy reduction.

An algorithm is Pareto dominated when another algorithm has no larger regret
and no smaller entropy reduction, with at least one strict improvement.

The final comparison CSV contains
`is_entropy_regret_pareto_efficient`, and the comparison folder contains a
Pareto-frontier plot.

## 5. Information-geometric policy path length and smoothness

For each independent run, the conditional policy path is analyzed through

`D_KL(pi_t || pi_{t-1})`.

The first step is zero because no previous policy exists.

Two versions are retained:

1. **Exact KL**. This follows the mathematical definition directly. It may be
   infinite when a deterministic policy changes support, which is expected for
   algorithms such as Greedy or UCB.
2. **Smoothed KL**. A very small probability floor is applied and the vector is
   renormalized before KL is evaluated. This produces a finite diagnostic that
   can be plotted and compared numerically.

The exact cumulative path length and the smoothed cumulative path length are
both saved. Smoothness is summarized by the average temporal standard deviation
of the per-run smoothed KL step sizes: lower values mean more even movement over
time.

The smoothing floor is configured through
`research_metrics.policy_kl_probability_floor`.

## 6. Theoretical entropy / misidentification check

The code defines

`p_t = P(A_t != a*) = 1 - P(A_t = a*)`

using the marginal action distribution.

It compares observed marginal action entropy with the proposal curve

`h(p_t) + p_t log(K-1)`

and with the asymptotic signature

`p_t log(1/p_t)`.

The first expression is exact when the total suboptimal-action mass is uniform
over the `K-1` suboptimal arms. In the general experiment it is therefore used
as a proposal-motivated reference curve rather than silently treated as an
identity.

The project saves:

- observed action entropy,
- `p_t`,
- proposal-predicted entropy,
- asymptotic entropy approximation,
- absolute errors,
- RMSE for both approximations.

## Important configuration

All experiment-level parameters are defined in `config.yaml`.

The belief observer section is:

```yaml
belief_observer:
  alpha_prior: 1.0
  beta_prior: 1.0
  optimal_arm_probability_samples: 128
  confident_entropy_threshold_normalized: 0.20
  wrong_correctness_threshold: 0.20
```

The proposal-metric numerical settings are:

```yaml
research_metrics:
  policy_kl_probability_floor: 1.0e-12
  entropy_efficiency_minimum_reduction: 1.0e-12
```

`optimal_arm_probability_samples` controls Monte Carlo accuracy for `q_t`.
Increasing it reduces Monte Carlo noise but increases runtime.

The correctness metrics currently require a unique true optimal arm.

## Installation

From the project directory:

```bash
python -m venv .venv
```

Windows CMD:

```bat
.venv\Scripts\activate
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Run tests

```bash
pytest -q
```

The current codebase includes sanity tests for the algorithms, entropy metrics,
belief metrics, experiment runner, proposal metrics, KL support changes, and
Pareto dominance.

## Run the experiment

```bash
python main.py
```

For a quick sanity run, temporarily use something like:

```yaml
experiment:
  horizon: 200
  n_runs: 20
```

For larger runs, remember that the belief observer performs posterior Monte
Carlo at every time step for every run and algorithm, so runtime grows
substantially.

## Output structure

All outputs stay under one experiment directory:

```text
results/
└── experiment/
    ├── config.yaml
    ├── algorithms/
    │   ├── greedy/
    │   │   ├── data/
    │   │   │   ├── all_records.csv
    │   │   │   └── summary.csv
    │   │   └── plots/
    │   │       ├── regret.png
    │   │       ├── entropy_decomposition.png
    │   │       ├── entropy_normalized.png
    │   │       ├── belief_analysis.png
    │   │       ├── policy_vs_belief_entropy.png
    │   │       ├── policy_belief_mismatch.png
    │   │       ├── belief_regret_phase.png
    │   │       ├── entropy_reduction_efficiency.png
    │   │       ├── entropy_dissipation.png
    │   │       ├── action_entropy_regret_phase.png
    │   │       ├── policy_kl_geometry.png
    │   │       └── entropy_misidentification_theory.png
    │   └── ...
    └── comparisons/
        ├── data/
        │   └── final_algorithm_comparison.csv
        └── plots/
            ├── regret_comparison.png
            ├── marginal_entropy_comparison.png
            ├── belief_entropy_comparison.png
            ├── belief_correctness_comparison.png
            ├── policy_belief_mismatch_comparison.png
            ├── entropy_efficiency_comparison.png
            ├── entropy_dissipation_comparison.png
            ├── action_entropy_regret_phase_comparison.png
            ├── policy_kl_path_comparison.png
            └── entropy_regret_pareto_frontier.png
```

## New columns in each algorithm's `summary.csv`

The time-resolved proposal metrics include:

- `marginal_entropy_reduction`
- `normalized_marginal_entropy_reduction`
- `marginal_entropy_dissipation_rate`
- `normalized_marginal_entropy_dissipation_rate`
- `regret_per_entropy_reduction`
- `action_misidentification_probability`
- `proposal_predicted_action_entropy`
- `asymptotic_predicted_action_entropy`
- `proposal_entropy_absolute_error`
- `asymptotic_entropy_absolute_error`
- `mean_policy_step_kl_divergence`
- `cumulative_mean_policy_kl_path_length`
- `infinite_policy_step_kl_fraction`
- `mean_smoothed_policy_step_kl_divergence`
- `std_smoothed_policy_step_kl_divergence`
- `cumulative_mean_smoothed_policy_kl_path_length`

## Interpretation reminders

- Low policy entropy means deterministic/concentrated behavior; it does not
  imply that the algorithm knows the correct arm.
- Low belief entropy means posterior concentration; it does not by itself imply
  correctness.
- `belief_correctness` distinguishes confident-and-correct from
  confident-but-wrong states.
- Policy-belief mismatch is descriptive. UCB, Greedy, Epsilon-Greedy, and EXP3
  are not designed to probability-match the Bayesian observer.
- The common observer is an external evaluator for non-Bayesian algorithms,
  not part of their decision rules.
- A negative entropy dissipation value is not an error; it means entropy rose
  from the previous step.
- Exact KL can legitimately become infinite when a deterministic policy changes
  support. Use the smoothed KL outputs for finite cross-algorithm plots, while
  preserving exact KL for the mathematical diagnostic.
- `p_t log(1/p_t)` is an asymptotic expression. It should not be expected to
  match well when `p_t` is not yet small.

## Not implemented yet

The next stage can add:

- KL-UCB,
- sliding-window UCB,
- nonstationary change-point environments and entropy reheating experiments,
- difficulty sweeps over reward gaps and number of arms,
- expected information gain,
- Information-Directed Sampling,
- deeper theoretical asymptotics and statistical confidence analysis.


## Plot cleanup in this revision

- Experiment defaults are now `n_runs: 100` and `horizon: 1000`.
- The marginal-MAP correctness line was removed from the marginal-action analysis plot because it collapses the across-run distribution to a single 0/1 mode indicator and was visually uninformative in the baseline. The underlying metric remains in CSV outputs for reproducibility.
- The regret / entropy-reduction ratio is omitted while marginal entropy reduction is `<= 0.05` nats to avoid unstable near-zero denominators; this rule is stated directly inside the efficiency figure.
- Duplicate belief-regret and marginal/action-regret phase figures were removed. The three proposal phase diagrams now have one consistent naming family: policy, belief, and marginal entropy versus cumulative regret.
- Raw and normalized action-entropy plots now use matching filenames: `action_entropy_decomposition_nats.png` and `action_entropy_decomposition_normalized.png`.
- Phase-diagram start and end points are plotted separately and labeled `Start (t=1)` and `End (t=T)`.
- Plot filenames, titles, legends, and axis labels were standardized for easier navigation.
- `clear_previous_results: true` is enabled by default so old plot filenames do not remain after rerunning the experiment.

## UCB1 entropy-reduction reference
For UCB1, the first K pulls are deterministic forced initialization. The action-entropy reduction metric therefore uses the first genuine UCB decision (t = K + 1) as its reference instead of t = 1. Pre-reference reduction/efficiency values are saved as NaN. Other algorithms keep t = 1 as the reference. Regret-per-entropy-reduction for UCB1 is also measured from the same post-warm-up reference.
