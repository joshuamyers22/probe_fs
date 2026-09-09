# Larger public-data application study

Status: development design frozen on 2026-09-09 after baseline validation and
cost calibration. No scientific application grid has run. The
[executable configuration](../experiments/public-application.json) pins this
protocol, the passing preflight and the implementation by SHA-256. Calibration
outcomes are excluded from scientific comparisons. Changes require a separately
documented protocol/configuration before fitting.

## Question and scope

Does probe gating offer a useful prediction-loss, selected-size, stability and
compute tradeoff on public regression and classification problems? The completed
simulations do not establish persistent gating superiority. Public data provide
no known-truth class-recall or false-selection endpoint, and this application
study will not assign either. Financial datasets remain deferred.

Use the three existing archives and checksums in
[`experiments/datasets.json`](../experiments/datasets.json). Downloads require no
account or API key. Preserve the dataset attribution and preprocessing boundaries
documented in the [pilot protocol](EXPERIMENT_PLAN.md).

## Frozen development design

| Item | Specification |
| --- | --- |
| Datasets | Superconductivity, MiniBooNE, Year Prediction MSD |
| Subset sizes | 3,000 and 5,000 development rows per dataset |
| Replication seeds | 501, 502, 503, 504, 505 |
| Outer / inner folds | Three / three; identical outer folds across methods |
| Primary gated budgets | One and two partial contexts; three shadows, quantile 0.9, one importance split |
| Matched comparator | Ungated partial contexts, allocated by existing cache-aware expected-fit accounting; report observed mismatch |
| Size path | 1, 2, 4, 8, 16, 32, and all available features, deduplicated |
| Size choice | One-SE rule using inner-fold losses only |
| Prediction learners | Ridge(alpha=1) for regression; logistic regression(C=1, liblinear, max_iter=1000) for classification |
| Preprocessing | Median imputation and standard scaling fitted within every training partition |
| Runtime controls | One BLAS thread; one isolated worker at a time |

This gives 30 dataset/subset/seed designs and 60 primary gated/ungated contrasts.
Within each design, every method receives identical original row IDs and folds.
Keep current deterministic sampling: independently draw each requested subset
with its seed and retain original row IDs. Do not assume the two subset sizes
are nested. Seeds can reuse source rows; they measure split/subsample sensitivity,
not independent samples of new populations.

Superconductivity keeps the first original row per exact formula before sampling.
MiniBooNE uses stratified subsets/folds and maps -999 to missing values before
fold-local imputation. Year Prediction samples only the first 463,715 development
rows. Its final 51,630 rows remain excluded from this phase, including tuning,
feature selection, preprocessing and cost calibration.

## External references

Evaluate the all-feature learner and an L1 selector on each of the 30 designs.
The L1 selector uses inner-CV loss of the final prediction learner to choose its
regularization, then refits on each outer training set. Every inner-training fold
gets fresh imputation, scaling, L1 selection and prediction fitting. Regression
uses Lasso alpha; classification uses L1 logistic regression with C=1/penalty.
The decreasing penalty grid is [10, 1, 0.1, 0.01, 0.001]. Exact mean-loss ties
choose the strongest penalty. Coefficients with absolute magnitude above 1e-10
select a feature; an empty set uses an intercept-only prediction model. Maximum
iterations are 200,000 and solver tolerance is 1e-4. Convergence warnings fail the
method and are retained in its checkpoint. This is a minimum application
comparison, not the full Section 12 baseline suite.

Report these references at their actual cost. Count selector fitting and tuning
as well as final prediction fits; do not pad cheap methods to match PGFS. Keep the
two primary budgets separate, and do not treat a shared all-feature reference as
an independent result for each budget.

## Analysis and held-out evaluation

Report MSE for regression and log loss for classification, with paired differences
from ungated and all-feature references. Also report mean selected size, selected
feature fraction, within-design outer-fold set Jaccard, model fits and elapsed
time. Plot loss against size and measured compute separately for each dataset and
subset size. No pooled cross-dataset loss score or post-hoc margin is primary.

Keep every prespecified seed, numerical failure and compute mismatch visible.
Summarize paired differences across the five seeds with individual points, mean,
standard deviation and range. Fold errors are not independent seed replication;
overlapping samples preclude treating descriptive intervals as population-level
confidence claims. Feature stability alone is not a measure of correctness.

The reserved Year Prediction test set belongs to a later, separately frozen
evaluation. Before opening it, record the final development training size, method
list, tuning rules, seeds, metrics, and a single evaluation command. All reported
test outcomes must then be retained; do not choose a method or revise settings
from that test. The development phase alone is not a final official-split result.

## Implementation and compute gate

1. Add an application runner that checkpoints each method and records original
   row IDs, folds, rankings, selected sets, source/driver hashes, versions and full
   compute accounting. Resume must reject changed identities and preserve cells.
2. Implement a fold-local L1 adapter with counted tuning fits. The existing
   `select_learner_native` scales before its internal CV and does not impute
   MiniBooNE missing values; `run_baseline_nested` counts final evaluations but
   not selector fitting. Do not use those helpers unchanged for this study.
3. Test training-only preprocessing/selection, shared folds, missing values,
   selector fit counts, identity rejection and unchanged resume. Test that no
   Year Prediction official test rows reach development code.
4. Run a separately labelled infrastructure/cost calibration on at most 600 rows
   per dataset with seed 599, excluded from reported scientific comparisons.
   Record fits, memory and elapsed time; inspect numerical failures, not method
   rankings, when deciding whether the planned grid is practical.
5. Project the complete grid cost before freezing. Default cap: eight total
   worker-hours and 8 GiB memory per worker. If projections exceed either cap,
   revise and document the grid before scientific fitting. Runtime exhaustion
   pauses the run with its checkpoints intact; it does not select a subset of
   favorable completed outcomes for reporting.
6. Freeze the final executable config and hashes, then run the development grid.
   Reanalyze from checkpoints in a clean workspace and add the complete results
   to a new version of the reproducibility package.

These prerequisites are implemented and the second calibration passes. The
initial 600-row calibration recorded one L1 convergence failure at 20,000
iterations and projected 10.52 worker-hours for the proposed 3,000/10,000-row
grid, exceeding the eight-hour cap. Before scientific fitting, the larger subset
was reduced to 5,000 rows and the solver allowance increased to 200,000 iterations.
All five replication seeds, penalties and outcome rules were retained. The
second calibration completed all 18 methods without warnings or failures and
projects 6.83 worker-hours for the revised grid. These choices used cost and
convergence diagnostics, not comparative predictive performance.

The parent enforces the remaining worker-time cap with a subprocess timeout.
Process peak memory is recorded and checked between methods; the 8 GiB threshold
is not a hard operating-system memory limit within an individual fit. Projections
use twice linear row scaling of observed worker time and are estimates, not
guarantees. See [calibration results](PUBLIC_CALIBRATION_RESULTS.md).

## Commands

Use the locked environment on Linux or macOS (or Linux under WSL), then download
the pinned archives if needed:

```sh
uv sync --locked --all-extras
uv run python -m pgfs.datasets
uv run python experiments/run_public_application.py --output results/public-application-v1
uv run python experiments/analyze_public_application.py --output results/public-application-v1 --report artifacts/public-application-v1
```

Repeat the run command to resume unchanged checkpoints. Failed methods remain
recorded; correcting their settings requires a new version and output directory.
The analyzer checks completeness, source/row/fold identities and method records,
and retains failures in tables and denominators. Plots show individual seeds and
means, with actual completed/planned counts. The final grid is 180 method records:
30 designs times four primary arms plus two external references.

To reproduce only the cost evidence without fitting or data downloads:

```sh
uv run python experiments/reproduce_public_calibration.py --output artifacts/public-cost-review
```
