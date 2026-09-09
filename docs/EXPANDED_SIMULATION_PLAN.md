# Expanded simulation batch v1

Status: completed. See [results](EXPANDED_SIMULATION_RESULTS.md).

Frozen before execution, following the initial feasibility pilot. This batch
examines endpoint feasibility and paired variability; it is not a confirmatory
methodology verdict. Financial datasets remain deferred.

- Sample sizes: 300, 1,000 and 3,000.
- Ten new seeds: integers 101 through 110, with every seed evaluated at every
  design and budget. Initial pilot seeds 11 and 29 are excluded.
- Correlations: 0.5 and 0.9. Signal-to-noise ratios: 0.25 and 2.0.
- Gated contexts: 2 and 4. Three marginal shadows, q=0.9, one importance split.
- Original learner, nested resampling, candidate sizes, and endpoint constraints
  remain fixed: scaled Ridge(alpha=1); three outer and two inner folds;
  k in {1,2,3,4,6}; absolute MSE margin 0.05, maximum size 6, minimum recall
  improvement 0.05. Ground-truth classes and null construction are unchanged.
- 120 generated datasets and 240 paired comparisons (480 nested evaluations).
  Four independent seed processes, one BLAS thread per process. Runtime depends
  on concurrent CPU contention and should not be directly compared with the
  serial pilot's wall-clock timings; measured fit counts remain comparable.

## Reproduce

```sh
uv sync --frozen --all-extras
uv run python experiments/run_simulation_batch.py
uv run python experiments/analyze_simulation_batch.py
```

`experiments/expanded-simulation.json` fixes the entire grid. Outputs are stored
under `results/expanded-simulation-v1/`, with per-seed configurations, logs,
provenance and cell checkpoints. Repeating the driver resumes completed cells.
The batch archives a checksum-verified source snapshot because the underlying
Git checkout may contain uncommitted work. No public data are loaded or refitted.

## Analysis fixed before viewing results

Report each (n, correlation, SNR, budget) separately, with all ten seeds accounted
for. Count compute matches, valid endpoint comparisons, and failure reasons.
Report per-arm constraint pass rates and paired loss/size differences over all
seeds. Compute paired class-recall differences only as qualified endpoint
differences when both arms satisfy constraints and compute is matched.

For valid comparisons, report the conditional mean paired recall difference,
its Monte Carlo standard error and a descriptive 95% Student-t interval when
there are at least two valid seeds. These intervals condition on validity; they
are not unconditional effects, multiplicity-adjusted tests, or permission to
discard failed comparisons. With zero valid seeds the conditional mean is
unavailable; with one seed uncertainty is unavailable. Fold-level observations
are never treated as independent simulation replications. Also show raw recall
differences across all seeds as diagnostics, explicitly separate from endpoints.

Report how many valid seeds clear the fixed +0.05 effect. A single favorable
seed or budget cannot establish a persistent advantage. No margins, seeds,
sample sizes or budgets are changed in response to intermediate results. The
per-run decision helper is not applied to a pooled table of seeds: it expects
budgets, not a study-level repeated-simulation inference design.
