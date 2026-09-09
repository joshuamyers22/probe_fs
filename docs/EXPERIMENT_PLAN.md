# Reproducible experiment pilot

Status: initial pilot completed; see [results](PILOT_RESULTS.md). The pilot
configuration was frozen before examining results. Financial datasets
are deferred. This pilot measures feasibility and runtime; it is not the primary
confirmatory study and cannot trigger its discontinuation rule.

## Run from the repository root

```sh
uv sync --frozen --all-extras
uv run python -m pgfs.datasets
uv run python -m pgfs.study --config experiments/pilot.json --output results/pilot-v1
```

The download command retrieves about 235 MB of compressed public archives,
verifies SHA-256 against `experiments/datasets.json`, and caches them in ignored
`data/raw/`. No account or API key is required. The UCI catalog lists all three
datasets under CC BY 4.0; cite the dataset authors and DOIs in the manifest.
Hashes pin bytes downloaded on 2026-09-08 (local date); an upstream change fails
verification rather than silently changing the benchmark.

The study checkpoints each design/seed/budget pair in strict JSON. Repeat the
same command to resume. Changes to configuration, package versions, source code,
or dependency lock require a new output directory. Source hashes supplement the
Git commit because a run may use uncommitted code. Run only one writer per output
directory. `--only simulation` and `--only public` can run the stages separately.

## Frozen pilot design

- Simulations: n=300; within-class correlation 0.5 and 0.9; signal-to-noise ratios
  0.25 and 2; seeds 11 and 29. Three classes with two features each, three
  correlated nulls and six independent nulls. Eight generated datasets total.
- Gated budgets: 2 and 4 contexts; three shadows, q=0.9, one importance split.
  Ungated context counts come from existing cache-aware matched-fit accounting.
- Both arms use the same nested folds: three outer, two inner. Regression uses
  median imputation, standard scaling and Ridge(alpha=1), each fitted inside the
  individual training partition. Candidate sizes: 1, 2, 3, 4, 6.
- Simulation endpoint: class recall subject to absolute MSE margin 0.05 against
  all features and maximum size 6; minimum recall difference 0.05. Preserve
  constraint failures and compute mismatch as inconclusive comparisons.
- Public datasets: 600 development rows each, seed 11, gated contexts 1 and 2;
  candidate sizes 1, 2, 4, 8, 16, 32. The same regression pipeline is used for
  regression; MiniBooNE uses LogisticRegression(C=1, liblinear, max_iter=1000).
  BLAS thread count is limited to one. This is a learner-specific pilot, not a
  comprehensive learner comparison.

## Public dataset boundaries

1. [Superconductivity](https://archive.ics.uci.edu/dataset/464/superconductivty+data):
   21,263 original rows, 81 predictors, target critical temperature. Keep the
   first original row per exact formula before seeded sampling. This prevents
   identical formulas spanning folds, but does not establish independence of
   chemically related compounds or equivalent alternative formula strings.
2. [MiniBooNE](https://archive.ics.uci.edu/dataset/199/miniboone+particle+identification):
   the raw header specifies 36,499 signal and 93,565 background events: 130,064
   observations with 50 predictors. The catalog's 130,065 includes one extra row
   relative to the actual event counts. Samples are stratified. Values of -999
   are treated as missing detector quantities and imputed within each model fit.
3. [Year Prediction MSD](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd):
   515,345 rows, 90 predictors. Sample only from the official first 463,715
   development rows; the final 51,630 rows remain untouched. Pilot CV within the
   development partition is not an artist-disjoint evaluation: artist IDs are
   unavailable in this file. The official external split is required for the
   eventual final reported benchmark.

Public results report predictive loss, all-feature reference loss, size,
stability, fit count and runtime. They have no known signal classes and receive
no class recall, false-selection endpoint, or primary-study decision verdict.

## Outputs and interpretation

`REPORT.md`, `results.csv`, `aggregates.json`, `performance-vs-compute.png`,
`provenance.json`, and individual `cell-*.json` are written under the output
directory. Cell files contain selected row IDs, method specifications, per-fold
selected sets, rankings, scores and separate split/context diagnostics. Plots
average seeds only within the same simulation design. With only two simulation
seeds, uncertainty estimates are descriptive; fold SEs are not independent
replication uncertainty. One importance split cannot estimate between-split
variability. Timing is hardware-dependent and includes both total cell elapsed
time and the existing model fit/predict timing counter.

## After the pilot

Inspect runtime, numerical failures, compute matching and endpoint feasibility.
Freeze a separate primary grid with more seeds and sample sizes before running
it. Evaluate the Section 16 criteria on that study; proceed to gated LOCO/global
shadow ablations, external baselines and sensitivity analyses according to the
specification. Public experiments are an application extension, not a substitute
for known-truth simulations. Increasing public sample sizes and scoring the
reserved Year Prediction test set require a separately frozen final protocol.
