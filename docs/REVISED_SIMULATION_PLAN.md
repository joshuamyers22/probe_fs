# Revised noisy-proxy benchmark v1

Protocol specified before generating any revised-benchmark outcomes. This is a
new exploratory benchmark informed by the completed endpoint audit; it does not
replace the original negative and inconclusive findings.

The estimand is coverage of noisy-proxy signal groups. Members of a class provide
complementary measurements of one latent Gaussian driver, not exact substitutes.
Keep the original generator and use class coefficients **(1.0, 0.30, 0.15)**.
These coefficients are selected from population risk calculations, without
inspecting fitted outcomes on the new seeds. The runner now forwards explicit
coefficients to the generator; omitted coefficients retain the original defaults.

For r measurements from a class with coefficient b and correlation rho, residual
signal variance is b²(1-rho)/(1-rho+r*rho). Subtracting the sum for two measurements
per class gives excess population MSE relative to all six signal measurements.
Independent null features and outcome noise do not change this population gap.

| Selection counts by class | Recall | Excess MSE, rho=0.5 | Excess MSE, rho=0.9 |
| --- | --- | --- | --- |
| (2, 1, 0) | 2/3 | 0.030000 | 0.025579 |
| (2, 1, 1) | 1 | 0.018750 | 0.005329 |
| (2, 0, 0) | 1/3 | 0.075000 | 0.106579 |

Thus partial and full coverage can satisfy the unchanged 0.05 margin at sizes
3 and 4; selecting only the strong class cannot. These are feasibility examples,
not oracle-selected inputs to either method. Finite-sample Ridge and estimated
rankings need not attain the population bounds. Lower SNR increases noise and does
not reduce these absolute population gaps.

Freeze the following design in `experiments/revised-simulation.json`:

- Fresh seeds 201–210, disjoint from pilot 11/29 and expanded study 101–110.
- n = 300, 1,000, 3,000; rho = 0.5, 0.9; SNR = 0.25, 2; gated contexts = 2, 4.
- Three signal classes, two proxies each; one three-feature correlated null block
  (rho=0.6) and six independent nulls; 15 features total.
- Complete candidate sizes 1–6, including the previously omitted size 5.
- Existing one-standard-error size selection, Ridge alpha=1, within-fold median
  imputation/scaling, three outer folds and two inner folds.
- One importance split, three marginal shadows, shadow quantile 0.9; existing
  expected-fit matching and measured-fit tolerance of 5%.
- Endpoint: both arms satisfy mean outer MSE <= their all-feature reference +0.05,
  with size cap 6. Minimum absolute recall effect remains +0.05.

Run all **240 paired comparisons**, with four seed workers and one BLAS thread per
worker. No early stopping, seed exclusion, margin changes, coefficient changes,
or selection-policy changes based on these outcomes. Analyze validity for all
seeds, qualified paired recall only for valid endpoints, and paired loss/size for
all seeds. Report each design separately with descriptive across-seed MCSE and
95% t intervals. Conditional intervals are not a confirmatory test, and design
groups reuse datasets across budgets. Ten seeds cannot establish a small effect.
No argmin or fixed-size sensitivity is part of this batch.

Reproduce from the repository root with locked dependencies:

```sh
uv sync --locked --all-extras
uv run python experiments/check_simulation_feasibility.py --config experiments/revised-simulation.json --output experiments/reports/revised-feasibility.json
uv run python experiments/run_simulation_batch.py --config experiments/revised-simulation.json --output results/revised-simulation-v1
uv run python experiments/analyze_simulation_batch.py --output results/revised-simulation-v1
```

Configuration, source hashes, dependency lock, versions, source ZIP, seed shards,
rankings, fold results and paired summaries are retained in the run directory.
The config includes hashes of this protocol, the preflight script and its report.
Those three files are also copied into the run directory before launch. Results
directories are ignored by Git; portable summaries and the plot are stored in
`experiments/reports` and `docs/figures`.

The core runner change gives this study a new source identity. Earlier results
remain unchanged; resuming them requires the source/dependency snapshot stored
with their original batch, rather than today's modified working tree.
