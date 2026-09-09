# Initial experiment results

Completed September 8, 2026 (local date), using the frozen
[pilot configuration](../experiments/pilot.json) and
[dataset manifest](../experiments/datasets.json).

22 paired comparisons completed: 16 simulation comparisons and six public-data
comparisons, totaling 44 nested evaluations and 100,671 model fits. Total cell
elapsed time was 212.8 seconds on this machine, excluding setup and plotting.
Every comparison met the 5% measured-fit tolerance; maximum imbalance was 2.45%.
All 22 checkpoints were then resumed without modifying or refitting any cell.

## Simulation findings

These are feasibility results, not a confirmatory study. Of 16 comparisons,
15 were inconclusive because one or both arms failed the prespecified endpoint
constraints. The one valid comparison (n=300, rho=0.9, SNR=2, seed=29, four gated
contexts) had class recall 1.0 for both arms, a difference of zero.

The pilot therefore does not establish a gating advantage and does not provide
a basis for the broad study's stopping decision. Raw recall curves include
constraint-failing runs for diagnostics and must not be read as qualified primary
endpoint curves. No loss margins, size limits or seeds were changed after viewing
results.

## Public-data findings

Each dataset used 600 development observations and seed 11, with three outer and
two inner folds. Lower loss is better. These runs assess feasibility; one subset
and seed are insufficient for a method comparison across the full populations.

| Dataset | Gated contexts | Gated loss | Ungated loss | All-feature reference loss | Mean size gated / ungated |
| --- | --- | --- | --- | --- | --- |
| Superconductivity (MSE) | 1 | 420.0294 | 400.5619 | 360.3549 | 21.33 / 32.00 |
| Superconductivity (MSE) | 2 | 405.4891 | 412.9314 | 360.3549 | 26.67 / 26.67 |
| MiniBooNE (log loss) | 1 | 0.30872 | 0.29885 | 0.29433 | 9.33 / 8.00 |
| MiniBooNE (log loss) | 2 | 0.30389 | 0.30528 | 0.29433 | 8.00 / 10.67 |
| Year Prediction (MSE) | 1 | 100.4149 | 102.4083 | 117.3711 | 1.00 / 2.00 |
| Year Prediction (MSE) | 2 | 100.4144 | 100.4149 | 117.3711 | 1.33 / 1.00 |

Gated versus ungated ordering varies with budget. The full-feature learner has
lower loss than either selected model on the Superconductivity and MiniBooNE
subsets. Both selection methods outperform that reference on the Year Prediction
subset. No ground-truth recall or false-selection endpoints are assigned to these
datasets. Year Prediction's official 51,630-row test partition remains untouched.

Superconductivity retained 15,542 distinct exact formula strings before sampling.
The MiniBooNE sample contained 50 values of -999; the loader mapped them to missing
values and the learner imputed them within each training partition.

## Implementation finding and validation

The preflight check exposed cancellation in the existing expected-fit calculation
at wider feature counts. Replacing `1 - exp(x)` with `-expm1(x)` restored the exact
single-context expectation of p+1 and corrected public-data compute allocation.
All experiments above used the corrected accounting. Added tests cover wide
budgets against a high-precision oracle, archive integrity, dataset labels,
reserved test boundaries, duplicate-formula handling, absent real-data ground
truth, strict JSON, and safe checkpoint resumption.

Lint, all 61 tests, and source/wheel builds passed. The existing knockoff tests
emit two scikit-learn deprecation warnings concerning `n_alphas`.

## Artifacts and next experiment

Full local outputs are in `results/pilot-v1/`: provenance, cell JSON with original
row IDs and per-fold diagnostics, CSV, aggregates, a report, PNG curves, and a
resume log. Data and generated results are ignored by Git; the configuration,
manifest, runner and this compact summary are intended for version control.

Next, freeze a larger simulation batch with additional seeds and sample sizes to
estimate endpoint feasibility and paired variability under the original
constraints. The pilot is small enough that the confirmed fit counts can guide
the compute budget. Expand the public subsets after that protocol is fixed;
retain the official Year Prediction test partition for final evaluation. The
gated LOCO ablation and external selector comparisons remain pending.
