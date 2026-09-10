# Public application results

Completed 2026-09-10 using the frozen [protocol](PUBLIC_APPLICATION_PLAN.md) and [configuration](../experiments/public-application.json), from commit `14de93d`. All 30 dataset/subset/seed designs and 180 method evaluations completed without failures or warnings. The run counted **644,782 model fits**, took **83.18 worker-minutes** (1.39 hours), and reached **0.231 GiB** peak process memory. Original worker invocations include loading/startup; resume and analysis time are excluded. The eight-worker-hour cap was respected.

All 60 gated/ungated comparisons met the prespecified 5% measured-fit tolerance; maximum imbalance was **1.85%**. Counts and results are descriptive, not independent population replications.

## Findings

The application study does not show a consistent predictive advantage for probe gating. Across the 60 matched comparisons, gated loss was lower in 19, higher in 21, and exactly tied in 20. The 20 ties are all Superconductivity comparisons, where every primary method retained all 81 features in every outer fold.

- **Superconductivity:** every primary method reproduced the all-feature loss and selected size, while spending thousands of additional fits. L1 retained most features and had slightly higher mean MSE. The all-feature baseline provides the same predictive result as the primary methods at three counted fits per design.
- **MiniBooNE:** the all-feature model had the lowest mean log loss at both sample sizes. Gated versus ungated ordering changed with sample size and budget. Selection reduced feature count, with higher mean loss than the all-feature reference.
- **Year Prediction:** L1 had the lowest mean MSE at both sample sizes, closely followed by the all-feature reference. At n=5,000 and T=1, gating beat matched ungated selection on all five seeds (mean MSE difference -1.2962; range -3.1679 to -0.0557), but selected 48.27 features versus 20.27. Its mean MSE, 96.6498, remained above L1 (93.9695) and all features (94.0461). The T=2 comparison favored ungated selection on average.

## Mean prediction loss

Each entry averages five prespecified seeds. Lower is better. Superconductivity and Year Prediction use MSE; MiniBooNE uses log loss. T identifies the gated context budget; its ungated comparator receives the matched context allocation. Loss units must not be pooled across datasets.

| Dataset | Rows | All features | L1 | Gated T=1 | Ungated match | Gated T=2 | Ungated match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Superconductivity | 3000 | 323.3928 | 323.6472 | 323.3928 | 323.3928 | 323.3928 | 323.3928 |
| Superconductivity | 5000 | 319.0701 | 319.1180 | 319.0701 | 319.0701 | 319.0701 | 319.0701 |
| MiniBooNE | 3000 | 0.244285 | 0.245156 | 0.254740 | 0.251637 | 0.256821 | 0.253199 |
| MiniBooNE | 5000 | 0.237105 | 0.238426 | 0.240686 | 0.242726 | 0.240871 | 0.240897 |
| Year Prediction | 3000 | 100.3434 | 99.6657 | 105.0797 | 103.1490 | 105.1374 | 103.9575 |
| Year Prediction | 5000 | 94.0461 | 93.9695 | 96.6498 | 97.9460 | 98.4079 | 97.7380 |

## Mean selected feature count

| Dataset | Rows | All features | L1 | Gated T=1 | Ungated match | Gated T=2 | Ungated match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Superconductivity | 3000 | 81.00 | 76.07 | 81.00 | 81.00 | 81.00 | 81.00 |
| Superconductivity | 5000 | 81.00 | 77.60 | 81.00 | 81.00 | 81.00 | 81.00 |
| MiniBooNE | 3000 | 50.00 | 39.27 | 27.87 | 28.00 | 26.27 | 23.07 |
| MiniBooNE | 5000 | 50.00 | 42.53 | 38.13 | 33.47 | 35.73 | 35.73 |
| Year Prediction | 3000 | 90.00 | 60.87 | 14.93 | 12.80 | 10.93 | 14.40 |
| Year Prediction | 5000 | 90.00 | 65.60 | 48.27 | 20.27 | 25.73 | 27.47 |

## Cost and interpretation

The all-feature baseline uses three model fits per design; L1 uses 96, including inner selector fits, prediction fits and outer refits. Primary methods use roughly 2,400–8,700 fits. These are nested-evaluation costs, not deployment prediction latencies. L1 is not uniformly faster in wall time: Superconductivity required roughly 51–78 seconds per L1 design despite its lower fit count. The tables retain measured method time separately from total worker time.

The observed Year Prediction T=1 gain is an accuracy/size tradeoff against the matched comparator, not broad superiority over the external references. At every dataset/sample-size/budget combination, the gated method's mean loss was no better than the all-feature mean (ties on Superconductivity). A smaller selected set can still matter for an application, but no new post-hoc loss margin is used to declare it acceptable.

Seeds and subset sizes can reuse original rows, and budgets reuse the same designs. The five-seed means and ranges describe this benchmark; they do not establish statistical significance, population equivalence, causal effects, or known-truth feature recovery. Trivial all-feature set stability is not evidence of selection correctness. Conclusions apply to these fixed Ridge/logistic learners and sample sizes.

Year Prediction used only its official development partition. Its reserved 51,630 test rows were not evaluated. A later official-split evaluation requires a separately frozen final protocol. Financial datasets remain deferred.

## Validation and reproduction

The analyzer validated the complete grid, checkpoint identities, original row IDs, fold partitions, specifications and fold-derived aggregates. All 360 primary-method outer reference losses exactly matched the corresponding independently counted all-feature baseline losses. Resume preserved all 180 method checkpoints and 30 shared data/fold records in bytes and modification times.

All 137 tests, Ruff and source/wheel builds passed. A verified extraction reproduced the method, contrast and seed-summary reports exactly, with no accepted float differences. The evidence archive contains 252 files and is approximately 5 MB.

The evidence archive includes all checkpoints, data/fold records, frozen configuration, source snapshot, runtime ledger, analysis tables and resume verification. It excludes raw public-data archives. The original methodology and calibration evidence remain separate and unchanged.

```sh
uv sync --locked --all-extras
uv run python experiments/reproduce_public_results.py --output artifacts/public-results-review
```

This verifies archive/member checksums, regenerates all tables and figures without refitting, and checks the three numerical JSON reports against recorded values. Finite float roundoff uses the existing declared tolerance (rtol=1e-12, atol=1e-14), with every accepted difference logged; counts and structure remain exact. No data downloads or account are needed for reanalysis.

- [Per-method results](../experiments/reports/public-application-v1/methods.csv), [paired contrasts](../experiments/reports/public-application-v1/contrasts.csv), [seed summaries](../experiments/reports/public-application-v1/summary.json), and [analysis provenance](../experiments/reports/public-application-v1/analysis-provenance.json).
- Superconductivity plots: [3,000 rows](figures/public-superconductivity-n3000.png), [5,000 rows](figures/public-superconductivity-n5000.png).
- MiniBooNE plots: [3,000 rows](figures/public-miniboone-n3000.png), [5,000 rows](figures/public-miniboone-n5000.png).
- Year Prediction plots: [3,000 rows](figures/public-yearpredictionmsd-n3000.png), [5,000 rows](figures/public-yearpredictionmsd-n5000.png).
