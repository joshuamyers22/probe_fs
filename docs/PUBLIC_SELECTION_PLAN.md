# Public selection-rule sensitivity

Protocol specified on 2026-09-10, after the complete public application study and before any new public-data refits. This is an exploratory paired intervention on the same development data, not independent confirmation or a change to the primary method.

## Question and fixed design

Does choosing the candidate size with minimum inner-validation loss reduce the predictive loss caused by the one-standard-error size rule, and what additional feature count does it require? Analyze all three datasets, both subset sizes (3,000 and 5,000), all five seeds (501–505), both budgets (T=1,2), and both gated and matched ungated methods. That gives 120 method-level paired comparisons and 360 paired outer folds. Include the original all-feature and L1 references in every comparison. Do not select favorable designs or introduce a post-hoc acceptable-loss margin.

Inherit the original pinned datasets, row order, train/validation/test partitions, learners, preprocessing, candidate paths, random seeds, ranking scores and tie order. Only the size rule changes: argmin chooses the first minimum of the ordered candidate path. One-SE remains the primary published method. Year Prediction's reserved official test partition is excluded. Financial data and artificial-corruption stress tests are separate later studies.

## Exact replay and validation

The checksummed public evidence already contains each outer-training ranking, the complete inner-validation mean-loss path, the selected size and the inner standard error at the minimum. Neither ranking estimation nor the candidate loss path depends on the size rule. Reuse these training-only artifacts; recompute argmin from the path and validate the recorded argmin and one-SE choices. Do not choose sizes from outer losses.

Refit both the recorded one-SE set and the argmin set independently on each original outer-training partition, using the original final-fit seed. Recompute preprocessing on training rows only. Verify that the one-SE predictions reproduce their recorded losses (rtol=1e-12, atol=1e-14; log every nonzero accepted difference). Validate original row IDs, feature names, X/y hashes and every nested partition before fitting. Confirm each one-SE set is the prefix of its argmin set and all-feature selections reproduce the original all-feature reference.

Before public refits, test replay against full nested evaluations under both rules, for regression and classification, with gated and ungated rankings. Test that changing outer holdout labels cannot change selected sets, tampered selection paths are rejected, and resume rejects changed identities. Freeze source, test, protocol, baseline archive and configuration identities before execution.

## Cost and failures

Exactly 720 new outer model fits are planned: 120 methods x three folds x two policies. No new ranking, shadow, inner-path or L1 fits are needed. Record incremental actual fit counts, fit/predict time, worker wall time and peak memory separately. The original 644,782-fit study remains the prerequisite evidence; replay cost is not the standalone cost of either selector. A standalone rerun has the same fit-count schedule as its original method but its runtime is not measured here.

Use one worker and one BLAS thread, with a one-worker-hour execution cap and an 8 GiB memory cap checked between methods. The original public fits and lightweight replay equivalence checks provide the cost basis; no new method parameters require convergence calibration. Treat convergence warnings as failures. Retain failed checkpoints and explicit planned/complete denominators. Any interrupted method is rerun as a whole; preserve completed cells on resume.

## Analysis fixed before refits

Report per-method paired argmin-minus-one-SE differences in outer loss, selected count and cross-fold Jaccard stability. For every dataset/size/arm/budget show all five seeds, their mean and range, and lower/higher/exact-tie counts. Report fold-level changed sizes and losses, and distances to all-feature and L1 references under each policy. Report the paired change in gated-minus-ungated loss to distinguish a general size-rule benefit from a gating benefit.

Do not pool MSE and log-loss magnitudes. Seeds overlap rows and budgets reuse data; show descriptive summaries without population significance or multiplicity-adjusted superiority claims. Stability can rise simply because larger sets overlap. Public data do not identify true signal features or support class-recall/FDR claims. Results already inspected in the original study make this exploratory even though this follow-up protocol is frozen.

Package every new checkpoint, source snapshot, input hashes, configuration, runtime and resume evidence, paired tables and figures. Reanalysis must require neither raw data nor fitting. Full replay requires only the pinned public archives, obtainable using the existing loader without an account.

```sh
uv sync --locked --all-extras
uv run python experiments/run_public_selection.py --output results/public-selection-v1 --cache data/raw
uv run python experiments/analyze_public_selection.py --output results/public-selection-v1 --report results/public-selection-v1/analysis
```
