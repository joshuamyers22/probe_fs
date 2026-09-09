# Expanded simulation results

The larger batch does not show a persistent gating advantage under the original
endpoint and learner configuration. All 240 planned pairs completed, using
790,320 model fits in approximately 8.2 minutes with four concurrent seed workers.
All measured compute imbalances were below 5%; the maximum was 3.14%.

The [frozen protocol](EXPANDED_SIMULATION_PLAN.md) used ten new seeds, sample
sizes 300/1,000/3,000, two correlations, two signal-to-noise ratios and two budgets.
The loss margin, maximum selected size, learner and resampling rules were retained
from the original pilot. Public and financial datasets were not run in this batch.

## Endpoint feasibility

| Sample size | Valid comparisons | Inconclusive comparisons | Total |
| --- | --- | --- | --- |
| 300 | 7 | 73 | 80 |
| 1,000 | 27 | 53 | 80 |
| 3,000 | 45 | 35 | 80 |
| Total | 79 | 161 | 240 |

All 161 invalid comparisons failed the predictive-loss requirement; no selected
set violated the size cap. Both arms failed in 115 pairs, only ungated failed in
24, and only gated failed in 22. Increasing sample size made valid comparisons
more common, especially in the stronger-signal regime. Low-SNR comparisons mostly
remained inconclusive even at n=3,000.

## Gated versus ungated recall

Among the 79 valid comparisons there were 77 ties, one gated win and one gated
loss. Both methods achieved perfect class recall in 74 of these comparisons.
This leaves little opportunity for an additional recall gain among the valid
results in this setup.

The single win was n=300, rho=0.9, SNR=2, four gated contexts, seed 110: recall
difference +0.1111. The corresponding design had only three valid seeds and a
conditional mean difference +0.0370, below the fixed +0.05 effect. Its descriptive
95% t interval was [-0.1223, +0.1964]. The single loss was n=1,000, rho=0.9,
SNR=0.25, four contexts, seed 107, with difference -0.1111.

No design/budget group with valid observations had a conditional mean recall
advantage of +0.05. Groups without valid observations have no endpoint estimate.
These are conditional descriptive results; zero-width intervals in groups of
observed ties do not establish certainty about unseen replications. Ten seeds
and substantial constraint-driven exclusion do not support a broad confirmatory
claim. The per-budget decision helper was not misapplied to pooled seed rows.

## What this implies for the next work

The tested configuration supplies no consistent evidence that shadow gating
improves the primary endpoint at matched compute. Before spending more on the
same grid, diagnose the interaction between the one-standard-error size rule,
the full-feature loss reference, and the redundant-class generator. That audit
should explain the high rate of loss-constraint failures and recall ceiling;
changing constraints to rescue these completed runs would invalidate their
prespecified interpretation. Any revised design should be separately frozen
and evaluated on new seeds. Gated LOCO and external-selector comparisons remain
unrun; this report makes no claims about them.

## Reproduction and artifacts

- [Configuration](../experiments/expanded-simulation.json).
- [All design-level statistics](../experiments/reports/expanded-simulation-v1.csv),
  including constraint failures and paired standard errors.
- [Figure](figures/expanded-simulation.png): endpoint validity, conditional recall
  differences, and all-seed loss differences.
- Full local results: `results/expanded-simulation-v1/`, including 240 cell JSON
  files, per-seed logs and provenance, checksum-verified source snapshot,
  `PAIRED_REPORT.md`, and analysis provenance with hashes of every input cell.

Run the two commands in the protocol to reproduce execution and analysis. Resume
was verified for all 240 cells: checksums and modification times stayed unchanged.
Lint, all 65 tests, and source/wheel builds passed; the existing knockoff tests
still emit two scikit-learn deprecation warnings.
