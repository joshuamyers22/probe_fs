# Context and shadow-scope control results

Neither conditioning-position matching nor sampled partial contexts shows a
consistent **qualified class-recall advantage** over its tested control. Local
and global shadows produce especially similar selected sets. Full conditioning
is not uniformly preferable either: partial contexts have a substantial
predictive-loss advantage in one small-sample, low-SNR regime.

All **960 nested evaluations** completed on 120 datasets and seeds 401–410,
using **1,289,472 actual model fits in 13.5 minutes** with four workers. The
[protocol](CONTEXT_CONTROLS_PLAN.md) retained the primary one-SE rule, revised
coefficients (1, 0.30, 0.15), complete size path 1–6, shared seeded feature order,
loss margin 0.05, size cap 6, and minimum recall effect +0.05. No argmin or public
datasets were run in this batch.

## Matched comparisons

Differences below are local gating minus the named control. Every planned
comparison met the 5% measured-fit tolerance: local/global counts matched exactly;
the largest local/LOCO imbalance was 3.08%.

| Control | Valid matched pairs / 240 | Local wins | Ties | Local losses |
| --- | --- | --- | --- | --- |
| Global-shadow gating | 146 | 17 | 114 | 15 |
| Gated LOCO at matched fit budgets | 122 | 33 | 63 | 26 |

All observed wins clear the fixed +0.05 effect at the individual-pair level.
However, **none of the available design-level qualified recall intervals excludes
zero**, in either direction. Failed endpoints remain inconclusive rather than
being counted as losses. There are 24 design/budget groups per control, each
with ten seeds. The controls reuse local results and budget groups reuse
datasets; pooled counts are descriptive, not independent trials of one effect.

| Sample size | Local/global valid / 80 | Local/matched-LOCO valid / 80 |
| --- | --- | --- |
| 300 | 37 | 21 |
| 1,000 | 57 | 51 |
| 3,000 | 52 | 50 |

Every invalid endpoint failed the loss requirement; no method exceeded the size
cap. For local/global, both arms failed in 77 pairs, only local in 11, and only
global in 6. For local/LOCO, the counts were 59, 29 and 30, respectively.

## Local versus global shadow thresholds

The mean within-fold Jaccard overlap of selected sets was **0.902** across this
grid. Raw outer scores and fit counts matched exactly, as expected when only
threshold scope changes. The findings supply little evidence that the more
specific local threshold materially improves selection here. This is not a
formal equivalence result.

One local/global group has a mean recall difference above +0.05: n=300, rho=0.5,
SNR=2, T=4, with +0.0741 and interval [-0.2446, +0.3928]. It has only three valid
seeds. The higher-n results do not show a persistent effect of that size.

## Partial contexts versus full conditioning

For compute matching, gated LOCO spent its budget on B=2 and B=4 importance
splits, versus one split with T=2 and T=4 partial contexts. Its full context was
never redundantly repeated. Thus this comparison changes both context structure
and allocation of estimation effort. The separate single-split LOCO controls
keep B fixed at unequal cost.

Mean selected-set Jaccard overlap with matched LOCO was **0.624**, so these methods
behave more differently than local/global thresholds. At n=3,000, rho=0.9, SNR=2,
T=4, local gating's qualified recall difference was +0.0556, with all ten seeds
valid and interval [-0.0006, +0.1118]. This is suggestive, but crosses zero; at
T=2 the mean was +0.0444 [-0.0410, +0.1299], below the fixed minimum effect.
It does not establish a persistent advantage across budgets.

There is a separate predictive-loss finding at **n=300, rho=0.9, SNR=0.25**:

| Budget | All-seed MSE difference, local minus matched LOCO | Descriptive 95% t interval | Valid recall pairs / 10 |
| --- | --- | --- | --- |
| T=2 | -0.3399 | [-0.4489, -0.2309] | 1 |
| T=4 | -0.3699 | [-0.5417, -0.1981] | 3 |

Partial contexts predict better in this regime. These loss estimates retain all
ten seeds, including constraint failures, and are not qualified recall evidence.
This finding argues against treating LOCO as a uniformly better replacement.
It also does not establish the value of gating versus ungated partial contexts,
which the preceding studies examined separately.

## Actual costs and natural-cost controls

| Method | Mean model fits | Mean fit/predict seconds | Endpoint passes / 120 | Mean selected size |
| --- | --- | --- | --- | --- |
| Local T=2 | 1,120.4 | 2.197 | 74 | 2.458 |
| Global T=2 | 1,120.4 | 2.198 | 77 | 2.464 |
| Gated LOCO B=2 | 1,140 | 2.474 | 75 | 2.731 |
| Local T=4 | 2,174.9 | 4.285 | 78 | 2.522 |
| Global T=4 | 2,174.9 | 4.286 | 80 | 2.558 |
| Gated LOCO B=4 | 2,238 | 4.857 | 76 | 2.775 |
| Gated LOCO B=1 | 591 | 1.275 | 50 | 2.531 |
| Ungated LOCO B=1 | 186 | 0.393 | 57 | 2.436 |

These are descriptive equal-weight grid averages. Timings come from the model
fit/predict counter and are hardware-dependent, not total batch elapsed time.
Matched fit counts do not imply identical runtime: LOCO fits fuller models.
Single-split LOCO is cheaper but passes the endpoint less often. Those points
belong on cost curves, not in matched-compute recall claims. No fits were shared
between methods; a common reference was independently refitted each time.

## Limits and project decision

Intervals are conditional, descriptive t intervals over at most ten valid seeds,
without multiplicity adjustment. Sparse groups yield very wide intervals, while
identical observed effects yield zero-width intervals; neither establishes
certainty about unseen seeds. Comparisons with different controls can have
different valid subsets. This batch tests one learner, one selection rule and a
specific noisy-proxy DGP, not all feature-selection settings.

Together with the earlier matched ungated comparisons, the controls provide no
basis for expanding a broad superiority claim for probe gating. Consolidate the
reproducible benchmark and its negative/inconclusive findings before adding more
methodology experiments. The useful next deliverable is a reviewable benchmark
package with all frozen protocols and result summaries. A later public-data
application study can assess practical loss/size/cost tradeoffs, but should not
be presented as evidence for known-truth class-recall superiority. Neither a
default switch to LOCO nor new endpoint tuning is justified by this diagnostic.

## Integrity and artifacts

Analysis validated every fitted specification and one-SE choice, reconstructed
all recorded data parameters, feature permutations, names and truth mappings,
and checked shared reference losses and local/global raw-score/fit-count
invariants on all 120 datasets. Exact score ties crossed 2 local-T2 outer
boundaries, 0 local-T4, 9 global-T2, 6 global-T4, 8 LOCO-B2, 0 LOCO-B4,
17 single-split gated LOCO and 0 ungated LOCO boundaries (360 folds per method).
The shared seed-based order avoids a fixed signal-first tie preference; this is
not an exhaustive order sensitivity study or an audit of every inner-ranking tie.

All **104 tests** and lint passed, with the two pre-existing scikit-learn warnings
in knockoff tests. Resume preserved every new checkpoint's hash and modification
time, as well as the original execution timing. All 960 checkpoints from the
three preceding study batches also remained unchanged.

- [Frozen protocol and commands](CONTEXT_CONTROLS_PLAN.md) and [configuration](../experiments/context-controls.json).
- [Paired contrasts](../experiments/reports/context-controls-v1.csv), [all-method summaries](../experiments/reports/context-methods-v1.csv), and [counts](../experiments/reports/context-control-counts.json).
- [Compute preflight](../experiments/reports/context-compute-preflight.json).
- [Endpoint plot](figures/context-controls.png) and [cost curves](figures/context-costs.png).
- Full local results: `results/context-controls-v1/`, including 960 checkpoints,
  source/protocol archive, frozen config, original and resume timings, analysis
  source copies, per-cell input hashes, and `PAIRED_REPORT.md`.

These simulations require no downloaded data or account.
