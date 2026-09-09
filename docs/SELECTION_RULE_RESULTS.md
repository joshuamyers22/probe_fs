# Selection-rule sensitivity results

Inner-CV argmin substantially improves endpoint validity in this grid, while
selecting larger sets. It does **not** establish a persistent benefit from shadow
gating. All **480 gated/ungated comparisons** completed: 240 per selection rule,
on the same 120 generated datasets and ten fresh seeds. The run used **1,586,040
actual model fits** in approximately **22.1 minutes**, with four workers total.
Every comparison met the 5% measured-fit tolerance; the maximum imbalance was
2.77%. Each policy used 793,020 fits, with no models shared between policies.

The [frozen protocol](SELECTION_RULE_PLAN.md) retained coefficients (1, 0.30, 0.15),
the 0.05 loss margin, size cap 6, complete size path 1–6 and +0.05 minimum recall
effect. A seeded feature permutation was shared across both policies and budgets,
with player names and truth classes remapped. Both rules were run through the
full nested procedure on seeds 301–310. Argmin remains a sensitivity analysis;
the primary one-SE method has not been replaced.

## Validity improves, with a size tradeoff

| Outcome across paired designs | Count |
| --- | --- |
| Valid under both rules | 155 |
| Recovered by argmin | 56 |
| Lost under argmin | 3 |
| Invalid under both | 26 |

One-SE yielded **158/240 valid comparisons (65.8%)**, versus **211/240 (87.9%)**
under argmin: a net increase of 53, or **22.1 percentage points**. All failures
were predictive-loss failures; none exceeded the size cap. Under one-SE, both
arms failed in 54 pairs, only gated in 16, and only ungated in 12. Under argmin,
the corresponding counts were 13, 11 and 5.

| n | SNR | One-SE valid / 40 | Argmin valid / 40 |
| --- | --- | --- | --- |
| 300 | 0.25 | 29 | 27 |
| 300 | 2 | 14 | 33 |
| 1,000 | 0.25 | 24 | 33 |
| 1,000 | 2 | 39 | 40 |
| 3,000 | 0.25 | 12 | 38 |
| 3,000 | 2 | 40 | 40 |

The clearest low-SNR improvement occurs at n=3,000: **12/40 to 38/40 valid**.
Argmin is not uniformly better on every dataset; at n=300 and low SNR, validity
falls slightly. These paired interventions support the earlier diagnosis that
the one-SE size tolerance contributes to failures under the tighter loss margin.

| Arm | Mean size, one-SE | Mean size, argmin | Mean MSE change, argmin minus one-SE |
| --- | --- | --- | --- |
| Gated | 2.619 | 4.035 | -0.03131 |
| Ungated | 2.569 | 4.046 | -0.03505 |

These are descriptive equal-weight averages over all 240 design/seed/budget rows,
including failed endpoints. They are not pooled independent-seed inference.
Per-design paired intervals for size, loss and validity are in the full table.
Larger argmin selections remain within the same size cap and fit budget.

## Gating still has no persistent advantage

| Selection rule | Valid pairs | Gated wins | Ties | Gated losses |
| --- | --- | --- | --- | --- |
| One-SE | 158 | 38 | 88 | 32 |
| Argmin | 211 | 64 | 95 | 52 |

The two rows condition on different valid subsets, so their win counts cannot
establish that changing the policy improves the gating effect. The paired-policy
analysis uses only the **155 designs valid under both rules** for that question,
with each design group retaining its own common-valid count.

No argmin design has a gated-minus-ungated recall interval excluding zero. Only
one argmin group's point estimate exceeds +0.05: n=300, rho=0.9, SNR=0.25, T=2,
with six valid seeds, mean +0.0741 and descriptive interval [-0.0464, +0.1945].
This is not a stable advantage across sample sizes or budgets.

One-SE has an isolated positive interval at n=1,000, rho=0.9, SNR=0.25, T=4:
+0.0794 [0.0017, 0.1570] among seven valid seeds. Under argmin, that group's
qualified estimate is +0.0123 [-0.0873, +0.1120] among nine valid seeds. Those
conditional means use different samples; the paired change on the seven common
valid seeds is -0.0635 [-0.1638, +0.0368].

The preceding benchmark's localized positive result at n=1,000, rho=0.5, SNR=2,
T=2 did not recur: the fresh one-SE estimate is -0.0123 [-0.0791, +0.0544], and
argmin is -0.0333 [-0.1087, +0.0421]. Seeds and feature order differ from the older
batch, so this does not isolate the effect of reordering.

All intervals are descriptive, conditional 95% t intervals over at most ten
seeds per design, without multiplicity adjustment. Budget groups reuse datasets.
One common-valid group has two identical observed policy-effect changes of
+0.1111, producing a nominal zero-width interval; two observations do not establish
certainty. The full paired table preserves this small denominator. Argmin has
20 valid pairs where both methods reach perfect recall, compared with zero under
one-SE; most valid argmin pairs still have room below that ceiling.

## Pairing, ties and reproducibility

The analysis verified all 240 cross-policy design matches: identical recorded
data parameters and permutations, outer rankings and scores, inner mean-loss
paths, reference losses, and method settings except the selection-rule flag.
Every outer size obeyed its configured rule, and every one-SE selected set was
a prefix of the corresponding argmin selection. Source, dependency and frozen
configuration identities were checked before comparison.

Exact gated score ties crossed the outer selection boundary in 3/720 one-SE
folds and 11/720 argmin folds; among valid pairs, the counts were 2/474 and 10/633.
Ungated had none. The shared seeded ordering avoids a fixed signal-first tie
preference, but this is not an exhaustive feature-order sensitivity study. The
tie diagnostic covers outer boundaries, not every inner-ranking tie.

All 480 checkpoints retained their hashes and modification times through a full
resume. The preceding expanded and revised batches also retained all 480 original
checkpoint hashes. Protocol copies, source archives, analysis source and input
hashes are retained locally. All **92 tests** and lint passed; the two existing
scikit-learn deprecation warnings remain in knockoff tests.

## Next work

The selection-rule sensitivity has answered its main feasibility question.
Preserve these results and move to the planned **gated LOCO and global-shadow
controls**, with a separately frozen protocol and paired compute accounting.
Those comparisons can test whether conditioning-position matching and randomized
contexts add value. Keep one-SE primary and argmin explicitly labelled as a
sensitivity; further endpoint or coefficient tuning should not be used to rescue
the current gating result. Large public-data expansion remains an application
benchmark, separate from evidence for a known-truth class-recall advantage.

## Artifacts

- [Protocol and commands](SELECTION_RULE_PLAN.md).
- [One-SE configuration](../experiments/selection-one-se.json) and [argmin configuration](../experiments/selection-argmin.json).
- [Paired policy table](../experiments/reports/selection-rule-v1.csv) and [transition counts](../experiments/reports/selection-rule-counts.json).
- [One-SE endpoint table](../experiments/reports/selection-one-se-v1.csv) and [argmin endpoint table](../experiments/reports/selection-argmin-v1.csv).
- [Figure](figures/selection-rules.png).
- [One-SE tie diagnostic](../experiments/reports/selection-one-se-ties.json) and [argmin tie diagnostic](../experiments/reports/selection-argmin-ties.json).
- Full local checkpoints and provenance: `results/selection-rule-v1/`.

No downloaded dataset or account is needed to reproduce these simulations.
