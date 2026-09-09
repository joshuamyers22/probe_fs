# Revised noisy-proxy simulation results

The revised benchmark creates room to measure class-recall differences, but does
not establish a persistent gating advantage. All 240 planned paired comparisons
completed on fresh seeds, using **792,336 model fits in approximately 8.3 minutes**
with four seed workers. All pairs met the 5% measured-compute tolerance; the
largest imbalance was 3.21%.

The [frozen protocol](REVISED_SIMULATION_PLAN.md) kept the 0.05 MSE margin, size
cap 6, +0.05 minimum recall effect and one-SE selection rule. It changed the
signal coefficients to (1, 0.30, 0.15), included every candidate size from 1 to 6,
and used seeds 201–210. Population calculations verified that both two-class and
three-class coverage can satisfy the endpoint. This is a new exploratory DGP,
not a revision of the completed original results.

## Endpoint validity and recall

| Sample size | Valid / 80 | Low SNR valid / 40 | Higher SNR valid / 40 |
| --- | --- | --- | --- |
| 300 | 39 | 27 | 12 |
| 1,000 | 47 | 13 | 34 |
| 3,000 | 54 | 14 | 40 |
| Total | 140 / 240 | 54 / 120 | 86 / 120 |

Of 100 invalid pairs, both arms failed in 68, only ungated failed in 18, and only
gated failed in 14. Every failure was the loss constraint; none violated the
size cap. Gated passed individually in 158 pairs and ungated in 154. Finite-sample
validity does not necessarily increase with n: the actual all-feature Ridge
reference also has estimation error, unlike the population preflight reference.
The table describes the observed pattern, not a causal decomposition of it.

Among the **140 valid pairs**, there were **32 gated wins, 19 losses and 89 ties**.
All 32 wins exceeded the fixed +0.05 recall difference. No valid pair had both
methods at perfect mean class recall, compared with 74/79 in the earlier study.
The recall ceiling is therefore absent in this batch. The validity change from
79/240 to 140/240 is descriptive across different DGPs and seeds; it cannot be
attributed separately to weaker coefficients or the added size-five candidate.

## The positive result is localized

At n=1,000, rho=0.5, SNR=2 and two gated contexts, seven of ten seeds were valid.
The conditional mean recall difference was **+0.1111**, with descriptive 95% t
interval **[+0.0272, +0.1950]**. Five valid pairs cleared +0.05. This is the only
design whose interval excludes zero; its lower bound still lies below +0.05.

At the same n/rho/SNR with four contexts, the difference was -0.0317
[-0.1855, +0.1220]. At n=3,000 with two contexts, it was +0.0444
[-0.0111, +0.1000], below the minimum effect. The apparent gain does not persist
across budget or sample size. Another group has a +0.2222 mean but only **one**
valid seed, so no uncertainty interval can be estimated.

These are conditional descriptive intervals over ten seeds per design, without
multiplicity adjustment. Groups reuse datasets across budgets. Pooled win counts
are not independent replications of a single common effect. Zero-width intervals
for observed ties do not establish certainty, and very wide t intervals with two
valid seeds can extend outside the parameter's possible range. Failed endpoints
remain in the validity denominator and never become qualified recall evidence.

## Size path and tie diagnostic

Size five was actually selected in 49/720 gated outer folds and 50/720 ungated
outer folds. Within valid pairs, each method selected it in 45/420 folds. These
counts demonstrate use of the complete path, not a causal effect of adding five.

A separately labelled, post-run diagnostic checked exact score ties at the outer
selection boundary. There were two gated boundary ties across all 720 folds and
none among the 420 folds belonging to valid pairs; ungated had none. Thus no
valid outer selection cut through an exact tie. The implementation still breaks
ties by player index, and simulated signal features precede nulls. This audit
does not inspect inner-ranking tie effects or establish order invariance. No
tie-breaking rules or scores were changed, and no additional models were fitted.

## Next experiment

The next useful test is a separately frozen **inner-CV argmin sensitivity** on
fresh seeds, compared with the current one-SE rule across the full design. The
earlier audit implicated the selection tolerance, and 100/240 revised pairs
still fail the loss requirement. Such a sensitivity would test whether validity
improves without assuming that gating benefits. It should retain the revised DGP,
margin, complete size path and matched-compute accounting, and explicitly address
feature-order ties. It is a deviation from the primary method, not a replacement
chosen to rescue this result. Do not expand only the isolated winning cell.

Gated LOCO, global-shadow ablations and external-selector benchmarks remain
separate unrun comparisons. This batch supports no superiority claim over them.

## Reproduction and artifacts

- [Configuration](../experiments/revised-simulation.json) and [protocol](REVISED_SIMULATION_PLAN.md).
- [Population preflight](../experiments/reports/revised-feasibility.json).
- [All 24 design summaries](../experiments/reports/revised-simulation-v1.csv), including paired MCSE, intervals and constraint failures.
- [Figure](figures/revised-simulation.png) and [outer tie diagnostic](../experiments/reports/revised-ranking-ties.json).
- Local full run: `results/revised-simulation-v1/`, with 240 checkpoints, source
  snapshot, protocol and preflight copies, frozen configuration, seed logs,
  analysis source, input hashes, and `PAIRED_REPORT.md`.

Follow the protocol commands; simulations require no downloaded dataset or
account. The analysis verifies frozen identities, actual generated coefficients,
and fitted candidate paths. Protocol/preflight hashes and every archived core
source matched their recorded identities. A full resume preserved every cell's
checksum and modification time. All 240 original expanded-study checkpoints
also remain unchanged. Lint and all 81 tests passed, with the two existing
scikit-learn deprecation warnings in knockoff tests.
