# Endpoint audit results

The failures have identifiable design causes: the simulated class members are
complementary noisy measurements, the size path omits five, and the one-SE rule
often shrinks sets beyond what the much tighter outer loss margin permits. The
strict loss requirement also favors full class coverage, producing a recall
ceiling among valid comparisons. This does not establish a hidden gating benefit.

The audit reproduced all 1,440 original outer-fold selected-set losses, then
scored fixed diagnostic alternatives. It used 6,840 additional model fits in
15.3 seconds. Original experiment outputs, method settings and constraints were
not changed. Alternative scores reuse old test folds and are exploratory.

## 1. The generator supplies noisy proxies, not fully interchangeable features

The implemented DGP is

    X_hj = sqrt(rho) U_h + sqrt(1-rho) e_hj
    y = sum_h beta_h U_h + outcome noise

where U_h and measurement errors are independent standard Gaussians. With r
members of a class observed, the conditional latent variance is

    Var(U_h | r measurements) = (1-rho) / (1-rho+r*rho).

Thus a second member reduces prediction error even after the first has been
selected. Class recall treats those two choices identically. The default
coefficients are 1, 0.85 and 0.7. The following excess MSE values compare the best
population predictor on a subset with the population predictor on all six signal
features; common outcome noise cancels from the difference.

| Within-class correlation | Best 3 features, one/class | Best 4 | Best 5 | All 6 |
| --- | --- | --- | --- | --- |
| 0.5 | 0.36875 | 0.20208 | 0.08167 | 0 |
| 0.9 | 0.10480 | 0.05743 | 0.02321 | 0 |

The permitted excess is only 0.05. Among the original candidate sizes
{1,2,3,4,6}, only six can meet that population oracle benchmark at either
correlation. At rho=0.9, five would be feasible but is absent from the path.
Finite-sample Ridge comparisons need not follow these population bounds exactly:
the actual reference also pays estimation error for fitting null predictors.
Indeed, known-truth three-feature sets passed on 40/120 datasets, five-feature
sets on 100/120, and six-feature sets on 118/120.

The weakest class has coefficient 0.7. Dropping that entire class increases
population MSE by at least 0.32667 (rho=0.5) or 0.46421 (rho=0.9), relative to the
full population predictor. Even omitting it in just one of three equally weighted
folds costs more than the 0.05 mean-loss allowance in this oracle calculation.
This explains the pressure toward perfect class recall among feasible sets.
Finite-sample reference error and test noise allow exceptions, as seen in the
actual results; the calculation is not a claim that finite-sample failures are
mathematically inevitable.

Lowering the current SNR parameter adds outcome noise. It does not reduce those
absolute class contributions relative to the fixed MSE margin. A noisy problem
therefore does not automatically create a feasible low-recall comparator.

## 2. The one-SE rule and the loss margin impose different tolerances

| SNR | Recorded selection decisions | Shrunk below inner-CV argmin | Median inner SE |
| --- | --- | --- | --- |
| 0.25 | 720 | 435 (60.4%) | 0.31496 |
| 2.0 | 720 | 147 (20.4%) | 0.04688 |

These count decisions across methods, budgets and folds, not independent
replications. In low-SNR designs, the median one-SE tolerance exceeds the outer
margin by more than sixfold. The one-SE rule is implemented as specified, but it
optimizes parsimony under a different tolerance from the endpoint's loss rule.

## 3. Paired diagnostic rescores support that explanation

| Diagnostic policy | Valid pairs / 240 | Gated passes / 240 | Ungated passes / 240 | Valid pairs with both recalls 1 |
| --- | --- | --- | --- | --- |
| Original one-SE | 79 | 103 | 101 | 74 |
| Saved inner-CV argmin size | 148 | 179 | 160 | 119 |
| Fixed top 5 | 140 | 154 | 153 | 123 |
| Fixed top 6 | 190 | 206 | 194 | 176 |

Using the argmin sizes recovered 69 previously invalid pairs and lost none of
the original valid pairs. Fixed top-six sets recovered 111 and lost none.
Fixed top-five recovered 91 but lost 30, so adding five should be tested through
inner selection rather than simply forcing it for every dataset.

These policies use saved rankings and independent-of-outer-outcome size rules;
no ranking fits were rerun. They are still post-hoc sensitivity checks on reused
test folds, not new evidence for method superiority. Their valid subsets differ,
so conditional recall averages across policies should not be compared as if
they represented the same population.

## Recommended next study

Do not simply repeat the existing grid at a larger n. First make an explicit
choice about the scientific target: coverage of noisy-proxy groups or selection
among genuinely substitutable measurements. The current generator is appropriate
for the former but its documentation overstates interchangeability.

For a revised benchmark, precompute population feasibility before simulation:
include weak class contributions that can be omitted within the original loss
margin, so a valid comparator can have less-than-perfect class recall. Require
both feasible sub-ceiling and full-coverage sets to exist under the size cap.
Use the complete size path through six, including five. Keep one-SE as the
primary rule if preserving the original method, and prespecify inner argmin as
a separate sensitivity analysis. Evaluate any revision on fresh seeds and retain
the completed original studies as results of their original specification.

## Artifacts and validation

- [Audit protocol](ENDPOINT_AUDIT_PLAN.md).
- [Population-risk table](../experiments/reports/endpoint-population-risk.json).
- [Policy summaries](../experiments/reports/endpoint-policy-summary.csv).
- [Figure](figures/endpoint-audit.png).
- Local checkpointed audit and source: `results/endpoint-audit-v1/`.

Tests independently verify the risk formula using Gaussian covariance matrices
and check the size-five feasibility finding. Lint and all 73 tests passed.
