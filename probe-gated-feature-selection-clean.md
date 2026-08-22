# Probe-Gated Feature Selection

## Conditioning-position-matched shadow screening for predictive feature selection

## 1. Summary

Probe-gated feature selection is a model-based wrapper method for producing a parsimonious predictive feature set. It estimates the out-of-sample contribution of each feature across prespecified conditioning contexts and subtracts a context-matched shadow threshold before aggregating those contributions. Features are ranked by their mean soft-gated contribution, and the number of selected features is chosen by inner cross-validation.

The method is intended for predictive screening. It does not provide valid \(p\)-values, false-discovery-rate control, causal relevance, or exact identification of a unique true feature set.

## 2. Scope

The primary method uses:

- one predictive learner and loss function;
- random-permutation conditioning contexts;
- marginally permuted, dimension-matched shadow features;
- soft gating;
- one ranking score per feature;
- selected-set size \(k\) as the only selection-path parameter;
- nested cross-validation for evaluation.

Full-conditioning LOCO is the principal alternative. Conditional probes, hard gates, and other context samplers are sensitivity analyses rather than components of the primary method.

## 3. Data and notation

Let \(X=(X_1,\ldots,X_p)\) be the predictors and \(Y\) the outcome. A player may be a single feature or a predefined block of features. Let \(L(A;D_{\mathrm{tr}},D_{\mathrm{va}})\) be validation loss from a model trained on \(D_{\mathrm{tr}}\) using player set \(A\) and evaluated on \(D_{\mathrm{va}}\).

For reproducibility, the following must be fixed before evaluation:

- predictive learner and its tuning protocol;
- loss function;
- outer and inner resampling schemes;
- context distribution;
- number of contexts \(T\);
- number of shadows \(m\);
- shadow-threshold quantile \(q\);
- candidate selected-set sizes \(\mathcal K\).

## 4. Conditioning-context distribution

For every repetition \(t\), sample a uniformly random permutation \(\pi_t\) of the \(p\) players. For feature \(j\), define its conditioning set as the players preceding it:

$$
S_{jt}=\{\ell:\pi_t(\ell)<\pi_t(j)\}.
$$

This permutation distribution is the sole context distribution in the primary method. The same permutations must be used for gated and ungated methods in paired comparisons.

The resulting score measures average predictive contribution across random positions in a feature ordering. It is not a claim that every partial submodel is scientifically meaningful. Whether partial-context averaging improves selection over full-conditioning LOCO is an empirical question tested directly in the primary study.

## 5. Raw predictive contribution

For split \(b\), context \(t\), and candidate player \(j\), define

$$
\Delta_{jtb}
=
L(S_{jt};D^{(b)}_{\mathrm{tr}},D^{(b)}_{\mathrm{va}})
-
L(S_{jt}\cup\{j\};D^{(b)}_{\mathrm{tr}},D^{(b)}_{\mathrm{va}}).
$$

Positive values indicate improved validation performance after adding \(j\). The base and augmented models use the same training data, validation observations, learner settings, and—where supported—random seed.

## 6. Shadow construction

For each \((j,t,b)\), construct \(m\) shadows by independently permuting the training rows of player \(j\) and independently permuting its validation rows. For a feature block, use one joint row permutation for the entire block so that dimension and within-block dependence are preserved.

For shadow \(r\), compute

$$
\Delta^{(r)}_{jtb}
=
L(S_{jt};D^{(b)}_{\mathrm{tr}},D^{(b)}_{\mathrm{va}})
-
L(S_{jt}\cup\{Z^{(r)}_j\};D^{(b)}_{\mathrm{tr}},D^{(b)}_{\mathrm{va}}).
$$

Define the local shadow threshold

$$
\tau_{jtb}=Q_q\!\left(
\Delta^{(1)}_{jtb},\ldots,\Delta^{(m)}_{jtb}
\right).
$$

The primary specification fixes \(q=0.9\). The exact empirical-quantile convention must be specified in the implementation.

Marginal shadows are negative controls, not conditionally exchangeable null variables. Correlation between \(X_j\) and \(X_{S_{jt}}\) may make the shadow comparison conservative or anti-conservative. The method therefore attaches no null probability or inferential interpretation to \(\tau_{jtb}\).

## 7. Soft-gated score

Define the local soft-gated contribution

$$
G_{jtb}
=
\max\!\left\{0,
\Delta_{jtb}-\max(0,\tau_{jtb})
\right\}.
$$

Within an inner-training dataset, the primary ranking score is

$$
I_j^{\mathrm{gate}}
=
\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}G_{jtb}.
$$

Rank players in decreasing order of \(I_j^{\mathrm{gate}}\). Resolve exact ties with a fixed deterministic rule, such as original column order.

The raw score

$$
I_j^{\mathrm{raw}}
=
\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\Delta_{jtb}
$$

is retained only for diagnostics and the ungated comparison.

## 8. Feature-selection path

Let \(j_{(1)},\ldots,j_{(p)}\) be the gated ranking. For each candidate size \(k\in\mathcal K\), define

$$
A_k=\{j_{(1)},\ldots,j_{(k)}\}.
$$

Choose \(k\) through inner cross-validation using predictive loss. Apply the one-standard-error rule: select the smallest \(k\) whose mean inner-validation loss is within one standard error of the minimum.

Selected-set size \(k\) is the only selection-path parameter. The method does not additionally threshold gated importance, win rate, or selection frequency.

## 9. Nested evaluation

For each outer fold \(o\):

1. Hold out the outer test fold.
2. Using only the outer-training data, estimate the gated ranking.
3. Use inner cross-validation to choose \(k_o\).
4. Re-estimate the ranking on all outer-training data using the frozen method specification.
5. Fit the predictive learner on the top \(k_o\) players.
6. Evaluate it once on the outer test fold.

Aggregate outer-fold losses to estimate generalization performance. Per-fold selected sets may be used to describe stability but must not be combined into a consensus set and reevaluated on the same outer folds.

If a final consensus set is required for deployment, define it after nested evaluation, refit on all development data, and evaluate it only on a genuinely untouched external test set.

## 10. Primary comparison

The decisive comparison is soft-gated ranking versus ungated ranking at matched computational budgets.

For several budgets \(C_1<\cdots<C_H\), allocate approximately the same number of model fits to each method:

- **Gated:** fewer contexts, with \(m\) shadow fits per feature-context pair.
- **Ungated:** additional contexts in place of shadow fits.

Use identical outer folds, inner folds, learners, hyperparameter protocols, and paired random seeds. Report both model-fit count and wall-clock time.

Plot each method's results against computational budget rather than reporting only one matched-compute point.

## 11. Primary endpoint

The primary simulation regime contains correlated, redundant signal groups. Before running the study, define ground-truth signal classes \(\mathcal C_1,\ldots,\mathcal C_g\), where any member of a class can supply the corresponding predictive information.

For selected set \(A\), define class recall

$$
R_{\mathrm{class}}(A)
=
\frac{1}{g}\sum_{h=1}^{g}
\mathbf 1\{A\cap\mathcal C_h\ne\varnothing\}.
$$

Also report:

- the number of selected players;
- the number selected outside every signal class;
- redundant selections beyond the first member of each class;
- outer-test predictive loss.

The primary endpoint is class recall subject to both constraints:

$$
L_{\mathrm{outer}}(A)
\le
L_{\mathrm{outer}}(A_{\mathrm{all}})+\delta
$$

and

$$
|A|\le k_{\max},
$$

where the noninferiority margin \(\delta\) and maximum set size \(k_{\max}\) are fixed before simulation.

The minimum effect size of interest is an absolute increase of 0.05 in class recall at a matched computational budget while satisfying both constraints.

## 12. Required alternatives and baselines

### Direct alternatives

- ungated random-permutation-context importance at matched compute;
- full-conditioning LOCO with the same learner and selection path;
- full-conditioning LOCO with the same marginal shadow gate;
- global shadow comparison without conditioning-position matching.

### External baselines

- no feature selection;
- regularized regression or another learner-native selector;
- recursive feature elimination with cross-validation;
- tuned stability selection;
- Boruta or a comparable shadow-feature method;
- model-X knockoffs when their assumptions are supportable.

Each baseline receives a reasonable prespecified tuning protocol. Results are reported as performance-versus-compute curves; simpler baselines are not forced to spend unused budget on irrelevant tuning parameters.

## 13. Simulation design

### Primary grid

Use one prespecified learner, one marginal-shadow construction, and one loss function. Vary only:

- sample size;
- predictor correlation;
- signal strength;
- computational budget.

The primary data-generating process must explicitly define the redundant signal classes used by the primary endpoint.

### Secondary regimes

After completing the primary grid, examine:

- independent null and signal features;
- weak signals near the selection boundary;
- nonlinear main effects;
- interactions with weak marginal effects;
- mixed continuous and categorical predictors;
- \(p\ge n\).

### Sensitivity analyses

- number of shadows \(m\);
- shadow quantile \(q\);
- full versus partial conditioning;
- marginal versus estimated conditional probes;
- alternative learner classes;
- fixed versus redrawn splits.

Conditional probes are evaluated only as a sensitivity analysis. If a reliable model for \(P(X_j\mid X_S)\) is available, conditional randomization methods must also be included as inferential alternatives. An add-a-probe comparison is not itself a conditional randomization test.

## 14. Reporting

### Main results

- outer-test predictive loss;
- class recall in regimes with defined redundant classes;
- selected-set size;
- false selections outside signal classes;
- model-fit count;
- wall-clock time.

### Diagnostics

- raw and gated importance rankings;
- variability across data splits;
- variability across conditioning contexts;
- within-class redundancy;
- sensitivity to probe construction;
- stability of selected sets across outer folds.

Context variability and split variability must be reported separately. Stability is always paired with predictive performance and selected-set size, since selecting nothing is trivially stable.

## 15. Interpretation

A successful study may support the claim that conditioning-position-matched shadow subtraction improves predictive feature selection at specified computational budgets and in specified correlated designs.

It may not support claims that the method:

- identifies causal variables;
- recovers a unique true support under redundancy;
- produces valid \(p\)-values;
- controls family-wise error or false discovery rate;
- makes marginal shadows exchangeable with correlated null features;
- estimates a universal null floor;
- is an exact Shapley-value estimator.

## 16. Decision criteria

Run the matched-compute gated-versus-ungated experiment first.

- If gating fails to improve the primary endpoint by the prespecified minimum effect across the primary computational budgets, discontinue the broad methodology study.
- If full-conditioning gated LOCO matches or exceeds sampled partial-context gating, adopt the simpler full-conditioning method as the primary contribution.
- If an advantage appears only under one unusually favorable tuning choice, treat it as exploratory rather than confirmatory.
- If the method selects nearly the same sets as an existing baseline, assess whether it offers a meaningful advantage in stability, predictive loss, interpretability, or computation before claiming a separate contribution.

## 17. Minimal algorithm

```text
input:
    development data, learner, loss
    context count T, split count B
    shadow count m, fixed quantile q
    candidate set sizes K

for each outer fold:
    hold out outer-test data

    using outer-training data only:
        for each inner split b:
            sample T random player permutations

            for each feature j and context t:
                S <- players preceding j in permutation t
                delta <- loss(S) - loss(S + j)

                for r in 1..m:
                    create a dimension-matched marginal shadow of j
                    shadow_delta[r] <- loss(S) - loss(S + shadow[r])

                tau <- empirical_quantile(shadow_delta, q)
                gated[j,t,b] <- max(0, delta - max(0, tau))

        score[j] <- mean over t and b of gated[j,t,b]
        rank features by decreasing score
        choose selected-set size k by inner validation and the 1-SE rule
        refit the final learner on the top k features

    evaluate once on the outer-test data

return:
    outer-test performance
    per-fold selected sets
    gated rankings
    compute and stability diagnostics
```

## 18. Paper structure

1. **Introduction:** instability in model-based feature rankings and the motivation for local negative controls.
2. **Related work:** wrapper selection, LOCO, sampled Shapley importance, shadow features, stability selection, conditional randomization, and knockoffs.
3. **Method:** context distribution, marginal shadows, soft gating, and the one-parameter selection path.
4. **Evaluation protocol:** nested resampling, matched-compute comparisons, and redundancy-aware metrics.
5. **Primary experiment:** gated versus ungated ranking across computational budgets.
6. **Context ablation:** sampled partial conditioning versus full-conditioning LOCO.
7. **Secondary experiments and application.**
8. **Discussion:** predictive scope, correlation sensitivity, computation, and limits of shadow calibration.
