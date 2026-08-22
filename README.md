# probe_fs — probe-gated feature selection

Implementation of the specification *Probe-Gated Feature Selection:
conditioning-position-matched shadow screening for predictive feature selection*.

The repository is named `probe_fs`; the installable Python package is `pgfs`.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/run_primary_experiment.py --n 300 --budgets 2 4 --outer 3 --inner 2
```

## The method in one screen

```python
from sklearn.ensemble import RandomForestRegressor
from pgfs import MethodSpec, StudyConstraints, make_primary, run_spec

data = make_primary(n=400, n_classes=4, class_size=3, rho=0.8, seed=1)

spec = MethodSpec(                      # Section 3: frozen before evaluation
    learner=RandomForestRegressor(n_estimators=200),
    loss="mse",
    n_contexts=8,                       # T
    n_importance_splits=2,              # B
    n_shadows=5,                        # m
    shadow_quantile=0.9,                # q
    candidate_k=(1, 2, 3, 5, 8, 12),    # K
    seed=1,
)

result = run_spec(
    data, spec,
    StudyConstraints(noninferiority_margin=0.05, k_max=8),  # Section 11
)
print(result.report())
```

## Section map

| Spec | Module | What it holds |
| --- | --- | --- |
| 3 | `config.py`, `players.py`, `losses.py` | frozen `MethodSpec`, players (features or blocks), loss registry |
| 4 | `contexts.py` | random-permutation contexts; full-conditioning LOCO |
| 5–7 | `importance.py` | `Delta`, `tau`, soft/hard gate, `I_gate`, ranking, diagnostics |
| 6 | `shadows.py` | marginal dimension-matched probes, empirical-quantile threshold |
| 8 | `selection.py` | selection path `A_k`, inner CV, one-standard-error rule |
| 9 | `nested.py` | nested evaluation, deployment finalization guard |
| 10 | `budget.py`, `experiments.py` | matched-compute accounting, budget ladder, primary comparison |
| 11, 14 | `metrics.py` | class recall, constraint-qualified endpoint, stability, reporting |
| 12 | `baselines.py`, `experiments.py` | external baselines and direct alternatives |
| 13 | `simulate.py` | primary redundant-class DGP, secondary regimes |
| 16 | `experiments.py` | decision criteria as executable rules |
| 17 | `importance.py` + `nested.py` | the minimal algorithm, as running code |

## Four implementation decisions worth knowing about

**Base-loss caching changes the compute accounting, not the method.** Within an
importance split, a permutation's conditioning sets are nested — the augmented set
of the player at position *i* is the base set of position *i+1* — so caching losses
on `frozenset(S)` turns `2p` base/augmented fits per context into about `p+1`.
Because the ungated arm buys many more contexts, it hits that cache far more often,
and the naive "each extra context costs `p−1` fits" rule overcharges it by roughly
10%. `budget.py` therefore matches budgets on the exact expectation

```
E[distinct base sets] = sum_s  C(p,s) * (1 - (1 - 1/C(p,s))^T)
```

and every experiment also reports *measured* fits and wall-clock time. One
consequence is worth stating plainly: distinct prefix sets saturate at `2^p`, so at
extreme budgets an exact match becomes infeasible rather than merely imprecise;
`BudgetPair.feasible` flags that instead of hiding it.

**Fits and thresholds are separate passes.** `estimate_importance` stores every
shadow contribution, then computes `tau` and gates in a vectorized second pass.
This is what lets `ImportanceResult.regate` produce the Section 12 global-shadow
comparator and the Section 13 sensitivity analyses over `q` from the *same* fits —
the comparison is then genuinely about conditioning-position matching rather than
about two runs that also differed in their random draws. Sensitivity to `m` is
deliberately *not* available this way: fewer shadows is a different set of draws,
not a re-reading of the ones you have.

**A comparison where either arm broke a constraint is inconclusive, not negative.**
Section 11 defines the endpoint only under the noninferiority and set-size
constraints. If the ungated comparator exceeds the loss margin, its recall is not
an endpoint value, so the difference is not an endpoint difference.
`decision_criteria` separates those budgets out and refuses to apply Section 16's
discontinuation rule to them — otherwise a comparator's constraint violation could
be read as evidence against gating.

**The leakage rules are enforced, not just documented.** The outer test fold enters
nothing but the final evaluation (a test corrupts each fold's held-out rows and
asserts that fold's ranking, `k`, and selected set are bit-identical), and
`finalize_for_deployment` raises unless given a genuinely untouched external test
set, because re-scoring a consensus set on the nested outer folds is exactly what
Section 9 forbids.

## What this code does not claim

Section 15, in code form. Nothing here returns a p-value, an FDR-controlled set, a
causal claim, an exact Shapley value, or a universal null floor. `tau` carries no
null probability — Section 6 is explicit that marginal shadows are negative
controls, not conditionally exchangeable null variables, and that correlation
between `X_j` and `X_S` can make the comparison conservative or anti-conservative
in ways the method does not model. The knockoff baseline *does* control FDR under
its Gaussian model-X assumptions; that guarantee belongs to the baseline and does
not transfer.

## Not yet implemented

- **Conditional probes** (Section 13 sensitivity): needs a model for `P(X_j | X_S)`.
  Section 13 also requires that when such a model is available, genuine conditional
  randomization tests be included as inferential alternatives — an add-a-probe
  comparison is not one, and should not be labelled as one.
- **Redrawn-versus-fixed splits** (Section 13 sensitivity): splits are currently
  fixed per estimation call.
- **Parallel execution.** The fit loop is serial. `estimate_importance` is the
  hot spot and parallelizes cleanly over `(b, t)`, but the base-loss cache is
  per-split, so a parallel version should shard by split to keep the cache and the
  fit accounting coherent.
- **Plotting.** `ComparisonTable.rows` is shaped for the Section 10
  performance-versus-compute curves; drawing them is left to the caller.

## Cost

One `estimate_importance` call costs about `B * (p + 1 + (T−1)(p−1) + T·p·m)` fits
before caching. Nested evaluation multiplies that by outer folds times
`(inner folds + 1)`. Start small — `T=2, B=1, m=3, 3 outer, 2 inner` — and check
`result.report()["main_results"]["model_fits_total"]` before scaling up.
