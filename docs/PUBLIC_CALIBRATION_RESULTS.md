# Public application infrastructure and cost calibration

Two calibration runs used 600 development rows from each pinned public dataset,
seed 599, three outer folds and three inner folds. Both are excluded from the
scientific application grid (seeds 501–505). No comparative predictive outcomes
were used to choose the final grid. Year Prediction's official test rows remain
excluded by the loader and runner.

| Calibration | Completed methods | Convergence failures | Original worker elapsed time | Projected grid worker-hours | Cost gate |
| --- | --- | --- | --- | --- | --- |
| v1: 20,000 solver iterations; proposed n=3,000/10,000 | 17/18 | One Superconductivity L1 failure | 174.83 seconds | 10.52 | Failed |
| v2: 200,000 solver iterations; revised n=3,000/5,000 | 18/18 | None | 184.49 seconds | 6.83 | Passed |

The penalty grid, five replication seeds and outcome rules did not change.
Calibration v2 counted **64,488 model fits**, including every successful L1 tuning
fit and refit. It emitted no warnings. Original measured peak process memory was
0.214 GiB; the conservative projection is 0.457 GiB, below the 8 GiB planning cap.
The failed v1 selector has unavailable fit counts; those partial attempts are
not silently included in a purported exact total. Its elapsed time is included.

Time projections multiply first-invocation worker time (including startup and
loading) by twice the row-count ratio and the planned seed count. Memory uses
twice the observed process peak plus ten dense full-size float64 arrays. These
are deliberately cautious heuristics for this learner and feature width, not
cross-hardware guarantees. Solver conditioning can change at larger n. The
runner's timeout enforces the eight-worker-hour cap across invocations; memory
is checked between methods, rather than enforced during individual fits.

Validation includes fold-local imputation/scaling, counted selector and final
prediction fits, an outer-held-out-data perturbation test, intercept-only empty
selection, shared original row/fold IDs, tamper and identity rejection, and
reserved Year Prediction row rejection. The analyzer retains failed methods and
refuses to treat calibration as scientific application evidence. A resume of v2
preserved all 18 method checkpoints and three shared data/fold records byte for
byte and in modification time.

Both runs' evidence, source snapshots, row IDs, folds, failure records and cost
reports are included in `benchmarks/public-calibration-results.zip` with archive
and member checksums. Reproduce the reports without any model fitting:

```sh
uv run python experiments/reproduce_public_calibration.py --output artifacts/public-cost-review
```

The [application protocol](PUBLIC_APPLICATION_PLAN.md) and executable config are
frozen for the 30-design, 180-method development grid. Full application results
and official-test evaluation remain future work.
