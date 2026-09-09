# Context and shadow-scope controls v1

Frozen before examining outcomes on seeds 401–410. This is a bounded diagnostic
study following the completed selection-rule sensitivity. It does not reopen a
broad methodology superiority claim: earlier studies found no persistent gating
advantage. No automatic adoption or discontinuation verdict will be drawn from
ten exploratory seeds.

## Data and common evaluation

Use all 12 revised DGP settings: n=300/1,000/3,000, rho=0.5/0.9, SNR=0.25/2.
Coefficients remain (1, 0.30, 0.15), with two noisy proxies per signal class,
one three-feature correlated null block (rho=0.6), and six independent nulls.
The established population feasibility calculation is unchanged. Generate each
dataset once per seed worker and use its shared seeded feature permutation
`derive_rng(seed, 60491)`, remapping feature names and ground truth.

All methods use the same outer/inner folds and seeds, Ridge alpha=1, within-fit
median imputation and scaling, three outer folds, two inner folds, candidate sizes
1–6, marginal shadows m=3 and quantile q=0.9 (higher empirical quantile convention).
Use the original **one-SE rule** throughout; this batch does not rerun argmin.
The loss margin is 0.05 against each arm's all-feature reference, size cap 6,
and minimum absolute class-recall difference +0.05. Retain the 5% measured-fit
matching tolerance. Failed constraints or fit matches are inconclusive.

## Eight methods on every dataset

At each partial-context budget T=2 and T=4, run:

1. Local soft-gated random-permutation contexts, one importance split.
2. Global-shadow soft gating on the same contexts and split, with the threshold
   pooled over all feature/context/shadow contributions within that split.
3. Full-conditioning gated LOCO, with **B=2 importance splits at T=2's budget**
   and **B=4 at T=4's budget**. Full conditioning always has one context.

Also run single-split full-conditioning gated LOCO and ungated LOCO once each
per dataset. These provide cheaper reference points on performance-versus-cost
curves; neither is labelled compute-matched to the partial-context methods.
Run every method independently through the complete nested evaluation, without
sharing fits. Sharing seeds means local/global raw contributions should agree;
changing shadow scope affects ranking and subsequent selection only.

For p=15, predicted importance fits per estimation are about 119.84 and 237.08
for local/global T=2/4, respectively; gated LOCO B=2/4 uses exactly 122/244.
Single-split gated and ungated LOCO use 61 and 16. The driver preflights the
actual formula and refuses a predicted mismatch greater than 5%. Every method
also spends 42 nested selection/reference fits beyond nine importance calls.
Report actual totals and wall-clock time; prediction is only a planning check.

Increasing LOCO's B spends its extra budget on independent importance splits,
because repeating the same deterministic full context is redundant. Therefore
the matched comparison changes context structure **and allocation of estimation
effort**. The single-split LOCO controls retain the same B as partial conditioning
at unequal cost. Neither comparison isolates context structure and cost at once.

This is **120 datasets × 8 methods = 960 nested evaluations**, with four workers
and one BLAS thread each. No early stopping, seed exclusion, tuning after viewing
results, or dataset/margin/selection-rule changes. Individual method checkpoints
allow a full resume without repeating fitted results.

## Analysis

Primary diagnostic contrasts are **local minus global** and **local minus
matched gated LOCO**, separately for T=2 and T=4. There are 480 paired contrasts;
each reuses a local result, so pooled contrasts are not independent trials.
For each n/rho/SNR/budget/contrast, report both-endpoint validity and compute
matches over all ten seeds, paired class recall only on valid matched seeds,
and paired MSE/size changes on all seeds. Use descriptive across-seed MCSE and
95% t intervals, with unavailable estimates for no valid seeds and unavailable
intervals for one. Do not subtract conditional means from different valid sets.
No multiplicity-adjusted or confirmatory inference is claimed.

Report every method's own constraint pass rate, all-seed MSE, selected size,
stability, measured fits and runtime. Plot these costs for both budgets and
natural-cost LOCO; natural-cost comparisons are descriptive, not qualified
matched-compute recall evidence. Audit shared all-feature references, local/global
raw outer scores and fit counts, exact ties crossing outer selection boundaries,
actual specs, feature permutations and truth remapping. A shared reference does
not mean fits were reused. Inner-CV rankings remain trained within each fold.

## Reproduction and provenance

```sh
uv sync --locked --all-extras
uv run python experiments/run_context_controls.py --config experiments/context-controls.json --output results/context-controls-v1
uv run python experiments/analyze_context_controls.py --output results/context-controls-v1
```

The config records this protocol's SHA-256. Provenance freezes the driver, core
sources, versions, configuration and dependency lock; a source ZIP includes the
protocol and build files. Analysis records its own source hash and every input
checkpoint hash. Keep full results locally under the ignored run directory and
portable summaries/plots under `experiments/reports` and `docs/figures`.
Earlier batches remain unchanged. No public or financial data are used.
