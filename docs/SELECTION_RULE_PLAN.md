# Selection-rule sensitivity v1

Frozen before fitting any outcomes on seeds 301–310. The question is whether
inner-CV argmin improves endpoint validity relative to the one-standard-error
rule, and whether any gated-versus-ungated recall advantage persists under both
rules. Argmin is a sensitivity analysis that deviates from the primary method.

## Paired design

Use the complete revised simulation grid: n=300/1,000/3,000, rho=0.5/0.9,
SNR=0.25/2, gated contexts=2/4, seeds=301–310. The class coefficients are
(1, 0.30, 0.15), with two features per class, three correlated nulls (rho=0.6),
and six independent nulls. Keep the Gaussian noisy-proxy generator and Ridge
alpha=1 with within-fit median imputation and scaling. The population feasibility
calculation from the revised benchmark still applies.

For each generated dataset, permute its 15 columns once using the child RNG
`derive_rng(seed, 60491)`, without consulting X, y or class membership. Remap player
names and truth indices to preserve the original classes. Both rules and both
budgets use the same permutation for a seed; it also remains fixed across n/rho/SNR.
This removes the fixed signal-first preference of column-order tie breaking but
does not prove order invariance. Record the permutation in every checkpoint.
This fresh paired one-SE baseline, rather than an older result with another seed
or ordering, is the comparator for the argmin sensitivity.

Run two complete nested evaluations per method/design: one-SE and inner-CV
argmin. Both use identical seeds, three outer folds, two inner folds, candidate
sizes 1–6, one importance split, three shadows, quantile 0.9, and the existing
local soft gate versus ungated matched-compute comparison. No fitted models are
shared between policies. The only policy-specific method setting is
`one_se_rule`: true versus false. Verify identical outer rankings, inner mean-loss
paths and reference losses across policies, and verify argmin actually selects
the recorded minimizing k. Equal inner-loss minima retain the existing smallest
candidate tie rule. Estimation seeds do not depend on the policy flag.

Keep the endpoint: both arms' mean outer MSE <= their all-feature reference +0.05,
size cap 6, minimum recall difference +0.05, measured fit imbalance <=5%.
Freeze **240 gated/ungated pairs per policy, 480 total**, on 120 underlying
datasets. Use two seed workers per policy, four total, one BLAS thread each.
Do not stop early, change margins, remove seeds, select only the previous winning
cell, or modify policies after inspecting outcomes.

## Analysis fixed before execution

Report each policy's original qualified endpoint summaries separately across all
24 n/rho/SNR/budget groups. Pair policies by the full design including seed.
Report valid-under-both, newly valid under argmin, validity lost under argmin,
and invalid-under-both counts, plus the across-seed paired validity difference.

For each arm, report argmin-minus-one-SE outer loss and selected size across all
ten seeds per group. Compare gated-minus-ungated recall effects across policies
only on their **common valid seed intersection**, with the intersection count.
Never subtract conditional means calculated on different valid seed subsets.
Use descriptive MCSE and 95% t intervals across seeds; with one valid seed the
interval is unavailable, with none the estimate is unavailable. No multiplicity
adjustment or confirmatory superiority decision is made. Budget groups reuse
datasets and must not be treated as independent replications in pooled inference.

Audit exact ties at outer selection boundaries separately. Changing feature order
can affect contexts as well as ties; differences from the previous batch cannot
be attributed to this change alone. Within this batch both policies share order.

## Reproduction

From the repository root, install locked dependencies and run both commands
(concurrently if four workers are desired):

```sh
uv sync --locked --all-extras
uv run python experiments/run_simulation_batch.py --config experiments/selection-one-se.json --output results/selection-rule-v1/one-se
uv run python experiments/run_simulation_batch.py --config experiments/selection-argmin.json --output results/selection-rule-v1/argmin
uv run python experiments/analyze_simulation_batch.py --output results/selection-rule-v1/one-se
uv run python experiments/analyze_simulation_batch.py --output results/selection-rule-v1/argmin
uv run python experiments/analyze_selection_rules.py --output results/selection-rule-v1
```

Each policy config records this protocol's SHA-256, and its run directory receives
a copy before launch. Batch provenance freezes all core source, configuration,
dependency lock and package versions. Both source ZIPs and input-cell hashes are
retained. Full checkpoints are local ignored artifacts; portable tables, the
paired-policy report and a plot are kept in the repository. Public and financial
datasets are outside this sensitivity study. Earlier outputs remain unchanged;
their saved source snapshots are needed to resume them after core changes.
