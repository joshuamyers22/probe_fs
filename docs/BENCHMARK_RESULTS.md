# Completed benchmark: evidence and interpretation

The experiments do not establish a persistent matched-compute class-recall
advantage for probe gating. They do identify useful distinctions: the original
DGP strongly constrained feasible recall, the one-SE size rule contributed to
loss-constraint failures, and partial contexts improved prediction over full
conditioning in one low-SNR regime.

Configurations were frozen before their respective fitted outcomes were examined.
Later revisions are separate exploratory studies on fresh seeds, not retroactive
changes or a confirmatory sequence.

| Study | Recorded work | Principal finding |
| --- | --- | --- |
| [Pilot](PILOT_RESULTS.md) | 16 simulation pairs and 6 public-data pairs | Only 1/16 simulation pairs met both endpoint constraints. Public data lack known-truth class recall. |
| [Expanded original DGP](EXPANDED_SIMULATION_RESULTS.md) | 240 pairs; seeds 101–110 | 79 valid; 77 ties, one gated win and one loss. Strong recall ceiling. |
| [Endpoint audit](ENDPOINT_AUDIT_RESULTS.md) | 240 diagnostic records; 6,840 additional fits | Complementary proxies, omitted size 5 and the one-SE tolerance help explain failures. Post-hoc rescores on reused folds. |
| [Revised DGP](REVISED_SIMULATION_RESULTS.md) | 240 pairs; seeds 201–210 | Weaker classes and complete size path: 140 valid; 32 wins, 19 losses, 89 ties. No persistent advantage. |
| [Selection-rule sensitivity](SELECTION_RULE_RESULTS.md) | 480 pairs; shared seeds 301–310 | Argmin increased validity from 158/240 to 211/240, selecting about 4 instead of 2.6 features. No consistent gating advantage. |
| [Context controls](CONTEXT_CONTROLS_RESULTS.md) | 960 evaluations; seeds 401–410 | Local/global sets strongly overlap. No available qualified recall interval excludes zero for either control. Partial contexts improve MSE over LOCO in one small-sample, low-SNR regime. |

The primary endpoint requires both arms to meet the loss margin and size cap,
plus compute matching. Failed endpoints remain inconclusive. Conditional effects
use valid pairs with explicit denominators. Selection-rule effects use the common
valid seed intersection. Budgets reuse datasets; pooled counts are not independent
replications of a single effect.

One-SE remains primary; argmin is a labelled sensitivity. Single-split LOCO is
reported at its cheaper actual cost. Matching LOCO's larger budgets uses extra
importance splits, changing allocation of estimation effort. Descriptive intervals
over ten seeds are neither multiplicity-adjusted nor formal equivalence tests.

The project now has a reproducible benchmark with negative, inconclusive and
localized positive findings. Application studies can assess loss/size/cost
tradeoffs, but should not imply established known-truth feature-selection
superiority. Financial datasets remain deferred.

The [reproduction guide](../REPRODUCIBILITY.md) provides verification, refit-free
reanalysis, a smoke test and historical-source staging. The
[catalog](../benchmarks/studies.json) connects each study to its records and source.
Original reports and frozen protocols remain intact.
