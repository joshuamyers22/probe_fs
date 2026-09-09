# Endpoint audit v1

Status: complete. See [findings](ENDPOINT_AUDIT_RESULTS.md).

This is a post-hoc diagnosis of the completed 240-pair expanded batch. Its saved
results remain unchanged. Reusing their held-out folds makes all alternative
policy scores exploratory; these are not fresh confirmatory evaluations, new
deployable methods, or newly matched-compute method comparisons.

Before diagnostic refitting, freeze these comparisons:

1. Reproduce the saved selected-set loss in each outer fold, with identical
   training rows, learner, preprocessing and seed, to verify pairing.
2. Use the already-saved inner-CV argmin size on the already-saved outer ranking.
   This isolates the final effect of the one-standard-error size rule without
   recomputing any importance scores or using outer outcomes to choose a size.
3. Evaluate the fixed top five and fixed top six of those same rankings. Five
   is absent from the original candidate path; these diagnostics do not choose
   among sizes based on the reused test outcomes.
4. Evaluate known-truth oracle sets of sizes three, five and six, chosen from
   the DGP's population conditional variance, independent of observed outcomes.
   Oracle results are diagnostic bounds on selection difficulty and are not
   deployable feature-selection procedures.

Also derive Gaussian population prediction-risk differences for the actual
noisy-proxy generator. Check the formula against a covariance-matrix calculation.
Report how often the one-SE rule shrinks the selected set, conditional endpoint
pass rates, and the recall ceiling. Keep the original 0.05 MSE margin and size
cap of six. Do not edit the DGP or retrofit the old study's constraints.

```sh
uv run python experiments/audit_endpoint.py
uv run python experiments/plot_endpoint_audit.py
```

Outputs and an archived diagnostic script are stored in
`results/endpoint-audit-v1/`. Parent source and all parent result hashes are
verified before any diagnostic refitting begins.
