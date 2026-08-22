"""Run the Section 16 sequence on a small simulated primary regime.

Section 16: "Run the matched-compute gated-versus-ungated experiment first." This
script does exactly that, then the context ablation, then the external baselines,
then prints the decision verdict. Defaults are small enough to finish in a minute
or two; the numbers they produce are a smoke test of the pipeline, not evidence
about the method.

    python examples/run_primary_experiment.py --n 300 --budgets 2 4 --outer 3 --inner 2
"""

from __future__ import annotations

import argparse
import json
import time

from sklearn.linear_model import Ridge

from pgfs import MethodSpec, StudyConstraints, make_primary
from pgfs.baselines import DEFAULT_BASELINES, run_baseline_nested
from pgfs.experiments import (
    context_ablation,
    decision_criteria,
    matched_compute_comparison,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--class-size", type=int, default=2)
    ap.add_argument("--nulls", type=int, default=6)
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--snr", type=float, default=2.0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--shadows", type=int, default=3)
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--outer", type=int, default=3)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--splits", type=int, default=1, help="importance splits B")
    ap.add_argument("--margin", type=float, default=0.05, help="noninferiority margin delta")
    ap.add_argument("--k-max", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--json-out", type=str, default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    data = make_primary(
        n=args.n,
        n_classes=args.classes,
        class_size=args.class_size,
        n_null_blocks=1,
        null_block_size=3,
        n_independent_nulls=args.nulls,
        rho=args.rho,
        snr=args.snr,
        seed=args.seed,
    )
    print(data.describe())

    # Everything below is fixed before any result is looked at (Sections 3, 11).
    base = MethodSpec(
        learner=Ridge(alpha=1.0),
        loss="mse",
        n_importance_splits=args.splits,
        n_shadows=args.shadows,
        shadow_quantile=args.quantile,
        candidate_k=tuple(range(1, min(args.k_max, len(data.players)) + 1)),
        seed=args.seed,
        label="pgfs",
    )
    constraints = StudyConstraints(
        noninferiority_margin=args.margin, k_max=args.k_max, min_effect=0.05
    )
    print(base.describe())
    print()

    t0 = time.perf_counter()
    print("=== Section 10: matched-compute gated vs ungated ===")
    comparison = matched_compute_comparison(
        data, base, constraints,
        gated_contexts=tuple(args.budgets),
        n_outer_folds=args.outer,
        n_inner_folds=args.inner,
        verbose=True,
    )
    print(comparison.to_markdown())
    print()
    for c in comparison.comparisons:
        print(
            f"budget {c['budget']}: recall gated {c['class_recall_a']:.3f} vs "
            f"ungated {c['class_recall_b']:.3f} (diff {c['difference']:+.3f}), "
            f"fits {c['fits_gated']} vs {c['fits_ungated']} "
            f"(ratio {c['fit_ratio']:.2f}), "
            f"clears 0.05: {c['meets_minimum_effect']}"
        )
    print()

    print("=== Section 12: context ablation and direct alternatives ===")
    ablation = context_ablation(
        data, base.replace(n_contexts=max(args.budgets)), constraints,
        n_outer_folds=args.outer, n_inner_folds=args.inner, verbose=True,
    )
    print(ablation.to_markdown())
    print()

    baseline_rows = []
    if not args.skip_baselines:
        print("=== Section 12: external baselines ===")
        for name, selector in DEFAULT_BASELINES.items():
            res = run_baseline_nested(
                data.X, data.y, data.players, base, selector, name,
                n_outer_folds=args.outer, signal_classes=data.signal_classes,
            )
            rep = res.report()
            baseline_rows.append(rep)
            print(
                f"{name:>20}: loss {rep['mean_outer_loss']:.4f}, "
                f"recall {rep.get('class_recall', float('nan')):.3f}, "
                f"size {rep['mean_selected_size']:.1f}, "
                f"false {rep.get('false_selections', float('nan')):.1f}, "
                f"{rep['wall_clock_seconds']:.1f}s"
            )
        print()

    print("=== Section 16: decision criteria ===")
    verdict = decision_criteria(comparison, constraints, ablation)
    print(verdict["summary"])
    print(json.dumps({k: v for k, v in verdict.items() if k != "summary"}, indent=2, default=str))
    print(f"\ntotal wall clock: {time.perf_counter() - t0:.1f}s")

    if args.json_out:
        payload = {
            "data": data.params,
            "spec": base.as_dict(),
            "constraints": constraints.__dict__,
            "comparison_rows": comparison.rows,
            "comparisons": comparison.comparisons,
            "ablation_rows": ablation.rows,
            "baselines": baseline_rows,
            "verdict": verdict,
        }
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
