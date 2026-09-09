"""Validate calibration completeness and project cost without comparing outcomes."""

import argparse
import itertools
import json
from pathlib import Path

from pgfs.datasets import sha256
from pgfs.study import fingerprint, write_json


def project(calibration, planned):
    config = json.loads((calibration/"frozen-config.json").read_text())
    provenance = json.loads((calibration/"provenance.json").read_text())
    if (config != provenance["identity"]["config"] or fingerprint(provenance["identity"]) != provenance["fingerprint"]
            or config["scientific_evidence"] or config["rows"] != [600] or config["seeds"] != [599]):
        raise ValueError("Invalid calibration identity or design")
    for key in ("datasets", "budgets", "outer_folds", "inner_folds", "importance_splits", "shadows",
                "quantile", "one_se_rule", "candidate_k", "include_all_features", "l1"):
        if planned[key] != config[key]:
            raise ValueError(f"Planned method differs from cost calibration: {key}")
    methods = {"all", "l1"} | {f"{arm}-T{t}" for arm in ("gated", "ungated") for t in config["budgets"]}
    expected = set(itertools.product(config["datasets"], methods))
    observed, inputs, details = set(), {}, []
    for path in sorted(calibration.glob("*/cell-*.json")):
        record = json.loads(path.read_text())
        design = record["design"]
        key = (design["dataset"], record["method"])
        content = dict(record)
        recorded_hash = content.pop("record_sha256")
        if fingerprint(content) != recorded_hash or record["run_fingerprint"] != provenance["fingerprint"]:
            raise ValueError("Changed calibration checkpoint")
        if sha256(path.parent/"data.json") != record["data_sha256"]:
            raise ValueError("Changed calibration row/fold data")
        if key not in expected or key in observed or design["rows"] != 600 or design["seed"] != 599:
            raise ValueError("Unexpected calibration checkpoint")
        observed.add(key)
        inputs[str(path.relative_to(calibration))] = sha256(path)
        details.append({"dataset": key[0], "method": key[1], "status": record["status"],
                        "seconds": record["elapsed_seconds"], "peak_memory_gib": record["peak_memory_gib"],
                        "model_fits": record.get("report", {}).get("main_results", {}).get("model_fits_total"),
                        "error_type": record.get("error_type"), "error": record.get("error"),
                        "warnings": record["warnings"]})
    if expected != observed:
        raise ValueError("Incomplete calibration")
    failures = sum(r["status"] != "complete" for r in details)
    worker_time = json.loads((calibration/"worker-time.json").read_text())
    # Use first invocation per design: resume costs do not inflate original work.
    first_invocations = {}
    for row in worker_time["invocations"]:
        first_invocations.setdefault(row["design"]["dataset"], row["seconds"])
    if set(first_invocations) != set(config["datasets"]):
        raise ValueError("Missing worker elapsed-time measurements")
    projections = []
    for dataset in config["datasets"]:
        measured_peak = max(r["peak_memory_gib"] for r in details if r["dataset"] == dataset)
        p = provenance["identity"]["datasets"][dataset]["features"]
        for n in planned["rows"]:
            projections.append({"dataset": dataset, "rows": n, "seeds": len(planned["seeds"]),
                                "worker_seconds": 2*first_invocations[dataset]*(n/600)*len(planned["seeds"]),
                                "estimated_peak_memory_gib": 2*measured_peak + 10*n*p*8/(1024**3)})
    hours = sum(r["worker_seconds"] for r in projections)/3600
    peak = max(r["estimated_peak_memory_gib"] for r in projections)
    return {"scientific_evidence": False, "calibration_config": config, "planned_config": planned,
            "calibration_fingerprint": provenance["fingerprint"], "checkpoint_sha256": inputs,
            "calibration_methods": details, "calibration_failures": failures, "projections": projections,
            "estimated_worker_hours": hours, "estimated_peak_memory_gib": peak,
            "within_compute_gate": failures == 0 and hours <= planned["worker_hours_cap"] and peak <= planned["memory_gib_cap"],
            "assumptions": ["Two times linear row scaling of first measured worker wall time, including loading/startup.",
                            "Memory: twice observed process peak plus ten dense full-size float64 arrays.",
                            "Heuristic estimates, not resource guarantees; convergence and conditioning can change with n.",
                            "One isolated worker; elapsed-time cap enforced by parent timeout; memory checked between methods.",
                            "No predictive losses or method rankings used in the cost gate."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("experiments/public-application.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = project(args.calibration, json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("calibration_failures", "estimated_worker_hours",
                                                   "estimated_peak_memory_gib", "within_compute_gate")}, indent=2))


if __name__ == "__main__":
    main()
