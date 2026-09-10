"""Verify and regenerate size-rule sensitivity results without data or refits."""

import argparse
from pathlib import Path

from analyze_public_selection import analyze
from benchmark_archive import extract_verified, verify_archive
from reproduce import check_summary
from run_public_selection import BASE_ARCHIVE, read

from pgfs.datasets import sha256
from pgfs.study import write_json

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "benchmarks/public-selection-results-v1.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory for verified extraction and analysis")
    args = parser.parse_args()
    manifest = verify_archive(RECORDS)
    verify_archive(BASE_ARCHIVE)
    print(f"Verified {len(manifest['files'])} selection-sensitivity evidence files and prerequisite public evidence.")
    if args.output is None:
        return
    if args.output.exists():
        raise ValueError("Use a new selection-sensitivity reproduction directory")
    extract_verified(RECORDS, args.output)
    recorded = args.output / "results/public-selection-v1"
    config = read(recorded/"frozen-config.json")
    if sha256(BASE_ARCHIVE) != config["baseline_archive_sha256"]:
        raise ValueError("Changed prerequisite public evidence")
    snapshot = verify_archive(recorded/"source-snapshot.zip")
    for name, expected in config["protocol_artifact_sha256"].items():
        if snapshot["files"][name]["sha256"] != expected:
            raise ValueError(f"Frozen source snapshot differs: {name}")
    extract_verified(BASE_ARCHIVE, recorded/"baseline")
    generated = args.output/"regenerated-analysis"
    analyze(recorded, generated)
    differences = []
    for name in ("methods.json", "folds.json", "contrasts.json", "summary.json"):
        differences.extend(check_summary(read(generated/name), read(recorded/"analysis"/name), name, allow_roundoff=True))
    methods = read(generated/"methods.json")
    write_json(args.output/"public-selection-verification.json", {
        "paired_method_records": len(methods), "failures": sum(r["status"] == "failed" for r in methods),
        "new_model_fits": sum(r["new_model_fits"] for r in methods),
        "relative_tolerance": 1e-12, "absolute_tolerance": 1e-14, "accepted_float_differences": differences,
    })
    print("Paired selection tables and figures reproduced without data downloads or model fitting.")


if __name__ == "__main__":
    main()
