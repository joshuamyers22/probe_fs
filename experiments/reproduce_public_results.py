"""Verify and reanalyze the complete public application evidence without refitting."""

import argparse
import json
from pathlib import Path

from analyze_public_application import analyze
from benchmark_archive import extract_verified, verify_archive
from reproduce import check_summary

from pgfs.study import write_json

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "benchmarks/public-application-results-v1.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory for extraction and refit-free analysis")
    args = parser.parse_args()
    manifest = verify_archive(RECORDS)
    print(f"Verified {len(manifest['files'])} public application evidence files.")
    if args.output is None:
        return
    if args.output.exists():
        raise ValueError("Use a new public-results reproduction directory")
    extract_verified(RECORDS, args.output)
    recorded = args.output / "results/public-application-v1"
    generated = args.output / "regenerated-analysis"
    analyze(recorded, generated)
    differences = []
    for name in ("methods.json", "contrasts.json", "summary.json"):
        expected = json.loads((recorded / "analysis" / name).read_text())
        actual = json.loads((generated / name).read_text())
        differences.extend(check_summary(actual, expected, name, allow_roundoff=True))
    methods = json.loads((generated / "methods.json").read_text())
    write_json(args.output / "public-results-verification.json", {
        "method_records": len(methods), "failures": sum(r["status"] == "failed" for r in methods),
        "relative_tolerance": 1e-12, "absolute_tolerance": 1e-14,
        "accepted_float_differences": differences,
    })
    print("Public application tables and figures reproduced from checkpoints without model fitting or data downloads.")


if __name__ == "__main__":
    main()
