"""Verify and reproduce both public-study cost calibrations without fitting."""

import argparse
import json
from pathlib import Path

from benchmark_archive import extract_verified, verify_archive
from public_application_cost import project
from reproduce import check_summary

from pgfs.study import write_json

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT/"benchmarks/public-calibration-results.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory for extraction and refit-free cost analysis")
    args = parser.parse_args()
    manifest = verify_archive(RECORDS)
    print(f"Verified {len(manifest['files'])} public calibration evidence files.")
    if args.output is None:
        return
    if args.output.exists():
        raise ValueError("Use a new calibration reproduction directory")
    extract_verified(RECORDS, args.output)
    verification = {}
    for version in (1, 2):
        recorded = json.loads((args.output/f"experiments/reports/public-cost-preflight-v{version}.json").read_text())
        generated = project(args.output/f"results/public-cost-calibration-v{version}", recorded["planned_config"])
        roundoff = check_summary(generated, recorded, f"calibration-v{version}", allow_roundoff=True)
        verification[f"v{version}"] = {"within_compute_gate": generated["within_compute_gate"],
                                        "calibration_failures": generated["calibration_failures"],
                                        "accepted_float_differences": roundoff}
    write_json(args.output/"calibration-verification.json", verification)
    print("Both cost reports reproduced from saved measurements; no model fits or data downloads.")


if __name__ == "__main__":
    main()
