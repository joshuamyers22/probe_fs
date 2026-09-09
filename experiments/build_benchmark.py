"""Build the recorded-evidence archive and a reviewable source workspace bundle."""

import argparse
import json
from pathlib import Path

from benchmark_archive import digest, source_paths, verify_archive, write_archive


def evidence_paths(root):
    catalog = json.loads((root / "benchmarks/studies.json").read_text())
    paths = set()
    for study in catalog["studies"]:
        directory = root / study["directory"]
        cells = list(directory.glob(study["checkpoint_glob"]))
        if len(cells) != study["checkpoints"]:
            raise ValueError(f"Incomplete recorded study: {study['id']}")
        for cell in cells:
            record = json.loads(cell.read_text())
            provenance = json.loads((cell.parent / "provenance.json").read_text())
            if record["run_fingerprint"] != provenance["fingerprint"]:
                raise ValueError(f"Checkpoint identity mismatch: {cell}")
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".csv", ".md", ".py", ".zip") and "__pycache__" not in path.parts:
                paths.add(path)
        analysis = directory / "analysis-provenance.json"
        if analysis.exists():
            for name, expected in json.loads(analysis.read_text()).get("cell_sha256", {}).items():
                if digest((directory / name).read_bytes()) != expected:
                    raise ValueError(f"Recorded analysis input changed: {directory / name}")
    policy_root = root / "results/selection-rule-v1"
    for path in policy_root.iterdir():
        if path.is_file() and path.suffix in (".json", ".csv", ".md", ".py"):
            paths.add(path)
    policy = json.loads((policy_root / "policy-analysis-provenance.json").read_text())
    for name, expected in policy["cell_sha256"].items():
        if digest((policy_root / name).read_bytes()) != expected:
            raise ValueError("Selection-policy evidence changed")
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/probe-fs-reproducibility-v1.zip"))
    parser.add_argument("--refresh-records", action="store_true",
                        help="Rebuild evidence from complete local results; not needed for source-only rebuilds")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = root / "benchmarks/recorded-results.zip"
    if args.refresh_records:
        write_archive(evidence, root, evidence_paths(root))
    recorded = verify_archive(evidence)
    bundle = write_archive(args.output.resolve(), root, source_paths(root))
    print(f"Evidence: {len(recorded['files'])} files, {evidence.stat().st_size:,} bytes")
    print(f"Workspace: {len(bundle['files'])} files, {args.output.stat().st_size:,} bytes; {args.output}")


if __name__ == "__main__":
    main()
