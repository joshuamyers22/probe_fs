"""Verify recorded evidence, reanalyze it, stage historical source, or run a smoke test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmark_archive import (
    copy_workspace,
    digest,
    extract_verified,
    overlay_snapshot,
    verify_archive,
)

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "benchmarks/recorded-results.zip"


def catalog():
    return json.loads((ROOT / "benchmarks/studies.json").read_text())["studies"]


def execute(workspace, *args):
    environment = dict(os.environ, PYTHONPATH=str(workspace / "src"), PYTHONNOUSERSITE="1")
    subprocess.run([sys.executable, *args], cwd=workspace, env=environment, check=True)


def unpack(output):
    extract_verified(RECORDS, output)
    for study in catalog():
        directory = output / study["directory"]
        if len(list(directory.glob(study["checkpoint_glob"]))) != study["checkpoints"]:
            raise ValueError(f"Wrong checkpoint count: {study['id']}")


def check_summary(actual, expected, label):
    """Identify differing leaves instead of hiding failures behind a study name."""
    differences = []

    def compare(a, b, path):
        if type(a) is not type(b):
            differences.append(f"{path}: types {type(a).__name__} != {type(b).__name__}")
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                differences.append(f"{path}: keys differ")
            for key in sorted(a.keys() & b.keys()):
                compare(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append(f"{path}: lengths {len(a)} != {len(b)}")
            for i, (left, right) in enumerate(zip(a, b)):
                compare(left, right, f"{path}[{i}]")
        elif a != b:
            differences.append(f"{path}: actual={a!r}, recorded={b!r}")

    compare(actual, expected, label)
    if differences:
        raise ValueError(f"Regenerated statistics differ ({len(differences)} leaves):\n" + "\n".join(differences[:12]))


def reanalyze(output):
    """Write only into a new evidence copy; never edit the original results."""
    if output.exists():
        raise ValueError("Use a new analysis output directory")
    unpack(output)
    summaries = {}
    for study in catalog():
        directory = output / study["directory"]
        if study["kind"] == "paired":
            before = json.loads((directory / "paired-summary.json").read_text())
            execute(ROOT, "experiments/analyze_simulation_batch.py", "--output", str(directory))
            check_summary(json.loads((directory / "paired-summary.json").read_text()), before, study['id'])
            summaries[study["id"]] = {"pairs": sum(r["seeds"] for r in before), "valid": sum(r["valid"] for r in before)}
        elif study["kind"] == "contexts":
            before = {name: json.loads((directory / name).read_text()) for name in
                      ("paired-summary.json", "method-summary.json", "summary.json")}
            execute(ROOT, "experiments/analyze_context_controls.py", "--output", str(directory))
            for name, expected in before.items():
                check_summary(json.loads((directory / name).read_text()), expected, f"contexts/{name}")
            summaries["contexts"] = before["summary.json"]
        elif study["kind"] == "pilot":
            from pgfs.study import summarize
            before = json.loads((directory / "aggregates.json").read_text())
            summarize(directory)
            check_summary(json.loads((directory / "aggregates.json").read_text()), before, "pilot")
            summaries["pilot"] = {"paired_cells": study["checkpoints"]}
        elif study["kind"] == "audit":
            from audit_endpoint import summarize
            before = json.loads((directory / "summary.json").read_text())
            config = json.loads((output / "results/expanded-simulation-v1/frozen-config.json").read_text())
            summarize(directory, config)
            check_summary(json.loads((directory / "summary.json").read_text()), before, "audit")
            summaries["audit"] = {"cells": before["cells"]}
    policy = output / "results/selection-rule-v1"
    before = json.loads((policy / "policy-summary.json").read_text())
    execute(ROOT, "experiments/analyze_selection_rules.py", "--output", str(policy))
    check_summary(json.loads((policy / "policy-summary.json").read_text()), before, "paired-policy")
    (output / "reanalysis-verification.json").write_text(json.dumps(summaries, indent=2, sort_keys=True)+"\n")
    print("All seven recorded study summaries and paired-policy statistics reproduced exactly without refitting.")


def stage(study_id, destination):
    study = next(s for s in catalog() if s["id"] == study_id)
    copy_workspace(ROOT, destination)
    unpack(destination)
    provenance = json.loads((destination / study["directory"] / "provenance.json").read_text())
    overlay_snapshot(destination / study["source_archive"], destination, provenance["identity"]["source_hashes"])
    python_version = provenance["identity"]["python"]
    instructions = {
        "pilot": "uv run python -m pgfs.study --config experiments/pilot.json --output results/replay-pilot --only simulation",
        "expanded": "uv run python experiments/run_simulation_batch.py --config experiments/expanded-simulation.json --output results/replay-expanded",
        "revised": "uv run python experiments/run_simulation_batch.py --config experiments/revised-simulation.json --output results/replay-revised",
        "one-se": "uv run python experiments/run_simulation_batch.py --config experiments/selection-one-se.json --output results/replay-one-se",
        "argmin": "uv run python experiments/run_simulation_batch.py --config experiments/selection-argmin.json --output results/replay-argmin",
        "audit": "uv run python experiments/audit_endpoint.py --parent results/expanded-simulation-v1 --output results/replay-audit",
        "contexts": "uv run python experiments/run_context_controls.py --config experiments/context-controls.json --output results/replay-contexts",
    }
    (destination / "REPLAY.md").write_text(
        f"# Replay {study_id}\n\nHistorical source hashes match the recorded study.\n\n"
        f"Recorded Python: {python_version}. Run `uv sync --locked --all-extras --python {python_version}`, then:\n\n```sh\n{instructions[study_id]}\n```\n\n"
        "Outputs use a new directory. Pilot public-data refits additionally require the pinned downloads; the command above runs its simulations only.\n"
    )
    print(f"Staged {study_id} at {destination}; see REPLAY.md. No scientific fits were run.")


def smoke(output):
    """Exercise complete fitting/analysis/resume paths on a tiny non-study grid."""
    workspace = output / "workspace"
    if output.exists():
        raise ValueError("Use a new smoke output directory")
    copy_workspace(ROOT, workspace)
    config = json.loads((ROOT / "experiments/selection-one-se.json").read_text())
    config.update(seeds=[9001], workers=1, outer_folds=2, inner_folds=2, budgets=[2],
                  purpose="Infrastructure smoke test only; not scientific benchmark evidence.", protocol_artifact_sha256={})
    config["simulation"].update(n=[80], rho=[.5], snr=[.25])
    for rule, flag in (("one-se", True), ("argmin", False)):
        c = dict(config, name="smoke-"+rule, one_se_rule=flag)
        path = workspace / f"experiments/smoke-{rule}.json"
        path.write_text(json.dumps(c, indent=2)+"\n")
        execute(workspace, "experiments/run_simulation_batch.py", "--config", str(path), "--output", f"results/smoke/{rule}")
        execute(workspace, "experiments/analyze_simulation_batch.py", "--output", f"results/smoke/{rule}")
    execute(workspace, "experiments/analyze_selection_rules.py", "--output", "results/smoke")
    controls = dict(config, name="smoke-contexts", rungs=[{"contexts": 2, "loco_splits": 2}, {"contexts": 4, "loco_splits": 4}])
    controls.pop("budgets")
    controls.pop("public")
    path = workspace / "experiments/smoke-contexts.json"
    path.write_text(json.dumps(controls, indent=2)+"\n")
    command = ("experiments/run_context_controls.py", "--config", str(path), "--output", "results/smoke-contexts")
    execute(workspace, *command)
    execute(workspace, "experiments/analyze_context_controls.py", "--output", "results/smoke-contexts")
    cells = sorted((workspace / "results").glob("**/cell-*.json"))
    before = {str(p): (digest(p.read_bytes()), p.stat().st_mtime_ns) for p in cells}
    execute(workspace, *command)
    for rule in ("one-se", "argmin"):
        execute(workspace, "experiments/run_simulation_batch.py", "--config", str(workspace / f"experiments/smoke-{rule}.json"), "--output", f"results/smoke/{rule}")
    after = {str(p): (digest(p.read_bytes()), p.stat().st_mtime_ns) for p in cells}
    if len(cells) != 10 or before != after:
        raise ValueError("Smoke resume changed recorded fits")
    result = {"paired_policy_cells": 2, "context_evaluations": 8,
              "resumed_cells_unchanged": 10, "scientific_evidence": False}
    (output / "smoke-verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print("Smoke passed: both selection policies, all context controls, paired analysis, and unchanged resume.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    extract = sub.add_parser("unpack")
    extract.add_argument("--output", type=Path, required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--output", type=Path, required=True)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("stage")
    prepare.add_argument("--study", choices=[s["id"] for s in catalog()], required=True)
    prepare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        manifest = verify_archive(RECORDS)
        print(f"Verified {len(manifest['files'])} recorded files against archive and member checksums.")
    elif args.command == "unpack":
        unpack(args.output.resolve())
    elif args.command == "analyze":
        reanalyze(args.output.resolve())
    elif args.command == "stage":
        stage(args.study, args.output.resolve())
    else:
        smoke(args.output.resolve())


if __name__ == "__main__":
    main()
