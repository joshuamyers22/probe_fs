"""Run a frozen simulation grid as independent, resumable seed shards.

From the repository root:
uv run python experiments/run_simulation_batch.py
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from pgfs.datasets import sha256
from pgfs.study import fingerprint, initialize, write_json


def run_seed(config_path: Path, output: Path, root: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "execution.log").open("a") as log:
        subprocess.run(
            [sys.executable, "-m", "pgfs.study", "--only", "simulation",
             "--config", str(config_path), "--output", str(output)],
            cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/expanded-simulation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/expanded-simulation-v1"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    config = json.loads(args.config.read_text())
    manifest = json.loads((root / "experiments/datasets.json").read_text())
    protocol_files = []
    for name, expected in config.get("protocol_artifact_sha256", {}).items():
        path = root / name
        if sha256(path) != expected:
            raise ValueError(f"Protocol artifact checksum mismatch: {name}")
        protocol_files.append(path)
    config["batch_driver_sha256"] = sha256(Path(__file__))
    provenance = initialize(output, config, manifest, root)
    write_json(output / "frozen-config.json", config)
    source_archive = output / "source-snapshot.zip"
    if not source_archive.exists():
        files = list((root / "src/pgfs").glob("*.py")) + [
            root / "pyproject.toml", root / "uv.lock", root / ".python-version",
            root / "experiments/datasets.json", args.config.resolve(), Path(__file__),
        ] + protocol_files
        with zipfile.ZipFile(source_archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            for path in files:
                zipped.write(path, str(path.relative_to(root)))
        write_json(output / "source-snapshot.json", {"sha256": sha256(source_archive)})
    else:
        expected = json.loads((output / "source-snapshot.json").read_text())["sha256"]
        if sha256(source_archive) != expected:
            raise ValueError("Source snapshot checksum mismatch")
    paths = []
    for seed in config["seeds"]:
        shard = dict(config, seeds=[seed], parent_fingerprint=provenance["fingerprint"])
        path = output / f"config-seed-{seed}.json"
        if path.exists() and fingerprint(json.loads(path.read_text())) != fingerprint(shard):
            raise ValueError(f"Changed shard configuration: {path}")
        write_json(path, shard)
        paths.append((path, output / f"seed-{seed}"))
    sim = config["simulation"]
    expected_cells = len(paths) * len(sim["n"]) * len(sim["rho"]) * len(sim["snr"]) * len(config["budgets"])
    started = time.perf_counter()
    print(f"Frozen batch: {expected_cells} pairs; {config['workers']} concurrent seed workers; one BLAS thread each", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        pending = {pool.submit(run_seed, path, destination, root): destination for path, destination in paths}
        while pending:
            done, _ = concurrent.futures.wait(pending, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                destination = pending.pop(future)
                future.result()
                print(f"Completed {destination.name}", flush=True)
            completed = len(list(output.glob("seed-*/cell-*.json")))
            print(f"Progress {completed}/{expected_cells}; elapsed {(time.perf_counter()-started)/60:.1f} min", flush=True)
    if len(list(output.glob("seed-*/cell-*.json"))) != expected_cells:
        raise ValueError("Completed checkpoint count differs from frozen design")
    print("All simulation cells completed. Generate analysis with experiments/analyze_simulation_batch.py", flush=True)


if __name__ == "__main__":
    main()
