"""Standard-library-only archive integrity and deterministic packaging helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import PurePosixPath

ROOT_FILES = ("README.md", "REPRODUCIBILITY.md", "Makefile", "MANIFEST.in", "pyproject.toml",
              "uv.lock", ".python-version", ".gitignore", "probe-gated-feature-selection-clean.md")
SOURCE_DIRS = ("src", "tests", "examples", "experiments", "docs", ".github", "benchmarks")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name or
            ":" in name or str(path) != name):
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def source_paths(root, include_records=True):
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for directory in SOURCE_DIRS:
        for path in (root / directory).rglob("*"):
            if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if any(part.endswith(".egg-info") for part in path.parts):
                continue
            if not include_records and path.name in ("recorded-results.zip", "recorded-results.sha256"):
                continue
            paths.append(path)
    return sorted(set(paths))


def write_archive(destination, root, paths):
    """Store a manifest for every member and normalize timestamps/order."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    manifest = {"format": 1, "files": {}}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(paths):
            name = str(path.relative_to(root))
            safe_name(name)
            data = path.read_bytes()
            manifest["files"][name] = {"sha256": digest(data), "bytes": len(data)}
            item = zipfile.ZipInfo(name, date_time=(2026, 9, 9, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            archive.writestr(item, data, compresslevel=9)
        item = zipfile.ZipInfo("BUNDLE-MANIFEST.json", date_time=(2026, 9, 9, 0, 0, 0))
        item.compress_type = zipfile.ZIP_DEFLATED
        item.external_attr = 0o100644 << 16
        archive.writestr(item, json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    temporary.replace(destination)
    destination.with_suffix(".sha256").write_text(digest(destination.read_bytes())+"  "+destination.name+"\n")
    return manifest


def verify_archive(path):
    expected = path.with_suffix(".sha256").read_text().split()[0]
    if digest(path.read_bytes()) != expected:
        raise ValueError(f"Archive checksum mismatch: {path.name}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for name in names:
            safe_name(name)
        manifest = json.loads(archive.read("BUNDLE-MANIFEST.json"))
        if manifest["format"] != 1 or set(names) != set(manifest["files"]) | {"BUNDLE-MANIFEST.json"}:
            raise ValueError("Archive contents differ from manifest")
        for name, entry in manifest["files"].items():
            data = archive.read(name)
            if len(data) != entry["bytes"] or digest(data) != entry["sha256"]:
                raise ValueError(f"Member checksum mismatch: {name}")
    return manifest


def extract_verified(path, destination):
    manifest = verify_archive(path)
    destination = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        for name in manifest["files"]:
            target = destination.joinpath(*safe_name(name).parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"Extraction escapes destination: {name}")
            if target.exists() and digest(target.read_bytes()) != manifest["files"][name]["sha256"]:
                raise ValueError(f"Refusing to overwrite changed file: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(archive.read(name))
    return manifest


def copy_workspace(root, destination):
    if destination.exists():
        raise ValueError(f"Use a new workspace directory: {destination}")
    destination.mkdir(parents=True)
    for path in source_paths(root, include_records=False):
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    # Historical provenance requires a Git HEAD. This isolated, empty commit
    # supplies it without committing user work or touching the original repo.
    subprocess.run(["git", "init", "--quiet", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "-c", "user.name=Benchmark reproduction",
                    "-c", "user.email=reproduction@example.invalid", "-c", "commit.gpgsign=false",
                    "commit", "--quiet", "--allow-empty", "-m", "Initialize isolated reproduction workspace"], check=True)


def overlay_snapshot(path, destination, source_hashes):
    """Overlay a verified historical snapshot in an isolated workspace only."""
    expected = json.loads(path.with_suffix(".json").read_text())["sha256"]
    if digest(path.read_bytes()) != expected:
        raise ValueError("Historical source archive checksum mismatch")
    destination = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Duplicate snapshot members")
        for name in archive.namelist():
            target = destination.joinpath(*safe_name(name).parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Snapshot path escapes workspace")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    for name, expected in source_hashes.items():
        if digest((destination / name).read_bytes()) != expected:
            raise ValueError(f"Historical source does not match recorded study: {name}")
