"""Portable evidence must be deterministic, complete, and safe to extract."""

import json
import runpy
import zipfile

import pytest


@pytest.fixture
def archive(monkeypatch):
    monkeypatch.syspath_prepend("experiments")
    return runpy.run_path("experiments/benchmark_archive.py")


def fixture_archive(archive, tmp_path):
    source = tmp_path / "source"
    (source / "results").mkdir(parents=True)
    cell = source / "results/cell.json"
    cell.write_text('{"loss": 0.5}\n')
    path = tmp_path / "evidence.zip"
    archive["write_archive"](path, source, [cell])
    return path, source, cell


def test_repeat_build_is_identical_and_verified_extraction_preserves_files(archive, tmp_path):
    path, source, cell = fixture_archive(archive, tmp_path)
    before = path.read_bytes()
    archive["write_archive"](path, source, [cell])
    assert path.read_bytes() == before
    out = tmp_path / "unpacked"
    archive["extract_verified"](path, out)
    extracted = out / "results/cell.json"
    assert extracted.read_bytes() == cell.read_bytes()
    timestamp = extracted.stat().st_mtime_ns
    archive["extract_verified"](path, out)
    assert extracted.stat().st_mtime_ns == timestamp


def test_changed_archive_and_changed_existing_result_are_rejected(archive, tmp_path):
    path, _, _ = fixture_archive(archive, tmp_path)
    out = tmp_path / "unpacked"
    archive["extract_verified"](path, out)
    (out / "results/cell.json").write_text("changed")
    with pytest.raises(ValueError, match="overwrite changed"):
        archive["extract_verified"](path, out)
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="Archive checksum"):
        archive["verify_archive"](path)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "results/../escape", "C:/escape", "results\\escape"])
def test_unsafe_member_names_rejected_even_with_correct_outer_checksum(archive, tmp_path, name):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr(name, b"data")
        zipped.writestr("BUNDLE-MANIFEST.json", json.dumps({"format": 1, "files": {
            name: {"sha256": archive["digest"](b"data"), "bytes": 4}}}))
    path.with_suffix(".sha256").write_text(archive["digest"](path.read_bytes()))
    with pytest.raises(ValueError, match="Unsafe"):
        archive["extract_verified"](path, tmp_path / "out")
    assert not (tmp_path / "escape").exists()


def test_inner_manifest_detects_changed_member_even_if_outer_checksum_updated(archive, tmp_path):
    path, _, _ = fixture_archive(archive, tmp_path)
    with zipfile.ZipFile(path) as zipped:
        manifest = zipped.read("BUNDLE-MANIFEST.json")
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr("BUNDLE-MANIFEST.json", manifest)
        zipped.writestr("results/cell.json", b"changed")
    path.with_suffix(".sha256").write_text(archive["digest"](path.read_bytes()))
    with pytest.raises(ValueError, match="Member checksum"):
        archive["verify_archive"](path)
