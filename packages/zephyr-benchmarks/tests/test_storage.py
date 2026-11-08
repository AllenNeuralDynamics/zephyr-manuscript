"""The paper's bucket: ZEPHYR_BUCKET, else the workspace pyproject.toml."""

import tomllib
from pathlib import Path

import pytest

from zephyr.benchmarks import storage
from zephyr.benchmarks.config import ResultsManifest

WORKSPACE = Path(__file__).resolve().parents[3] / "pyproject.toml"


def test_the_workspace_pyproject_names_the_bucket(monkeypatch):
    monkeypatch.delenv(storage.ENV, raising=False)
    table = tomllib.loads(WORKSPACE.read_text())["tool"][storage.TABLE]
    assert storage.bucket() == table["bucket"].rstrip("/")


def test_the_environment_overrides_it(monkeypatch):
    monkeypatch.setenv(storage.ENV, "s3://elsewhere/paper/")
    assert storage.bucket() == "s3://elsewhere/paper"
    assert storage.url("zephyr-manuscript-runs", "/analysis/") == (
        "s3://elsewhere/paper/zephyr-manuscript-runs/analysis"
    )
    assert storage.key("zephyr-dataset") == "elsewhere/paper/zephyr-dataset"


def test_no_bucket_is_an_error(monkeypatch):
    monkeypatch.delenv(storage.ENV, raising=False)
    monkeypatch.setattr(storage, "_from_pyproject", lambda: None)
    with pytest.raises(RuntimeError, match=storage.ENV):
        storage.bucket()


def test_a_manifest_root_is_under_the_bucket(monkeypatch):
    monkeypatch.setenv(storage.ENV, "s3://elsewhere")
    run = {"label": "Zephyr", "run": "face"}
    manifest = ResultsManifest(root="${ZEPHYR_BUCKET}/zephyr-runs", method=[run])
    assert manifest.root == "s3://elsewhere/zephyr-runs"
    assert ResultsManifest(root="D:/runs", method=[run]).root == "D:/runs"
