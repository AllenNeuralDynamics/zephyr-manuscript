import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")

with app.setup:
    import datetime as dt
    import hashlib
    import importlib.metadata
    import json
    import os
    import platform
    import re
    import subprocess
    import sys
    import tomllib
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import s3fs
    import torch

    from utils import test_sets
    from zephyr import features
    from zephyr.benchmarks import CONFIGS, cli, probe, storage
    from zephyr.config import PATH_MAP_ENV, ClipList, ResolvedClip, canonical, load
    from zephyr.quality_control.report import _spread_out

    BUCKET = storage.key()
    """The paper's bucket: ``ZEPHYR_BUCKET``, else ``[tool.zephyr-manuscript]`` in
    ``pyproject.toml`` (:mod:`zephyr.benchmarks.storage`)."""
    RUNS = f"{BUCKET}/zephyr-manuscript-runs"
    """The paper's runs; the analyses go to its ``analysis/<name>/``."""
    CACHE_S3 = f"{BUCKET}/zephyr-cache-v2"
    """The feature cache the runs trained and were scored on, one folder per recipe."""
    CHECKPOINT = "all-cameras/cv_0/seed-17"
    """The network both analyses use: the all-camera network of cv_0."""
    MANUSCRIPT = CONFIGS / "manuscript"
    OUT = test_sets.CACHE / "analysis"
    CACHE = OUT / "cache"
    """The cache entries of the clips scored here, downloaded from :data:`CACHE_S3`."""
    DATASETS = {
        "/data/zephyr-dataset": (f"{BUCKET}/zephyr-dataset", OUT / "zephyr-dataset"),
        "/data/zephyr-df-dataset": (
            f"{BUCKET}/zephyr-df-dataset",
            OUT / "zephyr-df-dataset",
        ),
    }
    """Where the capsule mounts each dataset (its cache keys name it): its copy on S3, and
    the local folder that holds the parts of it read here, named as the mount."""
    REPO = Path(__file__).resolve().parents[1]
    NOISE = "noise-null"
    DF_SCORES = "dynamic-foraging-unrefined-scores"
    os.environ[PATH_MAP_ENV] = ";".join(
        f"{local.as_posix()}={mount}" for mount, (_, local) in DATASETS.items()
    )
    fs = s3fs.S3FileSystem()

    def sha256(path: Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def get(source: str, local: Path) -> Path:
        """*source* on S3, downloaded once to *local*."""
        if not local.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            fs.get(source, str(local))
        return local

    def fetch(relative: str) -> Path:
        """A file of the runs, downloaded once into ``cache/analysis/checkpoints/``."""
        return get(f"{RUNS}/{relative}", OUT / "checkpoints" / relative)

    def experiment(name: str) -> dict:
        return tomllib.loads((MANUSCRIPT / "experiments" / name).read_text())

    def test_selection(name: str, fold: str, group: str) -> dict:
        """The ``clips`` and ``groups`` of test group *group* of *fold* in *name*."""
        (found,) = [f for f in experiment(name)["fold"] if f["name"] == fold]
        return found["test"][group]

    def remote(path: Path) -> str:
        """The S3 object of a file of a local dataset folder (:data:`DATASETS`)."""
        mount = canonical(path)
        for root, (bucket, _) in DATASETS.items():
            if mount.startswith(root + "/"):
                return bucket + mount.removeprefix(root)
        raise ValueError(f"{path} is in none of {sorted(DATASETS)}")

    def cache_folder(clip: ResolvedClip, recipe) -> str:
        """The cache folder *clip*'s entry is in on S3, as the capsule names it: the
        recipe's, with the camera when it is not the face's."""
        params = recipe.for_clip(clip)
        w, h = params.target_size
        bw, bh = clip.box[2:]
        name = f"features-{w}x{h}-box{bw}x{bh}-{params.select_fs:g}hz"
        camera = re.split(r"_span\d+_", clip.source.stem)[-1]
        return name if camera == "FaceCamera" else f"{name}-{camera}"

    def download(clips: list[ResolvedClip], recipe) -> None:
        """What scoring *clips* reads, and only that: each clip's thermistor and
        timestamps from its dataset on S3, and its cache entry into :data:`CACHE`
        (frames are read from the cache, so no video)."""
        for c in clips:
            for path in (c.thermistor, c.timestamps):
                if path is not None:
                    get(remote(path), Path(path))
            folder = f"{CACHE_S3}/{cache_folder(c, recipe)}"
            for path in features.paths(CACHE, features.prefix(c, recipe)).values():
                if fs.exists(f"{folder}/{path.name}"):
                    get(f"{folder}/{path.name}", path)

    def dataset_list(mount: str) -> Path:
        """A dataset's ``clips.toml``, downloaded into its local folder."""
        bucket, local = DATASETS[mount]
        return get(f"{bucket}/clips.toml", local / "clips.toml")

    def manuscript_list(name: str, groups: list[str], path: Path) -> Path:
        """A clip list of the manuscript's configs (absolute paths under a mount) cut to
        *groups*, its paths moved to the datasets' local folders; written to *path*,
        which ``zephyr evaluate`` scores whole."""
        source = MANUSCRIPT / "clips" / name
        text = source.read_text()
        clips = [c for c in tomllib.loads(text)["clip"] if c["group"] in groups]

        def local(value: str) -> str:
            for mount, (_, folder) in DATASETS.items():
                value = value.replace(mount, folder.as_posix())
            return value

        lines = [f"# {source.name} cut to {len(groups)} sessions, paths moved here", ""]
        lines += [text[: text.index("[[clip]]")].rstrip(), ""]
        for c in clips:
            lines.append("[[clip]]")
            for key in ("video", "timestamps", "thermistor"):
                lines.append(f"{key} = {json.dumps(local(c[key]))}")
            lines.append(f"group = {json.dumps(c['group'])}")
            lines.append(f"box = {json.dumps(c['box'])}")
            lines.append("")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines))
        return path

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", "-C", str(REPO), *args], check=False, capture_output=True, text=True
        )
        return done.stdout.strip()

    def record(name: str, **analysis) -> dict:
        """``analysis.json`` of *name*: what it ran, on which checkpoint."""
        model = fetch(f"{CHECKPOINT}/best.pt")
        out = {"checkpoint": CHECKPOINT, "checkpoint_sha256": sha256(model), **analysis}
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "analysis.json").write_text(json.dumps(out, indent=2))
        return out

    def provenance(folder: Path, started: float, **extra) -> None:
        """``provenance.json`` of a folder made here: the commit of this repository, the
        packages, the machine and the time it took."""
        now = dt.datetime.now(dt.UTC).timestamp()
        packages = {}
        for n in ("zephyr", "zephyr-benchmarks", "torch"):
            try:
                packages[n] = {"version": importlib.metadata.version(n)}
            except importlib.metadata.PackageNotFoundError:
                packages[n] = None
        rec = {
            "notebook": "notebooks/checkpoint_analyses.py",
            "repository": git("remote", "get-url", "origin"),
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "started_utc": dt.datetime.fromtimestamp(started, dt.UTC).isoformat(
                timespec="seconds"
            ),
            "finished_utc": dt.datetime.fromtimestamp(now, dt.UTC).isoformat(
                timespec="seconds"
            ),
            "wall_time_s": round(now - started),
            "machine": {
                "cpu": platform.processor(),
                "gpu": torch.cuda.get_device_name()
                if torch.cuda.is_available()
                else None,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "python": sys.version.split()[0],
            },
            "packages": packages,
            **extra,
        }
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "provenance.json").write_text(json.dumps(rec, indent=2))


@app.cell
def _():
    mo.md(f"""
    # Analyses of a finished network

    Two analyses the paper reads that no training run makes, run here on this machine
    from the all-camera network of cv_0 (`{CHECKPOINT}` in `s3://{RUNS}/`), then
    uploaded beside the runs into `s3://{RUNS}/analysis/<name>/`:

    1. **{NOISE}** (Figs. 4-5, S9): the quality-control null. The network predicts 24
       face-camera clips of cv_0's held-out animals (test A) again with Gaussian noise of
       4, 8, 16, 32 and 64 gray levels added to every frame (`zephyr-benchmarks probe`,
       seed 0, each level's trace saved).
    2. **{DF_SCORES}** (Fig. 3g-i, Fig. S8 "before"): the network scored as it is on each
       held-out dynamic-foraging animal of `dynamic-foraging.toml`'s three folds
       (`zephyr-benchmarks evaluate`, group `df_test_a`, as the refined runs score it).

    Each output gets `analysis.json` (the checkpoint and its SHA-256, what ran) and
    `provenance.json` (this repository's commit, packages, GPU, time). Everything is made
    in `notebooks/cache/analysis/` first; nothing reaches S3 until **Upload**. Receptive
    fields and timing stay in `manuscript_figures.py`, which computes them from the runs.

    Everything is read from S3, and only what scoring reads: each clip's thermistor and
    timestamps from its dataset (`s3://{BUCKET}/zephyr-dataset`, `zephyr-df-dataset`) and
    its cache entry from `s3://{CACHE_S3}`, the cache the runs were trained and scored on,
    into `notebooks/cache/analysis/` (no videos).
    """)
    return


@app.cell
def _():
    model = fetch(f"{CHECKPOINT}/best.pt")
    mo.md(f"Checkpoint `{model}`, SHA-256 `{sha256(model)}`.")
    return (model,)


@app.cell
def _():
    mo.md(r"""
    ## 1. Noise null
    """)
    return


@app.cell
def _():
    run_noise = mo.ui.run_button(label="Run the noise null")
    run_noise
    return (run_noise,)


@app.cell
def _(model, run_noise):
    noise_dir = OUT / NOISE / "test_a"
    if run_noise.value:
        _started = dt.datetime.now(dt.UTC).timestamp()
        _groups = test_selection("face.toml", "cv_0", "test_a")["groups"]
        _list = dataset_list("/data/zephyr-dataset")
        _clips, _recipe = probe._selections([_list], _groups)
        download(_spread_out(_clips, 24), _recipe)  # the clips probe picks
        _argv = [
            "--checkpoint",
            str(model),
            "--clips",
            str(_list),
            "--groups",
            *_groups,
        ]
        _argv += ["--cache", str(CACHE), "--out", str(noise_dir)]
        _argv += ["--max-clips", "24", "--save-traces", "--seed", "0", "--amp", "bf16"]
        record(
            NOISE,
            command="zephyr-benchmarks probe",
            experiment="face.toml",
            fold="cv_0",
            sets=["test_a"],
        )
        probe.main(_argv)
        provenance(noise_dir, _started, argv=["probe", *_argv])
    noise = (
        probe.load_probes(noise_dir) if any(noise_dir.glob("*.probe.json")) else None
    )
    mo.md("Not run yet.") if noise is None else noise
    return


@app.cell
def _():
    mo.md(r"""
    ## 2. Dynamic foraging before refining
    """)
    return


@app.cell
def _():
    run_df = mo.ui.run_button(label="Score the dynamic-foraging animals")
    run_df
    return (run_df,)


@app.cell
def _(model, run_df):
    df_folds = [f["name"] for f in experiment("dynamic-foraging.toml")["fold"]]
    if run_df.value:
        record(
            DF_SCORES,
            command="zephyr-benchmarks evaluate",
            experiment="dynamic-foraging.toml",
            folds=df_folds,
            sets=["df_test_a"],
        )
        for _fold in df_folds:
            _started = dt.datetime.now(dt.UTC).timestamp()
            _run = OUT / DF_SCORES / _fold / "seed-17"
            _groups = test_selection("dynamic-foraging.toml", _fold, "df_test_a")[
                "groups"
            ]
            _list = manuscript_list("clips-df.toml", _groups, _run / "df_test_a.toml")
            _clips = load(ClipList, _list, check_files=False)
            download(_clips.resolve(), _clips.preprocess)
            _argv = [
                "evaluate",
                "--checkpoint",
                str(model),
                "--clips",
                str(_list),
            ]
            _argv += ["--name", "df_test_a", "--cache", str(CACHE), "--amp", "bf16"]
            _argv += ["--out", str(_run / "evaluation.json")]
            cli.run(_argv)
            provenance(_run, _started, argv=_argv, groups=_groups)
    _rows = []
    for _fold in df_folds:
        _path = OUT / DF_SCORES / _fold / "seed-17" / "evaluation.json"
        if _path.is_file():
            _summary = json.loads(_path.read_text())["groups"]["df_test_a"]["summary"]
            _rows.append(
                {"fold": _fold}
                | {k: v for k, v in _summary.items() if isinstance(v, float)}
            )
    mo.md("Not run yet.") if not _rows else pd.DataFrame(_rows)
    return


@app.cell
def _():
    mo.md(r"""
    ## 3. Upload
    """)
    return


@app.cell
def _():
    upload = mo.ui.run_button(label=f"Upload to s3://{RUNS}/analysis/")
    upload
    return (upload,)


@app.cell
def _(upload):
    mo.stop(not upload.value)
    _done = []
    for _name in (NOISE, DF_SCORES):
        _local = OUT / _name
        if not (_local / "analysis.json").is_file():
            continue
        _dest = f"{RUNS}/analysis/{_name}"
        _mine = json.loads((_local / "analysis.json").read_text())
        if fs.exists(f"{_dest}/analysis.json"):
            _there = json.loads(fs.cat(f"{_dest}/analysis.json"))
            if _there.get("checkpoint_sha256") != _mine["checkpoint_sha256"]:
                raise SystemExit(
                    f"{_dest} holds another checkpoint's analysis: not overwritten"
                )
        _files = [p for p in sorted(_local.rglob("*")) if p.is_file()]
        for _path in _files:
            fs.put_file(str(_path), f"{_dest}/{_path.relative_to(_local).as_posix()}")
        _done.append(f"{len(_files)} files of `{_local}` -> `s3://{_dest}`")
    mo.md("\n".join(f"- {d}" for d in _done) or "Nothing to upload yet.")
    return


if __name__ == "__main__":
    app.run()
