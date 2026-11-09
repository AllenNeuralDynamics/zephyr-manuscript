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
    import shutil
    import subprocess
    import sys
    import tomllib
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch

    from utils import picks, runs, test_sets
    from zephyr import features
    from zephyr.benchmarks import analysis, cli, probe
    from zephyr.config import ClipList, ResolvedClip, load
    from zephyr.quality_control.report import _spread_out

    DATA = runs.DATA
    """The local copy of the paper's data (``scripts/sync_data.ps1``, ``ZEPHYR_DATA``);
    nothing else is read, and nothing is downloaded."""
    MOUNTS = runs.MOUNTS
    RUNS = runs.LOCAL_RUNS
    """Every run, as on S3 (``<bucket>/zephyr-manuscript-runs``)."""
    OUT = Path(os.environ.get("ZEPHYR_ANALYSES", test_sets.CACHE / "analyses"))
    """Each analysis's folder (``ZEPHYR_ANALYSES`` names another, as the figures read
    it), uploaded by ``scripts/upload_analyses.ps1`` into the runs' ``analysis/<name>/``."""
    # Everything runs.py computes from the data is written here, and read back from here.
    runs.COMPUTE = True
    runs.ANALYSES = OUT
    STAGE = DATA / "stage"
    """Cache entries gathered into one folder for ``zephyr evaluate`` (hard links into
    the copy's caches, on the same drive)."""
    CHECKPOINT = "all-cameras/cv_0/seed-17"
    """The network of the noise null and of dynamic foraging before refining."""
    MANUSCRIPT = runs.MANUSCRIPT
    REPO = Path(__file__).resolve().parents[1]
    NOTEBOOK = Path(__file__).resolve()
    ARGS = mo.cli_args()
    SCRIPT = mo.app_meta().mode == "script"
    """Run as a script, every analysis runs; in the editor, each waits for its button."""
    FORCE = bool(ARGS.get("force"))
    """``-- --force``: make again an analysis this commit already made."""
    ONLY = {s for s in str(ARGS.get("only") or "").split(",") if s}
    """``-- --only tables,errors``: only these analyses (as a script)."""
    SKIP = {s for s in str(ARGS.get("skip") or "").split(",") if s}
    """``-- --skip timing``: all but these analyses (as a script)."""

    def sha256(path: Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def require(path: Path) -> Path:
        """*path*, which must be in the copy: a missing one names the sync to run."""
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is not in the local copy {DATA}: run scripts/sync_data.ps1"
                " (or set ZEPHYR_DATA to the copy)"
            )
        return path

    def run_file(relative: str) -> Path:
        """A file of the runs, in the copy."""
        return require(RUNS / relative)

    def experiment(name: str) -> dict:
        return tomllib.loads((MANUSCRIPT / "experiments" / name).read_text())

    def test_selection(name: str, fold: str, group: str) -> dict:
        """The ``clips`` and ``groups`` of test group *group* of *fold* in *name*."""
        (found,) = [f for f in experiment(name)["fold"] if f["name"] == fold]
        return found["test"][group]

    def stage(clips: list[ResolvedClip], recipe, folder: Path) -> Path:
        """The cache entries of *clips* (one folder per camera in the copy) gathered
        into *folder*, the one cache ``zephyr evaluate`` reads: hard links, else
        copies."""
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        for c in clips:
            source = runs.cache_folder(c, recipe)
            for path in features.paths(source, features.prefix(c, recipe)).values():
                if path.is_file():
                    try:
                        os.link(path, folder / path.name)
                    except OSError:
                        shutil.copy2(path, folder / path.name)
        return folder

    def manuscript_list(name: str, groups: list[str], path: Path) -> Path:
        """A clip list of the manuscript's configs (absolute paths under a mount) cut to
        *groups*, its paths moved into the copy; written to *path*."""
        source = MANUSCRIPT / "clips" / name
        text = source.read_text()
        clips = [c for c in tomllib.loads(text)["clip"] if c["group"] in groups]
        lines = [
            f"# {source.name} cut to {len(groups)} sessions, paths in the copy",
            "",
        ]
        lines += [text[: text.index("[[clip]]")].rstrip(), ""]
        for c in clips:
            lines.append("[[clip]]")
            for key in ("video", "timestamps", "thermistor"):
                lines.append(f"{key} = {json.dumps(runs.local_path(c[key]))}")
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

    def require_clean() -> str:
        """The commit an analysis runs from: refuses a working tree that differs from
        it (tracked or untracked files), so every output names the code that made it."""
        changed = git("status", "--porcelain")
        if changed:
            raise SystemExit(
                "Commit first: the analyses run only from a clean working tree.\n"
                + changed
            )
        return git("rev-parse", "HEAD")

    def made_here(name: str) -> bool:
        """Whether analysis *name* was made from this commit."""
        path = OUT / name / "provenance.json"
        if not path.is_file():
            return False
        rec = json.loads(path.read_text())
        return rec.get("commit") == git("rev-parse", "HEAD") and not rec.get("dirty")

    def should_run(name: str, button) -> bool:
        """Run in the editor when its button is pressed; as a script, unless this
        commit already made it (``--force`` makes it again) or ``--only``/``--skip``
        leave it out."""
        if button.value:
            return True
        if not SCRIPT or (ONLY and name not in ONLY) or name in SKIP:
            return False
        return FORCE or not made_here(name)

    def begin(name: str) -> tuple[Path, float]:
        """Start analysis *name* from a clean commit, in an empty folder."""
        require_clean()
        folder = OUT / name
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        return folder, dt.datetime.now(dt.UTC).timestamp()

    def status(name: str):
        """What the editor shows of analysis *name*: whether, and from which commit,
        it was made."""
        path = OUT / name / "provenance.json"
        if not path.is_file():
            return mo.md(f"`{name}`: not made yet.")
        rec = json.loads(path.read_text())
        return mo.md(
            f"`{name}`: made from `{rec['commit'][:10]}` on {rec['finished_utc']}"
            f" in {rec['wall_time_s']} s, {len(rec['outputs'])} files."
        )

    def outputs(folder: Path) -> dict[str, str]:
        """The SHA-256 of every file of *folder* (but its provenance)."""
        return {
            p.relative_to(folder).as_posix(): sha256(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != "provenance.json"
        }

    def provenance(folder: Path | None, started: float, **extra) -> dict:
        """``provenance.json`` of an analysis folder: the commit (and whether it is on
        the remote), the notebook and lock file, the pinned packages, the data copy
        (its ``sync.json``), the machine, the command, the time it took and the
        SHA-256 of every output."""
        now = dt.datetime.now(dt.UTC).timestamp()
        packages = {}
        for n in ("zephyr", "zephyr-benchmarks", "torch", "marimo", "numpy"):
            try:
                packages[n] = {"version": importlib.metadata.version(n)}
            except importlib.metadata.PackageNotFoundError:
                packages[n] = None
        pins = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["uv"]
        commit = git("rev-parse", "HEAD")
        sync = DATA / "sync.json"
        rec = {
            "notebook": NOTEBOOK.relative_to(REPO).as_posix(),
            "notebook_sha256": sha256(NOTEBOOK),
            "repository": git("remote", "get-url", "origin"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": commit,
            "dirty": bool(git("status", "--porcelain")),
            "commit_on_remote": bool(git("branch", "-r", "--contains", commit)),
            "uv_lock_sha256": sha256(REPO / "uv.lock"),
            "zephyr_source": pins["sources"]["zephyr"],
            "manifest": runs.MANIFEST.relative_to(REPO).as_posix(),
            "data": {
                "root": str(DATA),
                "sync": json.loads(sync.read_text(encoding="utf-8-sig"))
                if sync.is_file()
                else None,
            },
            "command": sys.argv,
            "mode": mo.app_meta().mode,
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
                "os": platform.platform(),
            },
            "packages": packages,
            **extra,
        }
        if folder is not None:
            rec["outputs"] = outputs(folder)
            (folder / "provenance.json").write_text(json.dumps(rec, indent=2))
        return rec

    def progress(items, title: str):
        """A progress bar in the editor; the items as they are in a script."""
        return items if SCRIPT else mo.status.progress_bar(items, title=title)


@app.cell
def _():
    mo.md(f"""
    # The paper's analyses

    Everything the paper's figures need that is computed rather than read from a run:
    whatever needs the clips' truth, frames or videos, the GPU, or is worth caching.
    `manuscript_figures.py` then reads only the runs and these outputs. Each analysis is
    one folder of `{OUT}`, with its `provenance.json`; upload them into the runs'
    `analysis/` with `scripts/upload_analyses.ps1`. What is computed, and how, is
    `utils/runs.py` (the same functions the figures read the outputs with); the
    examples picked by hand are `utils/picks.py`.

    ## Data: download it first

    The notebook reads one local copy of the paper's data and runs, never S3: the
    folder `ZEPHYR_DATA` (now `{DATA}`). Make it with the AWS CLI and credentials that
    read the bucket (`aws sso login`):

    ```
    .\\scripts\\sync_data.ps1 -DryRun    # list what it would copy
    .\\scripts\\sync_data.ps1            # to D:\\zephyr-data (-Root elsewhere), ~2.2 TB
    ```

    It copies, in one folder (delete it when done), laid out as the Code Ocean capsule
    mounts them: the corpus (`zephyr-dataset/`), the dynamic-foraging sessions
    (`zephyr-df-dataset/`), the feature caches (`zephyr-cache/<recipe>/`), every run
    (`zephyr-manuscript-runs/`), and, of the sessions without a thermistor, their face
    camera and processed assets (`aind-open-data/`). Run it again to bring the copy up to
    date; its `sync.json` goes into every `provenance.json` made here. A file the copy
    lacks is an error naming the script.

    ## Running

    From a clean, committed working tree (an analysis refuses anything else), as a
    script:

    ```
    uv run --locked --all-packages --extra gpu python notebooks/analyses.py
    uv run --locked --all-packages --extra gpu python notebooks/analyses.py -- --skip timing
    ```

    Every analysis this commit has not made yet runs (`-- --force`: all of them again;
    `-- --only tables,errors` or `-- --skip timing` to choose). Time the methods
    (`timing`) on a quiet machine. In the editor (`marimo edit notebooks/analyses.py`)
    each waits for its button.
    """)
    return


@app.cell
def _():
    model = run_file(f"{CHECKPOINT}/best.pt")
    mo.md(f"Checkpoint `{model}`, SHA-256 `{sha256(model)}`.")
    return (model,)


@app.cell
def _():
    mo.md(r"""
    ## Checks: the runs' own scores

    Every score the figures show is a run's own `evaluation.json`, scored on Code Ocean
    by the pinned zephyr. This scores a sample of clips again here, from each run's saved
    trace and the clip's truth, and stops if any score differs.
    """)
    return


@app.cell
def _():
    run_checks = mo.ui.run_button(label="Check the runs' scores")
    run_checks
    return (run_checks,)


@app.cell
def _(run_checks):
    CHECKS = "checks"
    if should_run(CHECKS, run_checks):
        _folder, _started = begin(CHECKS)
        _rows = []
        _fold = runs.CV.folds[0]
        for _m in runs.CV.method:
            _name = runs.cv_name(_m.label, _fold)
            if runs.evaluation(_name) is None:
                continue
            _clips = runs.own_scores(_name).to_dict("records")
            for _c in _clips[:: max(1, len(_clips) // 10)][:10]:  # 10 spread clips
                _stem = _c["clip_id"].split("/")[-1]
                _here = analysis.score_prediction(
                    runs.truth_entry(_stem), runs.prediction(_name, _stem), _m.head
                )
                for _k in runs.SCORES:
                    if _k in _c and _k in _here:
                        _rows.append(
                            {
                                "method": _m.label,
                                "fold": _fold,
                                "stem": _stem,
                                "metric": _k,
                                "run": float(_c[_k]),
                                "here": float(_here[_k]),
                            }
                        )
        _checks = pd.DataFrame(_rows)
        _checks["differs"] = ~np.isclose(_checks["run"], _checks["here"], atol=1e-6)
        _checks.to_parquet(_folder / "scores.parquet")
        provenance(_folder, _started, fold=_fold, clips_per_method=10)
        if _checks["differs"].any():
            raise SystemExit(
                "The runs' scores differ from scoring here:\n"
                + _checks[_checks["differs"]].to_string()
            )
    checks_done = True
    status(CHECKS)
    return (checks_done,)


@app.cell
def _():
    mo.md(r"""
    ## Noise null (Figs. 4-5, S9)

    The all-camera network of cv_0 predicts 24 face-camera clips of cv_0's held-out
    animals (test A) again with Gaussian noise of 4, 8, 16, 32 and 64 gray levels added
    to every frame (`zephyr-benchmarks probe`, seed 0, each level's trace saved).
    """)
    return


@app.cell
def _():
    run_noise = mo.ui.run_button(label="Run the noise null")
    run_noise
    return (run_noise,)


@app.cell
def _(checks_done, model, run_noise):
    NOISE = "noise-null"
    if should_run(NOISE, run_noise) and checks_done:
        _folder, _started = begin(NOISE)
        _groups = test_selection("face.toml", "cv_0", "test_a")["groups"]
        _list = require(MOUNTS["/data/zephyr-dataset"] / "clips.toml")
        _clips, _recipe = probe._selections([_list], _groups)
        _cache = runs.cache_folder(_spread_out(_clips, 24)[0], _recipe)
        _argv = [
            "--checkpoint",
            str(model),
            "--clips",
            str(_list),
            "--groups",
            *_groups,
        ]
        _argv += ["--cache", str(_cache), "--out", str(_folder / "test_a")]
        _argv += ["--max-clips", "24", "--save-traces", "--seed", "0", "--amp", "bf16"]
        probe.main(_argv)
        provenance(
            _folder,
            _started,
            argv=["zephyr-benchmarks", "probe", *_argv],
            checkpoint={"run": CHECKPOINT, "sha256": sha256(model)},
            experiment="face.toml",
            fold="cv_0",
        )
    noise_done = True
    _dir = OUT / NOISE / "test_a"
    _noise = probe.load_probes(_dir) if any(_dir.glob("*.probe.json")) else None
    mo.vstack([status(NOISE), _noise])
    return (noise_done,)


@app.cell
def _():
    mo.md(r"""
    ## Dynamic foraging before refining (Fig. 3g-i, Fig. S8)

    The all-camera network of cv_0, as it is, scored on each left-out animal of
    `dynamic-foraging.toml`'s three folds (`zephyr-benchmarks evaluate`, group
    `df_test_a`, as the refined runs score it), one run folder per fold:
    `df-unrefined-scores/<fold>/seed-17/evaluation.json`.
    """)
    return


@app.cell
def _():
    run_df = mo.ui.run_button(label="Score the dynamic-foraging animals")
    run_df
    return (run_df,)


@app.cell
def _(checks_done, model, run_df):
    DF = "df-unrefined-scores"
    df_folds = [f["name"] for f in experiment("dynamic-foraging.toml")["fold"]]
    if should_run(DF, run_df) and checks_done:
        _folder, _started = begin(DF)
        _argvs = {}
        for _fold in df_folds:
            _run = _folder / _fold / "seed-17"
            _groups = test_selection("dynamic-foraging.toml", _fold, "df_test_a")[
                "groups"
            ]
            _list = manuscript_list("clips-df.toml", _groups, _run / "df_test_a.toml")
            _clips = load(ClipList, _list, check_files=False)
            _cache = stage(_clips.resolve(), _clips.preprocess, STAGE / DF / _fold)
            _argv = ["evaluate", "--checkpoint", str(model), "--clips", str(_list)]
            _argv += ["--name", "df_test_a", "--cache", str(_cache), "--amp", "bf16"]
            _argv += ["--out", str(_run / "evaluation.json")]
            cli.run(_argv)
            _argvs[_fold] = ["zephyr-benchmarks", *_argv]
        provenance(
            _folder,
            _started,
            argv=_argvs,
            checkpoint={"run": CHECKPOINT, "sha256": sha256(model)},
            experiment="dynamic-foraging.toml",
        )
    _rows = []
    for _fold in df_folds:
        _path = OUT / DF / _fold / "seed-17" / "evaluation.json"
        if _path.is_file():
            _summary = json.loads(_path.read_text())["groups"]["df_test_a"]["summary"]
            _rows.append(
                {"fold": _fold}
                | {k: v for k, v in _summary.items() if isinstance(v, float)}
            )
    mo.vstack([status(DF), pd.DataFrame(_rows) if _rows else None])
    return


@app.cell
def _():
    mo.md(r"""
    ## The corpus's tables (Fig. 2, Fig. S4, Table S1)

    The corpus's small `raw/` tables the figures count (sessions, animals, rigs, the
    selection flow), copied as they are; and the three cameras of Fig. S4's sessions
    (`picks.VIEW_SESSIONS`) at half size with each network's crop box.
    """)
    return


@app.cell
def _():
    run_corpus = mo.ui.run_button(label="Copy the corpus's tables")
    run_corpus
    return (run_corpus,)


@app.cell
def _(checks_done, run_corpus):
    CORPUS = "corpus"
    if should_run(CORPUS, run_corpus) and checks_done:
        _folder, _started = begin(CORPUS)
        (_folder / "raw").mkdir()
        for _name in runs.CORPUS_FILES:
            shutil.copy2(
                require(runs.dataset_dir() / "raw" / _name), _folder / "raw" / _name
            )
        for _session in picks.VIEW_SESSIONS:
            runs.camera_views(_session)
        provenance(_folder, _started, views=list(picks.VIEW_SESSIONS))
    status(CORPUS)
    return


@app.cell
def _():
    mo.md(r"""
    ## The clips the figures show (Figs. 1, 3a, 4, 6, S6, S7)

    For every new-animal clip of Zephyr on cv_0 (the figures' examples are picked among
    them) and every dynamic-foraging clip: its truth (target, inhales, exhales, excluded
    spans), output grid, crop box, three crops at its middle inhale, and its session's
    thumbnail from its camera (`runs.keep_truth`).
    """)
    return


@app.cell
def _():
    run_truth = mo.ui.run_button(label="Keep the clips' truth")
    run_truth
    return (run_truth,)


@app.cell
def _(checks_done, run_truth):
    TRUTH = "truth"
    if should_run(TRUTH, run_truth) and checks_done:
        _folder, _started = begin(TRUTH)
        _stems = sorted(
            s
            for s, v in runs.test_clips(runs.SINGLE).items()
            if v == runs.SETS["test_a"] and runs.is_local(s)
        )
        _df = sorted(runs.df_entries())
        for _stem in progress([*_stems, *_df], "Clips"):
            runs.keep_truth(_stem)
        provenance(
            _folder, _started, run=runs.SINGLE, clips=len(_stems), df_clips=len(_df)
        )
    status(TRUTH)
    return


@app.cell
def _():
    mo.md(r"""
    ## Clip tables (Figs. 3c-e, 4, S2, S3, S5, Table 1, Tables S4-S5)

    `analysis.clip_tables` of every cross-validation run (every method and variant on
    every fold) over its test clips, from its saved traces and the clips' truth: each
    breath found or missed, false inhales, onset lags, 10-s rate windows.
    """)
    return


@app.cell
def _():
    run_tables = mo.ui.run_button(label="Make the clip tables")
    run_tables
    return (run_tables,)


@app.cell
def _(checks_done, run_tables):
    TABLES = "tables"
    if should_run(TABLES, run_tables) and checks_done:
        _folder, _started = begin(TABLES)
        _names = runs.available(runs.CV_RUNS)
        for _name in _names:
            runs.tables(_name, progress=lambda x, n=_name: progress(x, n))
        provenance(_folder, _started, runs=_names)
    tables_done = True
    status(TABLES)
    return (tables_done,)


@app.cell
def _():
    mo.md(r"""
    ## Errors (Fig. 4d, f; Fig. 6a)

    `analysis.errors` of Zephyr on every new-animal clip of every fold: each missed and
    false inhale, what was near it, and the frames' motion around it.
    """)
    return


@app.cell
def _():
    run_errors = mo.ui.run_button(label="Find the errors")
    run_errors
    return (run_errors,)


@app.cell
def _(checks_done, run_errors):
    ERRORS = "errors"
    if should_run(ERRORS, run_errors) and checks_done:
        _folder, _started = begin(ERRORS)
        _names = [runs.cv_name(runs.REFERENCE, f) for f in runs.CV.folds]
        for _name in progress(_names, "Folds"):
            runs.set_errors(_name, runs.SETS["test_a"])
        provenance(_folder, _started, runs=_names)
    status(ERRORS)
    return


@app.cell
def _():
    mo.md(r"""
    ## Prominence (Fig. S5e)

    The inhalations the corpus's detector adds when its prominence fraction is lowered
    from 0.2 to 0.1 and 0.05, on the raw thermistor of every new-animal clip.
    """)
    return


@app.cell
def _():
    run_prominence = mo.ui.run_button(label="Detect at lower prominence")
    run_prominence
    return (run_prominence,)


@app.cell
def _(checks_done, run_prominence):
    PROMINENCE = "prominence"
    if should_run(PROMINENCE, run_prominence) and checks_done:
        _folder, _started = begin(PROMINENCE)
        runs.label_added()
        provenance(_folder, _started, settings=list(runs.PROMINENCE))
    status(PROMINENCE)
    return


@app.cell
def _():
    mo.md(r"""
    ## Breathing rates (Fig. 2b)

    Every thermistor breath's rate per animal, over every corpus clip of a session in a
    set, from the clips' events.
    """)
    return


@app.cell
def _():
    run_rates = mo.ui.run_button(label="Collect the breathing rates")
    run_rates
    return (run_rates,)


@app.cell
def _(checks_done, run_rates):
    RATES = "breath-rates"
    if should_run(RATES, run_rates) and checks_done:
        _folder, _started = begin(RATES)
        runs.animal_breath_rates()
        provenance(_folder, _started)
    status(RATES)
    return


@app.cell
def _():
    mo.md(r"""
    ## Quality control (Figs. 4-5, S8-S10)

    Every 30-s window of the manifest's `[quality]` sets, as their reports scored them,
    joined with each window's real scores where the thermistor is known (the reports
    never read it).
    """)
    return


@app.cell
def _():
    run_quality = mo.ui.run_button(label="Join the quality-control windows")
    run_quality
    return (run_quality,)


@app.cell
def _(checks_done, noise_done, run_quality):
    QUALITY = "quality"
    if should_run(QUALITY, run_quality) and checks_done and noise_done:
        _folder, _started = begin(QUALITY)
        runs.quality_windows()
        provenance(_folder, _started, reference=runs.RESULTS.quality.reference)
    quality_done = True
    status(QUALITY)
    return (quality_done,)


@app.cell
def _():
    mo.md(r"""
    ## One session in the task (Fig. 1b, c)

    The session of `picks.TASK_ASSET`: its behaviour from its processed asset, its two
    clips predicted by the fold that holds the animal out, and its whole thermistor.
    """)
    return


@app.cell
def _():
    run_task = mo.ui.run_button(label="Extract the task session")
    run_task
    return (run_task,)


@app.cell
def _(checks_done, run_task):
    TASK = "task"
    if should_run(TASK, run_task) and checks_done:
        _folder, _started = begin(TASK)
        runs.task()
        provenance(_folder, _started, asset=picks.TASK_ASSET, fold=picks.TASK_FOLD)
    status(TASK)
    return


@app.cell
def _():
    mo.md(r"""
    ## Sessions without a thermistor (Fig. 5)

    Each application session's face prediction on the Harp clock, its behaviour from
    its processed asset, and frames of the window Fig. 5b shows (the lowest window, at
    `picks.APPLICATION_SHOTS_S`).
    """)
    return


@app.cell
def _():
    run_application = mo.ui.run_button(label="Extract the sessions")
    run_application
    return (run_application,)


@app.cell
def _(checks_done, quality_done, run_application):
    APPLICATION = "application"
    if should_run(APPLICATION, run_application) and checks_done and quality_done:
        _folder, _started = begin(APPLICATION)
        for _s in progress(runs.RESULTS.application.session, "Sessions"):
            runs.application(_s.session)
        _w = runs.application_window()
        runs.application_frames(
            _w["session"], [_w["start"] + _t for _t in picks.APPLICATION_SHOTS_S]
        )
        provenance(
            _folder,
            _started,
            window={"stem": _w["stem"], "start": float(_w["start"])},
            shots_s=list(picks.APPLICATION_SHOTS_S),
        )
    status(APPLICATION)
    return


@app.cell
def _():
    mo.md(r"""
    ## Saliency (Fig. S6)

    Occlusion and gradient maps of the face network of each fold on one clip per
    new-animal unit (nearest its median correlation), and of the side and front
    networks of cv_0 on the same spans (GPU).
    """)
    return


@app.cell
def _():
    run_saliency = mo.ui.run_button(label="Map the saliency")
    run_saliency
    return (run_saliency,)


@app.cell
def _(checks_done, run_saliency):
    SALIENCY = "saliency"
    if should_run(SALIENCY, run_saliency) and checks_done:
        _folder, _started = begin(SALIENCY)
        runs.saliency()
        provenance(_folder, _started)
    status(SALIENCY)
    return


@app.cell
def _():
    mo.md(r"""
    ## Inference time (Fig. 3b, Table 1, Table S2)

    Every method's inference time on the same 8 test clips of cv_0 after one warm-up
    clip: the networks on this machine's GPU, the pixel methods and Facemap on its CPU.
    Run it on a quiet machine.
    """)
    return


@app.cell
def _():
    run_timing = mo.ui.run_button(label="Time the methods")
    run_timing
    return (run_timing,)


@app.cell
def _(checks_done, run_timing):
    TIMING = "timing"
    if should_run(TIMING, run_timing) and checks_done:
        _folder, _started = begin(TIMING)
        _timing = runs.wall_time(list(runs.METHODS))
        provenance(_folder, _started, methods=list(runs.METHODS))
    status(TIMING)
    return


if __name__ == "__main__":
    app.run()
