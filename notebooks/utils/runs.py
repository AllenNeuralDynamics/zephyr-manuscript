"""The paper's runs, and what is computed from them and the data.

Which runs are compared, and under which labels, is a results manifest of
``zephyr-benchmarks`` (``configs/results/``, :data:`MANIFEST`). Everything is read from
one local copy of the paper's data and runs (:data:`DATA`, ``scripts/sync_data.ps1``);
nothing is downloaded.

Two notebooks use this module. ``analyses.py`` sets :data:`COMPUTE`: whatever needs the
data (the clips' truth, frames and videos, the GPU) is computed, and written into
:data:`ANALYSES` (one folder per analysis, uploaded into the runs' ``analysis/``).
``manuscript_figures.py`` leaves it off: the same functions read what the analyses
wrote, and a missing output is an error, never computed there. Every score is a run's
own ``evaluation.json``, as Code Ocean scored it.
"""

import functools
import json
import os
import re
import time
import tomllib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zephyr import features
from zephyr.benchmarks import CONFIGS, analysis, baselines, common, timing
from zephyr.benchmarks.config import ReportedRun, ResultsManifest
from zephyr.benchmarks.runner import load_checkpoint
from zephyr.config import PATH_MAP_ENV, ClipList, load
from zephyr.evaluate import (
    HEAD_THRESHOLD,
    TRACES,
    aggregate_recordings,
    head_event_indices,
)
from zephyr.infer import predict_clip

from . import picks

REPO = Path(__file__).resolve().parents[2]
MANIFEST_ENV = "ZEPHYR_RESULTS"
MANIFEST = Path(
    os.environ.get(
        MANIFEST_ENV,
        REPO / "packages/zephyr-benchmarks/configs/results/manuscript.toml",
    )
)
"""The results manifest the notebook reports, unless ``ZEPHYR_RESULTS`` names another."""
RESULTS: ResultsManifest = load(ResultsManifest, MANIFEST)
S3_RUNS = RESULTS.root.rstrip("/")
"""Where the manifest's runs were uploaded."""
DATA = Path(os.environ.get("ZEPHYR_DATA", "D:/zephyr-data"))
"""The local copy of the paper's data and runs (``scripts/sync_data.ps1``): the only
place anything is read from."""
LOCAL_RUNS = DATA / "zephyr-manuscript-runs"
"""The runs, in the copy (as on S3, the manifest's ``root``)."""
ANALYSES = Path(os.environ.get("ZEPHYR_ANALYSES", LOCAL_RUNS / "analysis"))
"""Where the analyses are read from (and, with :data:`COMPUTE`, written to): the runs'
``analysis/`` once uploaded and synced; ``ZEPHYR_ANALYSES`` names another folder (e.g.
``notebooks/cache/analyses`` before uploading). A run folder ``analysis/<name>`` of the
manifest is read from here too."""
COMPUTE = False
"""Whether an analysis output that is missing is computed from the data (and written),
as ``analyses.py`` does, or is an error, as in the figures."""
MANUSCRIPT = CONFIGS / "manuscript"
"""The capsule's configs the runs were made from (experiments and clip lists)."""
CV = RESULTS.cross_validation


def cv_name(label: str, fold: str) -> str:
    """The name of a cross-validation method's run on one fold."""
    return f"{label} | {fold}"


METHODS: dict[str, ReportedRun] = {r.label: r for r in RESULTS.method} or (
    {r.label: r.model_copy(update={"fold": CV.folds[0]}) for r in CV.method}
    if CV is not None
    else {}
)
"""Every compared method, in the order tables list them: on the manifest's one split,
or, without one, each cross-validation method on its first fold (what one run of each
shows: its inference time, its settings)."""
VARIANTS: dict[str, ReportedRun] = {r.label: r for r in RESULTS.variant}
"""The first method trained differently, compared with it alone."""
LEARNING_CURVE: dict[str, ReportedRun] = {r.label: r for r in RESULTS.learning_curve}
"""The first method trained on parts of its training set."""
ALL = METHODS | VARIANTS | LEARNING_CURVE
CV_VARIANTS: dict[str, ReportedRun] = (
    {r.label: r for r in CV.variant} if CV is not None else {}
)
"""The first method trained differently on each cross-validation fold."""
CV_RUNS: dict[str, ReportedRun] = (
    {
        cv_name(m.label, f): m.model_copy(update={"fold": f})
        for f in CV.folds
        for m in (*CV.method, *CV_VARIANTS.values())
    }
    if CV is not None
    else {}
)
"""Each cross-validation method and variant on each fold (:func:`cv_name`)."""
RUNS = ALL | CV_RUNS
"""Every run by name: :data:`ALL` and :data:`CV_RUNS`."""
REFERENCE = next(iter(METHODS))
"""The method every other is compared with (Zephyr)."""
SINGLE = cv_name(REFERENCE, CV.folds[0])
"""The trained network the single-network analyses use (example clips, error cases):
the reference on the first cross-validation fold."""


def pooled(label: str) -> list[str]:
    """The runs *label* names: itself, or each run labelled ``<label> (<part>)`` (one
    per left-out animal, say), to be pooled."""
    return [
        n
        for n in RUNS
        if n == label or (n.startswith(f"{label} (") and n.endswith(")"))
    ]


def url(m: ReportedRun) -> str:
    return f"{S3_RUNS}/{m.run}/"


PIXEL_BIN = 2
"""The pixel methods' spatial binning, as the benchmark ran them."""


MOUNTS = {
    "/data/zephyr-dataset": DATA / "zephyr-dataset",
    "/data/zephyr-df-dataset": DATA / "zephyr-df-dataset",
}
"""Where the capsule mounts each dataset (its clip lists and cache keys name it), and
its folder in the copy, named as the mount so the cache keys are the capsule's."""
OPEN_DATA = DATA / "aind-open-data"
"""The sessions without a thermistor and the processed assets, in the copy (the capsule
mounts each at ``/data/<name>``)."""
os.environ[PATH_MAP_ENV] = ";".join(
    f"{local.as_posix()}={mount}" for mount, local in MOUNTS.items()
)


def local_path(path: str) -> str:
    """A path of the capsule's clip lists (under a ``/data`` mount), in the copy."""
    for mount, local in MOUNTS.items():
        if path.startswith(mount + "/"):
            return (local / path[len(mount) + 1 :]).as_posix()
    if path.startswith("/data/"):
        return (OPEN_DATA / path.removeprefix("/data/")).as_posix()
    return path


def manuscript_clips(name: str, groups: Iterable[str] | None = None) -> ClipList:
    """A clip list of the capsule's configs (``clips/<name>``), cut to *groups*, its
    paths in the copy."""
    raw = tomllib.loads((MANUSCRIPT / "clips" / name).read_text())
    wanted = None if groups is None else set(groups)
    clips_ = [
        c
        | {
            k: local_path(c[k])
            for k in ("video", "timestamps", "thermistor")
            if isinstance(c.get(k), str)
        }
        for c in raw["clip"]
        if wanted is None or c["group"] in wanted
    ]
    return ClipList.model_validate({**raw, "clip": clips_})


def dataset_dir() -> Path:
    """The corpus, in the copy."""
    return MOUNTS["/data/zephyr-dataset"]


def cache_dir() -> Path:
    """The feature caches the runs were trained and scored on, in the copy."""
    return DATA / "zephyr-cache"


RECIPE = "features-360x270-box96x96-60hz"
"""The face camera's feature cache: the corpus's clips at their own recipe."""


def cache_folder(clip, recipe) -> Path:
    """The copy's folder of *clip*'s cache entry, as the capsule names it: the
    recipe's, with the camera when it is not the face's."""
    params = recipe.for_clip(clip)
    w, h = params.target_size
    bw, bh = clip.box[2:]
    name = f"features-{w}x{h}-box{bw}x{bh}-{params.select_fs:g}hz"
    camera = re.split(r"_span\d+_", clip.source.stem)[-1]
    return cache_dir() / (name if camera == "FaceCamera" else f"{name}-{camera}")


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is not in the local copy {DATA}: run scripts/sync_data.ps1"
        )
    return path


# --- the analyses' outputs -------------------------------------------------------------


def made(name: str, path: str) -> Path:
    """File *path* of analysis *name*. Missing, it is an error unless :data:`COMPUTE`
    (then the caller makes it)."""
    out = ANALYSES / name / path
    if not COMPUTE and not out.exists():
        raise FileNotFoundError(
            f"{out} is missing: notebooks/analyses.py makes it (analysis {name!r});"
            " upload it with scripts/upload_analyses.ps1 and sync the runs, or set"
            " ZEPHYR_ANALYSES to the folder it was written to"
        )
    return out


def _memo(name: str, path: str, make: Callable, read: Callable, write: Callable):
    """What analysis *name* wrote at *path*, or, with :data:`COMPUTE`, *make* it and
    write it there."""
    out = made(name, path)
    if out.exists():
        return read(out)
    found = make()
    out.parent.mkdir(parents=True, exist_ok=True)
    write(found, out)
    return found


def _frame_memo(name: str, path: str, make: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    return _memo(name, path, make, pd.read_parquet, lambda df, p: df.to_parquet(p))


def _arrays_memo(name: str, path: str, make: Callable[[], dict]) -> dict:
    def read(p):
        with np.load(p, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}

    return _memo(name, path, make, read, lambda d, p: np.savez(p, **d))


# --- the runs ------------------------------------------------------------------------


def _runs_path(key: str) -> Path:
    """``<run>/<path>`` in the runs' copy; a run of the analyses
    (``analysis/<name>/...``) in :data:`ANALYSES`."""
    if key.startswith("analysis/"):
        return ANALYSES / key.removeprefix("analysis/")
    return LOCAL_RUNS / key


def _fetch(key: str) -> Path | None:
    """``<run>/<path>`` in the runs' copy; ``None`` if absent (not finished)."""
    path = _runs_path(key)
    return path if path.exists() else None


def folder(m: ReportedRun) -> str:
    """A run's folder under the root: ``<run>/<fold>``."""
    return f"{m.run}/{m.fold or RESULTS.fold}"


@functools.cache
def seeds(folder: str) -> tuple[str, ...]:
    """The seed folders a network run has finished (a ``best.pt`` and its scores), in
    its *folder* (``<run>/<fold>``, :func:`folder`)."""
    return tuple(
        sorted(p.parent.name for p in _runs_path(folder).glob("*/evaluation.json"))
    )


def evaluation(name: str, seed: str | None = None) -> dict | None:
    """The run's own ``evaluation.json`` (scored on Code Ocean), or ``None`` if the
    run has not finished."""
    return _evaluation(RUNS[name], seed)


@functools.cache
def _evaluation(m: ReportedRun, seed: str | None = None) -> dict | None:
    if m.kind == "network":
        found = seeds(folder(m))
        if not found:
            return None
        key = f"{folder(m)}/{seed or found[0]}/evaluation.json"
    else:
        key = f"{folder(m)}/{m.kind}/{m.variant}/evaluation.json"
    path = _fetch(key)
    return json.loads(path.read_text()) if path else None


def run_file(name: str, path: str) -> Path:
    """File *path* of network run *name*'s first seed (``config.json``, ``best.pt``)."""
    m = RUNS[name]
    return _require(_runs_path(f"{folder(m)}/{seeds(folder(m))[0]}/{path}"))


def training(name: str) -> pd.DataFrame:
    """The sessions a network run trained on (its first seed's ``train_videos.json``),
    with their ``subject`` and ``rig`` from the split."""
    stems = [
        Path(v).stem
        for v in json.loads(run_file(name, "train_videos.json").read_text())
    ]
    sessions = sorted({stem.rsplit("_span", 1)[0] for stem in stems})
    folds = pd.read_parquet(dataset_raw("folds.parquet")).set_index("session_id")
    return pd.DataFrame(
        {
            "session": sessions,
            "subject": folds.loc[sessions, "subject_id"].astype(str).to_numpy(),
            "rig": folds.loc[sessions, "rig"].to_numpy(),
        }
    )


def available(names) -> list[str]:
    """The methods of *names* whose run has finished."""
    return [n for n in names if evaluation(n) is not None]


# --- the corpus's own tables -----------------------------------------------------------

CORPUS_FILES = (
    "cameras.parquet",
    "clips.parquet",
    "folds.parquet",
    "inventory.parquet",
    "qc.parquet",
    "selection.json",
    "sync.parquet",
)
"""The corpus's small ``raw/`` tables the figures read (copied by the ``corpus``
analysis)."""


def corpus_root() -> Path:
    """The folder whose ``raw/`` holds the corpus's tables (:data:`CORPUS_FILES`): the
    copy's corpus when computing, else the ``corpus`` analysis."""
    if COMPUTE:
        return dataset_dir()
    made("corpus", "raw/folds.parquet")
    return ANALYSES / "corpus"


def dataset_raw(name: str) -> Path:
    """One of the corpus's tables (:data:`CORPUS_FILES`)."""
    return _require(corpus_root() / "raw" / name)


# --- scores, the runs' own -------------------------------------------------------------


def _subjects() -> dict[str, str]:
    folds = pd.read_parquet(dataset_raw("folds.parquet"))
    return dict(zip(folds["session_id"], folds["subject_id"], strict=True))


SCORES = (
    "correlation",
    "inhale_f1",
    "exhale_f1",
    "kl_ibi",
    "head_inhale_f1",
)
"""The per-clip scores of a run's ``evaluation.json``."""


def own_scores(name: str) -> pd.DataFrame:
    """Per-clip scores of run *name* on its test clips (``test_a``-``c``), as its own
    ``evaluation.json`` gives them (``analyses.py`` checks a sample against scoring them
    again)."""
    keys = ("clip_id", "recording", "group", *SCORES)
    return pd.DataFrame(
        [
            {k: c[k] for k in keys if k in c}
            for c in evaluation(name)["clips"]
            if c["group"] in SETS
        ]
    )


def scores(names=ALL, level: str = "clips") -> pd.DataFrame:
    """Per-clip (``clips``) or per-recording (``per_recording``) scores of every
    finished method in *names* on its test clips (:func:`own_scores`); one ``inhale``
    column holds the F1 each method reports (the head's if it has one)."""
    return _scores([(name, name) for name in names], level)


def cv_scores(level: str = "per_recording", variants: bool = False) -> pd.DataFrame:
    """:func:`scores` of every cross-validation method (or, with *variants*, every
    variant) on every fold of the manifest's ``[cross_validation]``, with a ``fold``
    column; the ``set`` is the fold's own."""
    if CV is None:
        return pd.DataFrame()
    frames = [
        _scores([(m.label, cv_name(m.label, f))], level).assign(fold=f)
        for f in CV.folds
        for m in (CV.variant if variants else CV.method)
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _tag(name: str) -> str:
    """What run *name*'s analysis files are named by: its run, fold and variant (never
    its label, which a manifest may point at another run)."""
    m = RUNS[name]
    tag = f"{m.run}-{m.fold or RESULTS.fold}-{m.variant.replace('/', '-') or 'net'}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_")


def _level(clips: pd.DataFrame, level: str) -> pd.DataFrame:
    """Clip rows as they are (``clips``), or averaged per recording within each group
    (``per_recording``), as ``evaluate`` does."""
    if level == "clips":
        return clips
    rows = []
    for group, part in clips.groupby("group", sort=False):
        rows += [
            r | {"group": group} for r in aggregate_recordings(part.to_dict("records"))
        ]
    return pd.DataFrame(rows)


def _scores(named: list[tuple[str, str]], level: str) -> pd.DataFrame:
    """*named*: ``(label, run name)`` pairs; the label goes in the ``method`` column."""
    subjects = _subjects()
    frames = []
    for label, name in named:
        if evaluation(name) is None:
            continue
        m = RUNS[name]
        seed = seeds(folder(m))[0] if m.kind == "network" else "—"
        df = _level(own_scores(name), level).assign(
            method=label, seed=seed, head=m.head
        )
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["session"] = df["recording"].str.split("#").str[-1]
    df["subject"] = df["session"].map(subjects)
    df["set"] = df["group"].map(SETS)
    head_f1 = df.get("head_inhale_f1", np.nan)
    df["inhale"] = np.where(df["head"], head_f1, df["inhale_f1"]).astype(float)
    df = df.drop(columns="head")
    if level == "clips":
        df["stem"] = df["clip_id"].str.split("/").str[-1]
    return df


SETS = {"test_a": "A: new animals", "test_b": "B: new sessions", "test_c": "C: 359 Hz"}
METRICS = {
    "correlation": "Correlation",
    "inhale": "Inhale F1",
    "exhale_f1": "Exhale F1 (DSP)",
    "kl_ibi": "KL-IBI (lower is better)",
}


@functools.cache
def head_threshold(name: str) -> float:
    """The onset threshold run *name* was scored at: its checkpoint's own, chosen in
    training on its validation clips, as its ``evaluation.json`` records it (the
    scorer's for a run without a head)."""
    if not RUNS[name].head:
        return HEAD_THRESHOLD
    own = (evaluation(name) or {}).get("head_threshold") or {}
    if own.get("value") is None:
        raise ValueError(f"{name}: its evaluation.json records no onset threshold")
    return float(own["value"])


def test_clips(name: str) -> dict[str, str]:
    """Clip stem -> test set of the clips run *name* was scored on, as its own
    ``evaluation.json`` lists them."""
    return {
        c["clip_id"].split("/")[-1]: SETS[c["group"]]
        for c in evaluation(name)["clips"]
        if c["group"] in SETS
    }


# --- the clips (computing only) ----------------------------------------------------------


@functools.cache
def _listed() -> dict[str, tuple]:
    """Clip stem -> ``(resolved clip, recipe)`` over the corpus copy's clip list."""
    clip_list = load(ClipList, dataset_dir() / "clips.toml", check_files=False)
    return {c.source.stem: (c, clip_list.preprocess) for c in clip_list.resolve()}


def clip(stem: str) -> tuple:
    """``(resolved clip, recipe)`` of any clip of the corpus copy."""
    return _listed()[stem]


@functools.cache
def _all_entries() -> dict[str, features.ClipEntry]:
    """Every clip of the corpus copy with a cache entry, train and validation too."""
    clip_list = load(ClipList, dataset_dir() / "clips.toml", check_files=False)
    cache = cache_dir() / RECIPE
    out = {}
    for c in clip_list.resolve():
        found = features.lookup(c, clip_list.preprocess, cache)
        if found is not None:
            out[c.source.stem] = found
    return out


def _span(stem: str) -> str:
    """A clip's recording span (``<session>_span<k>``): its stem without the camera."""
    return stem.rsplit("_", 1)[0]


@functools.cache
def _spans() -> dict[str, str]:
    """Recording span -> the stem of the corpus copy's clip of it."""
    return {_span(s): s for s in _all_entries()}


def is_local(stem: str) -> bool:
    """Whether clip *stem* itself (its frames, not only its span) is at hand: in the
    corpus copy when computing, else among the clips the ``truth`` analysis kept."""
    if COMPUTE:
        return stem in _all_entries()
    return (ANALYSES / "truth" / f"{stem}.npz").is_file()


def entry(stem: str) -> features.ClipEntry:
    """The cache entry of any clip of the corpus copy."""
    found = _all_entries().get(stem)
    if found is None:
        raise FileNotFoundError(
            f"no cache entry for {stem} under {cache_dir() / RECIPE}"
        )
    return found


def truth_entry(stem: str) -> features.ClipEntry:
    """The entry that holds clip *stem*'s truth: its own, or, for a clip of another
    camera than the copy's, the copy's clip of the same recording span (the same
    thermistor, events and clock; never its frames); a dynamic-foraging clip's own."""
    if stem in _all_entries():
        return entry(stem)
    local = _spans().get(_span(stem))
    if local is not None:
        return entry(local)
    if stem in df_entries():
        return df_entries()[stem][0]
    raise KeyError(f"no clip of {_span(stem)} in the copy")


TRUTH = ("times", "signal", "inhale", "exhale", "excluded")
"""What the figures need of :func:`zephyr.benchmarks.analysis.truth`."""


def truth(stem: str) -> dict:
    """:func:`zephyr.benchmarks.analysis.truth` of a clip (computing), or the part of
    it the ``truth`` analysis kept (:data:`TRUTH`)."""
    if COMPUTE:
        return analysis.truth(truth_entry(stem))
    with np.load(made("truth", f"{stem}.npz")) as f:
        return {k: f[k] for k in TRUTH}


def _kept(stem: str, key: str):
    with np.load(made("truth", f"{stem}.npz")) as f:
        return f[key]


def grid(stem: str) -> np.ndarray:
    """The clip's output grid (s), which predictions are put on."""
    if COMPUTE:
        return np.load(truth_entry(stem).times)
    return _kept(stem, "grid")


def _session_camera(stem: str) -> tuple[str, str]:
    session, camera = re.split(r"_span\d+_", stem)
    return session, camera


def _thumbnail_path(stem: str) -> Path:
    session, camera = _session_camera(stem)
    name = f"{session}_{camera}.png"
    if not COMPUTE:
        return made("truth", f"thumbnails/{name}")
    for root in (dataset_dir(), MOUNTS["/data/zephyr-df-dataset"]):
        if (root / "thumbnails" / name).is_file():
            return root / "thumbnails" / name
    raise FileNotFoundError(f"no thumbnail {name} in the copy")


def thumbnail(stem: str) -> np.ndarray:
    """The session's whole first frame from the clip's camera."""
    from PIL import Image

    return np.asarray(Image.open(_thumbnail_path(stem)).convert("L"))


def box(stem: str) -> tuple[float, float, float, float]:
    """The clip's crop box in the pixels of its :func:`thumbnail` (the clip list's box
    is in the frame the recipe resizes to, ``target_size``)."""
    if not COMPUTE:
        return tuple(float(v) for v in _kept(stem, "box"))
    if stem in df_entries():
        found, width = df_entries()[stem][1], df_recipe().target_size[0]
    else:
        clip_, recipe = clip(stem)
        found, width = clip_.box, recipe.target_size[0]
    scale = thumbnail(stem).shape[1] / width
    return tuple(v * scale for v in found)


def gray(stem: str) -> tuple[np.ndarray, np.ndarray]:
    """``(frame_times, frames)``: the clip's gray crops, memory-mapped ``(T, 1, H, W)``
    (computing only)."""
    e = truth_entry(stem)
    return np.load(e.frame_times), np.load(e.frames, mmap_mode="r")


def inhale_frames(stem: str) -> list[np.ndarray]:
    """Three consecutive gray crops of the clip at its middle inhale (Fig. 1d)."""
    if not COMPUTE:
        return list(_kept(stem, "frames"))
    frame_t, frames = gray(stem)
    inhale = truth(stem)["inhale"]
    i = int(np.searchsorted(frame_t, inhale[len(inhale) // 2]))
    return [np.asarray(frames[j, 0]) for j in (i - 1, i, i + 1)]


def output_fs(stem: str) -> float:
    """The rate (Hz) of the clip's output grid."""
    if not COMPUTE:
        return float(_kept(stem, "output_fs"))
    if stem in df_entries():
        return float(df_recipe().output_fs)
    return float(clip(stem)[1].output_fs)


def keep_truth(stem: str) -> None:
    """Write what the figures need of clip *stem* into the ``truth`` analysis: its
    truth (:data:`TRUTH`), output grid, crop box (on its thumbnail), three crops at its
    middle inhale and output rate; and the session's thumbnail from its camera."""
    import shutil

    t = truth(stem)
    out = ANALYSES / "truth"
    (out / "thumbnails").mkdir(parents=True, exist_ok=True)
    np.savez(
        out / f"{stem}.npz",
        **{k: np.asarray(t[k]) for k in TRUTH},
        grid=grid(stem),
        box=np.asarray(box(stem), float),
        frames=np.stack(inhale_frames(stem)),
        output_fs=output_fs(stem),
    )
    source = _thumbnail_path(stem)
    if not (out / "thumbnails" / source.name).is_file():
        shutil.copy2(source, out / "thumbnails" / source.name)


# --- predictions: the runs' saved traces ---------------------------------------------------


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp(dev: torch.device) -> torch.dtype | None:
    if dev.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@functools.cache
def network(name: str) -> tuple:
    """``(model, mean, std, window, frame_chunk)`` of a network's first seed."""
    model, mean, std, state = load_checkpoint(run_file(name, "best.pt"), device())
    params = state.get("params") or {}
    return (
        model,
        mean,
        std,
        int(params.get("infer_window", 1024)),
        int(params.get("frame_chunk", 256)),
    )


@functools.cache
def _trace_files(traces: str, suffix: str = ".parquet") -> dict[str, str]:
    """Clip stem -> trace file name in a run's *traces* folder (its
    ``feat-<stem>-<key><suffix>``), listed once."""
    return {
        p.name.removeprefix("feat-").removesuffix(suffix).rsplit("-", 1)[0]: p.name
        for p in _runs_path(traces).glob(f"feat-*{suffix}")
    }


def _traces(name: str) -> tuple[str, str]:
    """Run *name*'s ``traces/`` folder and the suffix of its files."""
    m = RUNS[name]
    if m.kind == "network":
        return f"{folder(m)}/{seeds(folder(m))[0]}/{TRACES}", ".parquet"
    if m.kind == "facemap":
        return f"{folder(m)}/facemap/{m.variant}/traces", ".npy"
    return f"{folder(m)}/pixel/{m.variant.split('/')[0]}/traces", ".npy"


def _saved(
    name: str, stem: str, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """``(trace, onset)`` as run *name* saved it when it scored clip *stem* (its
    ``traces/``), on the clip's output grid *times*; ``None`` if it saved none."""
    traces, suffix = _traces(name)
    found = _trace_files(traces, suffix).get(stem)
    if found is None:
        return None
    path = _runs_path(f"{traces}/{found}")
    if suffix == ".npy":
        raw = np.load(path)
        return (raw if RUNS[name].kind == "facemap" else _calibrated(name, raw)), None
    s = pd.read_parquet(path)
    signal = s["signal"].to_numpy()
    onset = s["onset"].to_numpy() if "onset" in s else None
    if "time" in s and not np.array_equal(s["time"].to_numpy(), times[: len(s)]):
        at = s["time"].to_numpy()
        signal = np.interp(times, at, signal)
        onset = None if onset is None else np.interp(times, at, onset)
    return signal, onset


def _calibrated(name: str, raw: np.ndarray) -> np.ndarray:
    """A pixel method's output-grid trace as its variant reports it."""
    trace = common.blind_polarity(raw)
    if RUNS[name].variant.endswith("/calibrated"):
        fitted = evaluation(name)
        trace = fitted["sign"] * common.shift(trace, fitted["lag_samples"])
    return trace


def _predicted(
    name: str, e: features.ClipEntry
) -> tuple[np.ndarray, np.ndarray | None]:
    """``(trace, onset)`` predicted here (what :func:`wall_time` times)."""
    m = RUNS[name]
    if m.kind == "pixel":
        method = m.variant.split("/")[0]
        raw = baselines.pixel_trace(method, e, common.FS, PIXEL_BIN)
        return _calibrated(name, common.to_output_grid(e, raw)), None
    model, mean, std, window, chunk = network(name)
    dev = device()
    return predict_clip(
        model,
        e,
        mean,
        std,
        window=window,
        device=dev,
        frame_chunk=chunk,
        amp_dtype=_amp(dev),
    )


def prediction(name: str, stem: str) -> dict:
    """Run *name*'s trace on clip *stem* as it saved it (z-scored, on the clip's output
    grid), its onset probability (zeros without a head) and, with a head, its inhales at
    the run's own onset threshold."""
    times = grid(stem)
    found = _saved(name, stem, times)
    if found is None:
        raise FileNotFoundError(f"{name}: the run saved no trace of {stem}")
    signal, onset = found
    n = min(len(times), len(signal))
    signal = signal[:n].astype(np.float32)
    signal = (signal - signal.mean()) / (signal.std() or 1.0)
    onset = np.zeros(n, np.float32) if onset is None else onset[:n].astype(np.float32)
    out = {"signal": signal, "onset": onset, "times": times[:n]}
    if RUNS[name].head:
        out["threshold"] = head_threshold(name)
        peaks = head_event_indices(out["onset"], out["times"], out["threshold"])
        out["head"] = out["times"][peaks]
    return out


# --- the analysis tables, per run -------------------------------------------------------------


def tables(
    name: str, progress: Callable[[Iterable], Iterable] = iter
) -> dict[str, pd.DataFrame]:
    """:func:`zephyr.benchmarks.analysis.clip_tables` of one run over every test clip,
    labelled by method, clip and set (the ``tables`` analysis). *progress* wraps the
    loop over clips (e.g. a progress bar)."""
    paths = {k: made("tables", f"{_tag(name)}-{k}.parquet") for k in analysis.TABLES}
    if all(p.is_file() for p in paths.values()):
        return {k: pd.read_parquet(p) for k, p in paths.items()}
    parts: dict[str, list] = {k: [] for k in analysis.TABLES}
    sets = test_clips(name)
    for stem in progress(list(sets)):
        label = {"method": name, "stem": stem, "set": sets[stem]}
        found = analysis.clip_tables(
            prediction(name, stem), truth(stem), RUNS[name].head
        )
        for k, v in found.items():
            parts[k].append(v.assign(**label))
    out = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    for k, v in out.items():
        paths[k].parent.mkdir(parents=True, exist_ok=True)
        v.to_parquet(paths[k])
    return out


def errors(name: str, stem: str) -> pd.DataFrame:
    """:func:`zephyr.benchmarks.analysis.errors` of one method on one test clip
    (computing only: it reads the clip's frames)."""
    frame_t, frames = gray(stem)
    found = analysis.errors(
        prediction(name, stem), truth(stem), RUNS[name].head, frame_t, frames
    )
    return found.assign(stem=stem, method=name)


def set_errors(name: str, set_name: str) -> pd.DataFrame:
    """:func:`errors` of every clip of one test set (``SETS`` value), the ``errors``
    analysis."""

    def make() -> pd.DataFrame:
        stems = sorted(s for s, v in test_clips(name).items() if v == set_name)
        return pd.concat([errors(name, s) for s in stems], ignore_index=True)

    return _frame_memo("errors", f"{_tag(name)}-{set_name.split(':')[0]}.parquet", make)


def animal_breath_rates() -> dict[str, np.ndarray]:
    """Every thermistor breath's rate (Hz) per animal, over the clips of every session
    the corpus assigns to a set, from each clip's cached events (``breath-rates``)."""

    def make() -> pd.DataFrame:
        folds = pd.read_parquet(dataset_raw("folds.parquet"))
        used = folds[folds["set"].notna()]
        subject = dict(
            zip(used["session_id"], used["subject_id"].astype(str), strict=True)
        )
        parts = []
        for stem, e in _all_entries().items():
            animal = subject.get(stem.rsplit("_span", 1)[0])
            if animal is not None:
                rate = 1.0 / np.diff(np.load(e.events)["onset_times"])
                parts.append(pd.DataFrame({"animal": animal, "rate_hz": rate}))
        return pd.concat(parts, ignore_index=True)

    rates = _frame_memo("breath-rates", "rates.parquet", make)
    return {a: g["rate_hz"].to_numpy() for a, g in rates.groupby("animal", sort=True)}


PROMINENCE = (0.1, 0.05)
"""The lowered prominence fractions of :func:`label_added` (the corpus's is 0.2)."""


def label_added() -> pd.DataFrame:
    """Inhalations the corpus detector adds, per breath of its labels, when its
    prominence fraction is lowered to each of :data:`PROMINENCE`, on the raw thermistor
    of every new-animal clip of the reference over the folds (the ``prominence``
    analysis; ``setting`` is the fraction as printed, ``fold`` the clip's)."""

    def make() -> pd.DataFrame:
        clips_ = (
            cv_scores("clips")
            .query("method == @REFERENCE and set == @SETS['test_a']")
            .drop_duplicates("stem")
        )
        parts = []
        for stem, fold in zip(clips_["stem"], clips_["fold"], strict=True):
            if not is_local(stem):
                continue
            t = truth(stem)
            for frac in PROMINENCE:
                parts.append(
                    analysis.added_inhales(
                        t["inhale"], analysis.redetect(t, prominence_frac=frac)
                    ).assign(fold=fold, setting=f"{frac:g}")
                )
        return pd.concat(parts, ignore_index=True)

    return _frame_memo("prominence", "added.parquet", make)


# --- inference time --------------------------------------------------------------------------


def wall_time(labels, n_clips: int = 8) -> pd.DataFrame:
    """Inference seconds per clip of each method of *labels* (its run on the first
    fold; :mod:`zephyr.benchmarks.timing`: the same test clips for all, one untimed
    warm-up clip, files read into the OS cache first). Networks on the GPU at each one's
    own inference window; pixel and Facemap on the CPU. The ``timing`` analysis."""
    path = made("timing", "timing.json")
    done = json.loads(path.read_text()) if path.is_file() else {}
    todo = [n for n in labels if n not in done]
    if todo and not COMPUTE:
        raise FileNotFoundError(f"{path} has not timed {todo}: run analyses.py")
    if todo:
        stems = [s for s in sorted(test_clips(SINGLE)) if is_local(s)]
        entries = [entry(s) for s in stems]
        timed = timing.pick_clips(entries, n_clips)
        warm = timing.pick_warmup(entries, timed)
        timing.warm_file_cache([*timed, warm])
    for label in todo:
        name = label if label in RUNS else cv_name(label, CV.folds[0])
        m = RUNS[name]
        if m.kind == "network":
            model = network(name)[0]
            dev = device()

            def infer(e, name=name, dev=dev):
                _predicted(name, e)  # timed: always inference
                if dev.type == "cuda":
                    torch.cuda.synchronize()

            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            result = timing.time_method(infer, timed, warm)
            result["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
            result["peak_gb"] = (
                torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None
            )
            result["device"] = timing.hardware()["gpu"] or "cpu"
        elif m.kind == "pixel":
            method = m.variant.split("/")[0]
            result = timing.time_method(
                timing.pixel_method(method, common.FS, PIXEL_BIN), timed, warm
            )
            result["device"] = "cpu (numpy)"
        else:
            fitted = evaluation(name)
            result = timing.time_method(
                timing.facemap_method(
                    m.variant,
                    _require(_runs_path(f"{folder(m)}/facemap/bases.npz")),
                    n_components=fitted["n_components"],
                    lags=np.asarray(fitted["lags"]),
                    seed=0,
                ),
                timed,
                warm,
            )
            result["device"] = "cpu (numpy)"
        result["run"] = name
        result["video_s"] = [len(np.load(e.times)) / timing.CLIP_FS for e in timed]
        result["when"] = time.strftime("%Y-%m-%d %H:%M")
        done[label] = result
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(done, indent=2))
    rows = []
    for label in labels:
        r = done[label]
        seconds = np.array([c["seconds"] for c in r["clips"]])
        video = np.array(r["video_s"])
        rows.append(
            {
                "method": label,
                "device": r["device"],
                "parameters (M)": r.get("params_m"),
                "s per clip (median)": float(np.median(seconds)),
                "s per min of video": float(np.median(seconds / video * 60)),
                "x real time": float(np.median(video / seconds)),
                "frames per s": float(np.median(video * timing.CLIP_FS / seconds)),
                "peak GPU memory (GB)": r.get("peak_gb"),
            }
        )
    return pd.DataFrame(rows)


# --- a run's own held-out clips of other cameras, and saliency ----------------------------


def held_out_entries(
    name: str, spans: Iterable[str]
) -> list[tuple[features.ClipEntry, tuple[int, ...], tuple[int, int]]]:
    """Network run *name*'s own held-out clips (test set A of its ``config.json``) of
    the recording *spans*, as the run read them: their entries in the copy's feature
    cache of their camera. Each comes with its box and the frame size
    (``target_size``) the box is on."""
    config = json.loads(run_file(name, "config.json").read_text())
    listed = config["test_clip_lists"]["test_a"]
    wanted = set(spans)
    clips_ = [
        c
        | {
            k: local_path(c[k])
            for k in ("video", "timestamps", "thermistor")
            if isinstance(c.get(k), str)
        }
        for c in listed["clip"]
        if Path(c["video"]).stem.rsplit("_", 1)[0] in wanted
    ]
    clip_list = ClipList.model_validate({**listed, "clip": clips_})
    out = []
    for c in clip_list.resolve():
        cache = cache_folder(c, clip_list.preprocess)
        found = features.lookup(c, clip_list.preprocess, cache)
        if found is None:
            raise FileNotFoundError(f"no cache entry for {c.name} under {cache}")
        _require(Path(c.video))
        out.append((found, tuple(c.box), tuple(clip_list.preprocess.target_size)))
    return out


def with_unit(table: pd.DataFrame) -> pd.DataFrame:
    """*table* (a ``session`` and ``subject`` per row) with its ``rig`` and ``unit``
    (an animal on one rig, the unit of comparison)."""
    folds = pd.read_parquet(dataset_raw("folds.parquet"))
    rig = table["session"].map(
        dict(zip(folds["session_id"], folds["rig"], strict=True))
    )
    return table.assign(rig=rig, unit=table["subject"].astype(str) + "@" + rig)


SALIENCY_BINS = np.arange(-1.0, 1.0 + 1e-9, 1 / 60) - 1 / 120
"""Offsets (s) around each inhale the saliency profile is binned at."""


def _picked(name: str) -> pd.Series:
    """For every new-animal unit of run *name*, the clip nearest the unit's median
    correlation."""
    c = scores([name]).query("set == @SETS['test_a']")
    c = with_unit(c[c["stem"].map(is_local)])
    median = c.groupby("unit")["correlation"].transform("median")
    typical = (c["correlation"] - median).abs()
    return c.loc[typical.groupby(c["unit"]).idxmin()].set_index("unit")["stem"]


def _face_saliency() -> dict:
    keys = ("occlusion", "trace", "gradient", "random", "profile")
    rows = {k: [] for k in (*keys, "crop")}
    units, of, stems = [], [], []
    dev = device()
    for f in CV.folds:
        name = cv_name(REFERENCE, f)
        model, mean, std, _, _ = network(name)
        model = model.to(dev).float().eval()
        control = analysis.randomized(model)
        for unit, stem in _picked(name).items():
            found = analysis.clip_saliency(
                model,
                control,
                truth_entry(stem),
                mean,
                std,
                truth(stem)["inhale"],
                SALIENCY_BINS,
                device=dev,
            )
            for k in keys:
                rows[k].append(found[k])
            t, frames = gray(stem)
            rows["crop"].append(np.asarray(frames[:: max(1, len(t) // 50), 0]).mean(0))
            units.append(unit)
            of.append(f)
            stems.append(stem)
    out = {k: np.array(v) for k, v in rows.items()}
    return out | {
        "unit": np.array(units),
        "fold": np.array(of),
        "stem": np.array(stems),
        "offset_s": (SALIENCY_BINS[:-1] + SALIENCY_BINS[1:]) / 2,
    }


def _camera_saliency(label: str, fold: str, spans: dict[str, str]) -> dict:
    """The maps of camera network *label* of *fold* on its own clips of *spans*
    (span -> unit), with each clip's whole first frame and its box."""
    import cv2

    dev = device()
    name = cv_name(label, fold)
    model, mean, std, _, _ = network(name)
    model = model.to(dev).float().eval()
    control = analysis.randomized(model)
    keys = ("occlusion", "trace", "gradient", "random", "profile")
    rows = {k: [] for k in (*keys, "crop", "frame", "box")}
    units, stems = [], []
    for e, box_, size in held_out_entries(name, spans):
        found = analysis.clip_saliency(
            model,
            control,
            e,
            mean,
            std,
            analysis.truth(e)["inhale"],
            SALIENCY_BINS,
            device=dev,
        )
        for k in keys:
            rows[k].append(found[k])
        frames = np.load(e.frames, mmap_mode="r")
        rows["crop"].append(np.asarray(frames[:: max(1, len(frames) // 50), 0]).mean(0))
        video = cv2.VideoCapture(str(e.video))
        _, image = video.read()
        video.release()
        rows["frame"].append(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), size))
        rows["box"].append(box_)
        stem = Path(str(e.video)).stem
        units.append(spans[stem.rsplit("_", 1)[0]])
        stems.append(stem)
    out = {k: np.array(v) for k, v in rows.items()}
    return out | {
        "unit": np.array(units),
        "fold": np.array([fold] * len(units)),
        "stem": np.array(stems),
        "offset_s": (SALIENCY_BINS[:-1] + SALIENCY_BINS[1:]) / 2,
    }


def saliency() -> dict[str, dict]:
    """Supplementary Fig. 6's maps, by camera (the ``saliency`` analysis; it runs the
    networks on the GPU). The face network of each fold on its new animals: for every
    unit, the clip nearest the unit's median correlation, two 8.5-s windows
    (:func:`zephyr.benchmarks.analysis.clip_saliency`). The side and front networks of
    fold cv_0 on the same spans: their clips are each run's own held-out list."""
    face = _arrays_memo("saliency", "face-cv.npz", _face_saliency)
    out = {"Face camera": face}
    fold = CV.folds[0]
    spans = {
        str(s).rsplit("_", 1)[0]: str(u)
        for s, u, f in zip(face["stem"], face["unit"], face["fold"], strict=True)
        if f == fold
    }
    for camera in ("side", "front"):
        out[f"{camera.title()} camera"] = _arrays_memo(
            "saliency",
            f"{camera}-{fold}.npz",
            lambda camera=camera: _camera_saliency(
                f"Zephyr, {camera} camera", fold, spans
            ),
        )
    return out


def clip_scores(name: str) -> pd.DataFrame:
    """Every clip a network run scored, as its own ``evaluation.json`` gives it: the
    ``set`` (its group), ``stem``, ``session`` (the recording), ``camera`` (after the
    span in the stem), ``correlation`` and ``inhale`` (the head's inhale F1 at the
    checkpoint's own onset threshold)."""
    found = evaluation(name)
    rows = pd.DataFrame(found["clips"])
    stem = rows["clip_id"].map(lambda c: Path(c).name)
    return pd.DataFrame(
        {
            "set": rows["group"],
            "stem": stem,
            "session": rows["recording"].str.rsplit("#", n=1).str[1],
            "camera": stem.str.split(r"_span\d+_", regex=True).str[1],
            "correlation": rows["correlation"],
            "inhale": rows["head_inhale_f1"],
        }
    )


def run_of(names: Iterable[str], stem: str) -> str:
    """The run of *names* (pooled, :func:`pooled`) that scored clip *stem*."""
    for n in names:
        if stem in set(clip_scores(n)["stem"]):
            return n
    raise KeyError(f"none of {list(names)} scored {stem}")


# --- dynamic foraging (computing only) ----------------------------------------------------


@functools.cache
def df_recipe():
    """The recipe of the dynamic-foraging clips."""
    return manuscript_clips("clips-df.toml").preprocess


@functools.cache
def df_entries() -> dict[str, tuple[features.ClipEntry, tuple[int, ...]]]:
    """Each dynamic-foraging clip's cache entry and crop box (on the 360 x 270 frame),
    by stem, from the capsule's clip list and the copy's caches."""
    clip_list = manuscript_clips("clips-df.toml")
    out = {}
    for c in clip_list.resolve():
        found = features.lookup(
            c, clip_list.preprocess, cache_folder(c, clip_list.preprocess)
        )
        if found is not None:
            out[Path(str(c.video)).stem] = (found, tuple(c.box))
    return out


# --- Fig. 1b, c: one session in the task ---------------------------------------------------


def _asset_table(asset: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(_require(OPEN_DATA / asset / f"{name}.parquet"))


def _odor_sites(sites: pd.DataFrame) -> pd.DataFrame:
    """The odor sites of a session's ``sites`` table: onset, leaving, kind and reward."""
    from . import plots

    odor = sites[
        (sites["site_label"] == "RewardSite") & sites["odor_onset_time"].notna()
    ]
    return pd.DataFrame(
        {
            "odor": odor["odor_onset_time"].to_numpy(),
            "leave": odor["stop_time"].to_numpy(),
            "kind": [
                plots.site_kind(bool(c), bool(r))
                for c, r in zip(odor["has_choice"], odor["has_reward"], strict=True)
            ],
            "reward": odor["reward_onset_time"].to_numpy(),
        }
    )


def _task() -> dict[str, pd.DataFrame]:
    from zephyr.events import detect

    asset = picks.TASK_ASSET
    session = asset.split("_processed")[0]
    name = cv_name(REFERENCE, picks.TASK_FOLD)
    speed = _asset_table(asset, "position_velocity")["velocity"]
    licks = _asset_table(asset, "licks")
    sites = _odor_sites(_asset_table(asset, "sites"))
    out = {
        "licks": pd.DataFrame({"time": licks.index[licks["is_lick_onset"]].to_numpy()}),
        "sites": sites,
    }
    for span in (1, 2):
        stem = f"{session}_span{span}_FaceCamera"
        p, t = prediction(name, stem), truth(stem)
        g = p["times"]
        out[f"clip{span}"] = pd.DataFrame(
            {
                "time": g,
                "speed": np.interp(g, speed.index.to_numpy(), speed.to_numpy()),
                "video": analysis.breathing_rate(p["head"], g),
                "thermistor": analysis.breathing_rate(t["inhale"], g),
            }
        )
        out[f"truth{span}"] = pd.DataFrame({"time": t["times"], "signal": t["signal"]})
    # The thermistor over the whole session, its events found as the corpus finds them
    # (the clip list's detector), on a 60-Hz grid.
    raw = pd.read_parquet(
        _require(dataset_dir() / "raw" / "thermistor" / f"{session}.parquet")
    )
    events = load(ClipList, dataset_dir() / "clips.toml", check_files=False).events
    inhale = detect(events.method, raw, params=events.params).inhale
    g = np.arange(raw["Time"].iloc[0], raw["Time"].iloc[-1], 1 / 60)
    session_rate = pd.DataFrame(
        {
            "time": g,
            "speed": np.interp(g, speed.index.to_numpy(), speed.to_numpy()),
            "rate": analysis.breathing_rate(inhale, g),
        }
    )
    out["session"] = session_rate.assign(
        thermistor_hz=1 / float(np.median(np.diff(raw["Time"])))
    )
    lags = np.arange(-2.0, 5.0 + 1e-9, 1 / 60)
    inside = sites[sites["odor"].between(g[0] - lags[0], g[-1] - lags[-1])]
    parts = []
    for kind, group in inside.groupby("kind"):
        for signal in ("rate", "speed"):
            x = analysis.event_aligned(
                g, session_rate[signal].to_numpy(), group["odor"].to_numpy(), lags
            )
            low, high = analysis.bootstrap_mean(x)
            parts.append(
                pd.DataFrame(
                    {
                        "site": kind,
                        "signal": signal,
                        "lag": lags,
                        "mean": np.nanmean(x, 0),
                        "low": low,
                        "high": high,
                        "n": len(x),
                    }
                )
            )
    out["aligned"] = pd.concat(parts, ignore_index=True)
    return out


TASK_TABLES = (
    "licks",
    "sites",
    "clip1",
    "clip2",
    "truth1",
    "truth2",
    "session",
    "aligned",
)


def task() -> dict[str, pd.DataFrame]:
    """Fig. 1b, c: the session of :data:`picks.TASK_ASSET` in the task (the ``task``
    analysis): its ``licks`` and odor ``sites``; each of its two clips predicted by the
    fold that holds the animal out (``clip1``, ``clip2``: speed, the video's and the
    thermistor's breathing rate) and its thermistor (``truth1``, ``truth2``); the whole
    session's thermistor rate and speed (``session``, with the thermistor's
    ``thermistor_hz``); rate and speed around odor onset per kind of site
    (``aligned``)."""
    paths = {k: made("task", f"{k}.parquet") for k in TASK_TABLES}
    if all(p.is_file() for p in paths.values()):
        return {k: pd.read_parquet(p) for k, p in paths.items()}
    out = _task()
    for k, p in paths.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        out[k].to_parquet(p)
    return out


# --- sessions without a thermistor ----------------------------------------------------------


def _application_clip(session: str, camera: str = "FaceCamera") -> dict:
    """The capsule's clip of one camera of an application session (its box, frame
    and frame rate)."""
    raw = tomllib.loads((MANUSCRIPT / "clips" / "clips-application.toml").read_text())
    found = next(c for c in raw["clip"] if c["group"] == f"{session}_{camera}")
    return found | {"target_size": raw["preprocess"]["target_size"]}


def application(session: str) -> dict:
    """One of the manifest's sessions without a thermistor (the ``application``
    analysis): the face camera's prediction on the Harp clock (``times``, ``signal``,
    ``onset``; ``video_time``, its time in the video, which quality-control windows use)
    and its inhalations (``head``), and from its processed asset the running ``speed``,
    ``sites``, ``licks``, ``subject`` and the ``date`` it was acquired."""
    entry_ = next(s for s in RESULTS.application.session if s.session == session)

    def make() -> pd.DataFrame:
        pred = pd.read_parquet(
            _require(
                LOCAL_RUNS
                / "application"
                / f"{session}_FaceCamera"
                / "video.prediction.parquet"
            )
        )
        clock = pd.read_csv(
            _require(
                OPEN_DATA / session / "behavior-videos" / "FaceCamera" / "metadata.csv"
            ),
            usecols=["ReferenceTime"],
        )
        return pred.assign(
            times=clock["ReferenceTime"].to_numpy()[pred["frame"].to_numpy()]
        )

    pred = _frame_memo("application", f"{session}/prediction.parquet", make)
    speed = _frame_memo(
        "application",
        f"{session}/speed.parquet",
        lambda: _asset_table(entry_.asset, "position_velocity")[["velocity"]],
    )["velocity"]
    sites = _frame_memo(
        "application",
        f"{session}/sites.parquet",
        lambda: _asset_table(entry_.asset, "sites"),
    )

    def licks_() -> pd.DataFrame:
        licks = _asset_table(entry_.asset, "licks")
        return pd.DataFrame({"time": licks.index[licks["is_lick_onset"]].to_numpy()})

    licks = _frame_memo("application", f"{session}/licks.parquet", licks_)

    def meta_() -> dict:
        subject = json.loads(
            _require(OPEN_DATA / entry_.asset / "subject.json").read_text()
        )
        acquired = json.loads(
            _require(OPEN_DATA / entry_.asset / "acquisition.json").read_text()
        )
        return {
            "subject": str(subject["subject_id"]),
            "date": acquired["acquisition_start_time"],
        }

    meta = _memo(
        "application",
        f"{session}/meta.json",
        meta_,
        lambda p: json.loads(p.read_text()),
        lambda d, p: p.write_text(json.dumps(d)),
    )
    times = pred["times"].to_numpy()
    return {
        "times": times,
        "video_time": pred["time"].to_numpy(),
        "signal": pred["signal"].to_numpy(),
        "onset": pred["onset"].to_numpy(),
        "head": times[pred["inhale"].to_numpy(bool)],
        "speed": speed,
        "sites": sites,
        "licks": licks["time"].to_numpy(),
        "subject": meta["subject"],
        "date": pd.Timestamp(meta["date"]),
    }


def application_frames(
    session: str, video_times: Sequence[float]
) -> tuple[list[np.ndarray], tuple[float, ...]]:
    """Gray frames of an application session's face video at *video_times* (s in the
    video, frame *i* at *i* / the clip's frame rate, as its prediction), and the crop
    box in their pixels (the ``application`` analysis)."""

    def make() -> dict:
        import cv2

        first = _application_clip(session)
        video = cv2.VideoCapture(_require(Path(local_path(first["video"]))).as_posix())
        frames = []
        for t in video_times:
            video.set(cv2.CAP_PROP_POS_FRAMES, round(t * first["frame_rate"]))
            _, image = video.read()
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        video.release()
        scale = frames[0].shape[1] / (first.get("frame") or first["target_size"])[0]
        return {
            "frames": np.stack(frames),
            "box": np.asarray(first["box"], float) * scale,
        }

    key = "-".join(f"{t:.3f}" for t in video_times)
    found = _arrays_memo("application", f"{session}/frames-{key}.npz", make)
    return list(found["frames"]), tuple(float(v) for v in found["box"])


def is_application_video(stem: str) -> bool:
    """Whether *stem* is a video of one of the manifest's application sessions."""
    try:
        application_session(stem)
    except KeyError:
        return False
    return True


def application_session(stem: str) -> str:
    """The manifest's application session a video of it (*stem*, any camera) is from:
    the longest session name the stem starts with."""
    found = [
        s.session
        for s in RESULTS.application.session
        if stem == s.session or stem.startswith(f"{s.session}_")
    ]
    if not found:
        raise KeyError(f"{stem}: not a video of the manifest's application sessions")
    return max(found, key=len)


# --- quality control ----------------------------------------------------------------------


def quality_folders(reports: str) -> list[Path]:
    """The report folders a ``[quality]`` set names (a glob under the root)."""
    if reports.startswith("analysis/"):
        found = sorted(ANALYSES.glob(reports.removeprefix("analysis/")))
    else:
        found = sorted(LOCAL_RUNS.glob(reports))
    if not found:
        raise FileNotFoundError(f"no folder {reports} in the runs")
    return found


def report_truth(folder: Path) -> pd.DataFrame:
    """The real scores of each prediction in *folder* (``zephyr predict
    --quality-control``) over the whole clip and over each window its report measures
    (:func:`zephyr.benchmarks.analysis.window_truth`): the paper's check on the
    label-free scores; empty for videos without a thermistor (computing only)."""
    from zephyr.predict import SUFFIX

    parts = []
    for path in sorted(Path(folder).glob(f"*{SUFFIX}")):
        stem = path.name.removesuffix(SUFFIX)
        try:
            e = truth_entry(stem)
        except KeyError:
            continue  # a video without a thermistor
        if e.thermistor is None:
            continue
        table = analysis.window_truth(e, pd.read_parquet(path))
        parts.append(table.assign(stem=stem))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _quality_windows() -> pd.DataFrame:
    from zephyr.benchmarks.probe import load_probes
    from zephyr.quality_control import load_reports
    from zephyr.quality_control.reference import Reference

    quality = RESULTS.quality
    validation = pd.DataFrame(
        Reference.load(_require(_runs_path(quality.reference))).validation
    )
    parts = [
        validation.rename(columns={"clip_id": "stem"}).assign(
            set="Validation", role="validation"
        )
    ]
    for s in quality.set:
        folders = quality_folders(s.reports)
        if s.noise is not None:
            table = pd.concat([load_probes(f, windows=True) for f in folders])
            table = table[table["sigma"] == s.noise].drop(columns="sigma")
        else:
            table = pd.concat([load_reports(f)[1] for f in folders], ignore_index=True)
            truth_ = pd.concat([report_truth(f) for f in folders], ignore_index=True)
            if len(truth_):
                windows = truth_[~truth_["whole"]].drop(columns=["whole", "end"])
                table = with_truth(table, windows)
        if s.role in ("application", "camera"):
            # only the manifest's sessions, whatever else the reports folder holds
            table = table[table["stem"].map(is_application_video)]
        parts.append(table.assign(set=s.label, role=s.role))
    windows = pd.concat(parts, ignore_index=True)
    return windows.assign(session=windows["stem"].map(video_session))


def quality_windows() -> pd.DataFrame:
    """Every 30-s window of the manifest's ``[quality]`` sets, one row each: ``set``,
    ``role`` (``validation`` for the reference's own windows), ``stem``, its
    ``session`` (:func:`video_session`), the measures and, where the truth is known,
    ``truth_<metric>`` (the ``quality`` analysis). Sets of sessions without a thermistor
    keep only the manifest's ``[application]`` sessions."""
    return _frame_memo("quality", "windows.parquet", _quality_windows)


_SESSION = re.compile(r"\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def video_session(stem: str) -> str:
    """The acquisition session a video (*stem*: a clip, span or camera of it) is from:
    the ``<subject>_<date>_<time>`` its name holds."""
    found = _SESSION.search(stem)
    if found is None:
        raise ValueError(f"{stem}: no session in the name")
    return found.group(0)


def with_truth(table: pd.DataFrame, truth_: pd.DataFrame) -> pd.DataFrame:
    """*table* (a report's windows) with each window's real scores, by video and start."""
    return (
        table.assign(_start=table["start"].round(6))
        .merge(
            truth_.assign(_start=truth_["start"].round(6)).drop(columns="start"),
            on=["stem", "_start"],
            how="left",
            validate="one_to_one",
        )
        .drop(columns="_start")
    )


def application_window() -> pd.Series:
    """Fig. 5b's window: the lowest of the sessions without a thermistor (face camera)
    on the first quality-control measure, with its ``session``."""
    from . import style

    label = next(q.label for q in RESULTS.quality.set if q.role == "application")
    windows = quality_windows()
    apps = windows[windows["set"] == label]
    row = analysis.example_windows(apps, next(iter(style.QC_MEASURES))).iloc[0]
    return pd.concat([row, pd.Series({"session": application_session(row["stem"])})])


def quality_traces(label: str, stem: str) -> pd.DataFrame:
    """One video's prediction (``time``, ``signal``, ``onset``, ``inhale``) in the
    ``[quality]`` set *label*."""
    from zephyr.predict import SUFFIX

    reports = next(s.reports for s in RESULTS.quality.set if s.label == label)
    for f in quality_folders(reports):
        if (f / f"{stem}{SUFFIX}").is_file():
            return pd.read_parquet(f / f"{stem}{SUFFIX}")
    raise FileNotFoundError(f"{label}: no prediction of {stem}")


def quality_noise() -> tuple[pd.DataFrame, Path]:
    """The ``[quality]`` noise null over whole clips (one row per clip and level:
    ``stem``, ``sigma``, the measures and ``truth_<metric>``), and its folder."""
    from zephyr.benchmarks.probe import load_probes

    (found,) = quality_folders(next(s.reports for s in RESULTS.quality.set if s.noise))
    return load_probes(found), found


def quality_noise_traces(stem: str) -> dict[float, pd.DataFrame]:
    """One clip's saved trace at every noise level of the ``[quality]`` null."""
    from zephyr.benchmarks.probe import probe_traces

    return probe_traces(quality_noise()[1], stem)


# --- Fig. S4: what the three cameras see -------------------------------------------------------


def camera_views(session: str) -> dict[str, tuple[np.ndarray, tuple[float, ...]]]:
    """One session's three cameras at half size, with the crop box each network reads
    (the first span's box, in the shown pixels): the face camera's thumbnail and the
    corpus's clip list; the side and front cameras' first frame of the clip, at the
    frame size their clip lists' boxes are on (the ``corpus`` analysis)."""
    lists = {
        "Face camera": None,
        "Side camera": "clips-side-720.toml",
        "Front camera": "clips-front-720.toml",
    }

    def make() -> dict:
        import cv2

        out = {}
        for camera, name in lists.items():
            key = camera.split()[0].lower()
            if name is None:
                stem = f"{session}_span1_FaceCamera"
                clip_, recipe = clip(stem)
                image = thumbnail(stem)
                found, width = clip_.box, recipe.target_size[0]
            else:
                listed = manuscript_clips(name, [session])
                c = next(c for c in listed.resolve() if "_span1_" in c.source.stem)
                video = cv2.VideoCapture(_require(Path(str(c.video))).as_posix())
                _, frame = video.read()
                video.release()
                size = tuple(listed.preprocess.target_size)
                image = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size)
                found, width = c.box, size[0]
            scale = image.shape[1] / width / 2  # shown at half size
            out[f"{key}_image"] = image[::2, ::2]
            out[f"{key}_box"] = np.asarray(found, float) * scale
        return out

    found = _arrays_memo("corpus", f"views/{session}.npz", make)
    return {
        camera: (
            found[f"{camera.split()[0].lower()}_image"],
            tuple(float(v) for v in found[f"{camera.split()[0].lower()}_box"]),
        )
        for camera in lists
    }
