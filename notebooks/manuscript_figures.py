import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import os
    import tomllib
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    from utils import architecture, corpus_flow, picks, plots, runs, style, values
    from zephyr.benchmarks import analysis
    from zephyr.corpus.config import DatasetSpec

    style.use_style()

    FIGURES = Path(
        os.environ.get(
            "ZEPHYR_MANUSCRIPT_FIGURES", runs.REPO / "manuscript" / "figures"
        )
    )
    LAYOUT = tomllib.loads((FIGURES / "layout.toml").read_text(encoding="utf8"))


@app.cell
def intro():
    mo.md(f"""
    # Manuscript figure panels

    Draws each data panel of the manuscript's figures at its printed size, as laid out
    in `{FIGURES / "layout.toml"}`, and writes it as `<figure>/<panel>.svg` beside the
    figure's `.typ` file, which embeds the panels as vectors and adds their letters,
    and the tables and `values.toml` the text cites. It reads only the runs of the
    results manifest (`{runs.MANIFEST.name}`), in the local copy `{runs.LOCAL_RUNS}`,
    and what `analyses.py` made from them and the data (`{runs.ANALYSES}`): every score
    is a run's own; nothing is trained, predicted, or read from the raw data here. Set
    `ZEPHYR_ANALYSES` to read the analyses from elsewhere (before they are uploaded),
    `ZEPHYR_MANUSCRIPT_FIGURES` to write elsewhere. The examples picked by hand are
    `utils/picks.py`.

    Run from a clean, committed working tree, as a script:

    ```
    uv run --locked --all-packages python notebooks/manuscript_figures.py
    ```
    """)
    return


@app.function
def size(figure: str, name: str) -> tuple[float, float]:
    """A panel's printed width and height (mm), from the layout."""
    spec = LAYOUT[figure]["panels"][name]
    return spec["width_mm"], spec["height_mm"]


@app.cell
def example_clip():
    # A typical new-animal clip of the single network (runs.SINGLE), picked by hand
    # (picks.EXAMPLE_CLIP).
    stem = picks.EXAMPLE_CLIP
    truth = runs.truth(stem)
    zephyr = runs.prediction(runs.SINGLE, stem)
    origin = float(zephyr["times"][0])
    # The excerpt of it shown in Fig. 1e: start (s from the clip's start) and length.
    excerpt = picks.EXCERPT
    return excerpt, origin, stem, truth, zephyr


@app.cell
def fig1bc_md():
    mo.md("""
    ## Fig. 1b, c: sniffing in the task

    One session of a new animal. b: the thermistor's breathing rate over the whole
    session (its events found as the corpus finds them) and speed around odor onset,
    apart for sites where the animal stopped and was rewarded, stopped and was not
    (unrewarded), or ran past (skipped: the task saw no stop), and breathing rate
    against speed; c: a minute of one clip, predicted by the fold that holds the animal
    out, with licks, rewards and odor sites. The behaviour comes from the session's processed
    asset (the same clock as the thermistor).
    """)
    return


@app.function
def task_window(
    sites: pd.DataFrame, t0: float, t1: float, window: float
) -> tuple[tuple[int, int], float]:
    """``(score, start)`` of the *window* in ``[t0, t1]`` holding the most kinds of
    whole odor sites, then the most sites (the earliest such, of equals)."""

    def score(start: float) -> tuple[int, int]:
        whole = sites[(sites["odor"] >= start) & (sites["leave"] <= start + window)]
        return whole["kind"].nunique(), len(whole)

    start = max(np.arange(t0, t1 - window, 0.5), key=score)
    return score(start), float(start)


@app.cell
def fig1bc_data():
    # The session of picks.TASK_ASSET (the task analysis): its behaviour, its two clips
    # predicted by the fold that holds the animal out, and its whole thermistor (its
    # events found as the corpus finds them, on a 60-Hz grid).
    _task = runs.task()
    lick_times = _task["licks"]["time"].to_numpy()
    odor_sites = _task["sites"]
    task_clips = [_task["clip1"], _task["clip2"]]
    task_truth = [_task["truth1"], _task["truth2"]]
    task_session = _task["session"].drop(columns="thermistor_hz")
    task_thermistor_hz = float(_task["session"]["thermistor_hz"].iloc[0])
    task_aligned = _task["aligned"]
    return (
        lick_times,
        odor_sites,
        task_aligned,
        task_clips,
        task_session,
        task_thermistor_hz,
        task_truth,
    )


@app.cell
def fig1b(lick_times, odor_sites, task_clips, task_truth):
    _window = 60.0
    # the clip whose best minute holds the most kinds of site, then the most sites;
    # its second half is shown
    _found = []
    for _c, _t in zip(task_clips, task_truth, strict=True):
        _time = _c["time"]
        _found.append(
            (*task_window(odor_sites, _time.iloc[0], _time.iloc[-1], _window), _c, _t)
        )
    _, _start, _clip, _truth = max(_found, key=lambda found: found[0])
    _start, _window = _start + _window / 2, _window / 2
    _ex = _clip[(_clip["time"] >= _start) & (_clip["time"] <= _start + _window)]
    _sites = odor_sites[
        (odor_sites["leave"] > _start) & (odor_sites["odor"] < _start + _window)
    ]
    fig1b = plots.task_stretch_panel(
        *size("fig1", "b"),
        _ex.assign(time=_ex["time"] - _start, rate=_ex["thermistor"]),
        _truth.assign(time=_truth["time"] - _start),
        _sites.assign(odor=_sites["odor"] - _start, leave=_sites["leave"] - _start),
        lick_times - _start,
        odor_sites["reward"].dropna().to_numpy() - _start,
    )
    plots.save_panel(fig1b, FIGURES / "fig1" / "b.svg")
    fig1b
    return


@app.cell
def fig1c(task_aligned, task_session, task_thermistor_hz):
    fig1c = plots.rate_speed_panel(*size("fig1", "c"), task_session, task_thermistor_hz)
    plots.save_panel(fig1c, FIGURES / "fig1" / "c.svg")
    _counts = task_aligned.groupby(["site", "signal"])["n"].first()
    v_fig1 = {
        f"sites_{s.lower()}": values.count(_counts[(s, "rate")]) for s in style.SITES
    }
    mo.vstack([fig1c, mo.ui.table(_counts.reset_index(), selection=None)])
    return (v_fig1,)


@app.cell
def fig1d_md():
    mo.md("""
    ## Fig. 1d: the network

    Read off the loaded checkpoint, so the layer sizes are the model's own; the input
    is three consecutive crops of the same clip.
    """)
    return


@app.cell
def fig1d(stem):
    # Three consecutive crops of the example clip at its middle inhale (the truth
    # analysis), through the checkpoint's own layers.
    _net = runs.network(runs.SINGLE)[0]
    _svg = architecture.network_svg(
        _net,
        runs.inhale_frames(stem),
        *size("fig1", "d"),
        runs.output_fs(stem),
    )
    (FIGURES / "fig1" / "d.svg").write_text(_svg, encoding="utf8")
    mo.Html(_svg)
    return


@app.cell
def fig1e_md():
    mo.md("""
    ## Fig. 1e: an example prediction

    The clip's face frame with the crop the network sees, then four seconds of the
    same test-A clip: the network's trace
    over the thermistor and its onset probability, with the thermistor's inhales
    (black) and exhales (grey) and the head's inhales (blue).
    """)
    return


@app.cell
def fig1e(excerpt, origin, stem, truth, zephyr):
    _t0, _window = excerpt
    fig1e = plots.prediction_panel(
        *size("fig1", "e"),
        truth,
        zephyr,
        origin,
        _t0,
        _window,
        runs.thumbnail(stem),
        runs.box(stem),
    )
    plots.save_panel(fig1e, FIGURES / "fig1" / "e.svg")
    fig1e
    return


@app.cell
def fig2_md():
    mo.md("""
    ## Fig. 2: the corpus

    From the corpus's own tables (the corpus analysis) and its spec: the selection
    flow, the breathing-rate mix and the split.
    """)
    return


@app.cell
def fig2_data():
    dataset = runs.corpus_root()
    spec = DatasetSpec.model_validate(
        tomllib.loads(
            (
                runs.REPO / "packages/zephyr-dataset/configs/vr-foraging-sniff.toml"
            ).read_text(encoding="utf8")
        )
    )
    folds = pd.read_parquet(dataset / "raw" / "folds.parquet")
    return dataset, folds, spec


@app.cell
def fig2a(dataset, spec):
    # From the dataset version the corpus starts at, the hand-picked ephys sessions
    # joining it there.
    _steps = corpus_flow.shorten(corpus_flow.flow(dataset, spec)[1:], sets=False)
    _steps[0]["name"] = "recorded"
    fig2a = corpus_flow.flow_panel(*size("fig2", "a"), _steps)
    plots.save_panel(fig2a, FIGURES / "fig2" / "a.svg")
    # recorded, passed the thermistor rules, passed the clip and camera rules, corpus
    v_selection = {
        k: values.count(s["sessions"])
        for k, s in zip(
            ("recorded", "thermistor", "clips", "corpus"), _steps, strict=True
        )
    }
    fig2a
    return (v_selection,)


@app.cell
def optional_cameras(dataset, folds, spec):
    # Videos of the optional cameras missing from corpus sessions, which keep their
    # required cameras: never recorded in the session, or left out on review (the
    # camera not on the animal).
    _corpus = set(folds.loc[folds["set"].notna(), "session_id"])
    _cameras = pd.read_parquet(dataset / "raw" / "cameras.parquet")
    _clips = pd.read_parquet(dataset / "raw" / "clips.parquet")
    _dropped = spec.dropped_cameras()
    v_cameras_missing = {}
    for _camera in spec.source.optional_cameras:
        _recorded = set(_cameras.loc[_cameras["camera"] == _camera, "session_id"])
        _cut = set(_clips.loc[_clips["camera"] == _camera, "session_id"])
        _reviewed = {s for s in _corpus & _recorded if (s, _camera) in _dropped}
        assert (_corpus & _recorded) - _cut == _reviewed, _camera
        _key = _camera.removesuffix("Camera").lower()
        v_cameras_missing[f"{_key}_not_recorded"] = values.count(
            len(_corpus - _recorded)
        )
        v_cameras_missing[f"{_key}_reviewed"] = values.count(len(_reviewed))
    v_cameras_missing
    return (v_cameras_missing,)


@app.cell
def fig2b(folds):
    # Every thermistor breath's rate over the corpus, and per animal.
    _rates = runs.animal_breath_rates()
    fig2b = plots.corpus_rate_panel(*size("fig2", "b"), _rates, step=0.5)
    plots.save_panel(fig2b, FIGURES / "fig2" / "b.svg")
    _r = np.concatenate([np.asarray(r) for r in _rates.values()])
    _r = _r[(_r >= 2) & (_r <= 15)]
    v_rates = {
        "breaths": values.count(len(_r)),
        "animals": values.count(len(_rates)),
        "slow": values.pct((_r < 4).mean()),
        "medium": values.pct(((_r >= 4) & (_r < 8)).mean()),
        "fast": values.pct((_r >= 8).mean()),
    }
    fig2b
    return (v_rates,)


@app.cell
def fig2c(folds):
    _used = folds[folds["set"].notna()]
    fig2c = plots.corpus_animals_panel(*size("fig2", "c"), _used)
    plots.save_panel(fig2c, FIGURES / "fig2" / "c.svg")
    _dates = pd.to_datetime(_used["date"])
    _per = (
        _used.assign(date=_dates)
        .groupby("subject_id")
        .agg(
            sessions=("session_id", "nunique"),
            days=("date", lambda d: (d.max() - d.min()).days),
            rigs=("rig", "nunique"),
        )
    )
    _clip_s = 600  # two 300-s clips per session
    v_corpus = {
        "sessions": values.count(_used["session_id"].nunique()),
        "animals": values.count(_used["subject_id"].nunique()),
        "rigs": values.count(_used["rig"].nunique()),
        "months": values.count(_used["month"].nunique()),
        "hours": values.count(_used["session_id"].nunique() * _clip_s / 3600),
        "median_sessions": f"{_per['sessions'].median():g}",
        "median_days": f"{_per['days'].median():g}",
        "multi_rig": values.count((_per["rigs"] > 1).sum()),
    }
    fig2c
    return (v_corpus,)


@app.cell
def fig3_md():
    mo.md("""
    ## Fig. 3: the benchmark

    Every method on test set A (new animals), as each run scored itself, and the
    clip-by-clip tables (onset lags, rate windows) for the networks.
    """)
    return


@app.cell
def fig3_methods():
    order = [
        "Pixel (motion)",
        "Pixel (SNR)",
        "Facemap",
        "TS-CAN",
        "RhythmFormer",
        "PhysFormer",
        "PhysNet",
        "Zephyr",
    ]
    methods = runs.available(order)
    # The unit of comparison is an animal on one rig (Methods, Statistics): every
    # score is averaged within a unit, and units are what is counted and resampled.
    _folds = pd.read_parquet(runs.dataset_raw("folds.parquet"))
    _rig = dict(zip(_folds["session_id"], _folds["rig"], strict=True))
    _subject = dict(zip(_folds["session_id"], _folds["subject_id"], strict=True))

    def with_unit(table: pd.DataFrame) -> pd.DataFrame:
        """*table* (a ``session`` and ``subject`` per row) with its ``rig`` and
        ``unit``."""
        rig = table["session"].map(_rig)
        return table.assign(rig=rig, unit=table["subject"].astype(str) + "@" + rig)

    def rate_share(breaths: pd.DataFrame) -> pd.DataFrame:
        """:func:`analysis.rate_share` of one method's *breaths* (a ``stem`` per row),
        over animal x rig units; the animal from the session metadata."""
        session = breaths["stem"].astype(str).str.rsplit("_span", n=1).str[0]
        units = with_unit(
            breaths.assign(session=session, subject=session.map(_subject))
        )
        return analysis.rate_share(units, bins=analysis.RATE_BINS_HZ[1:])

    test_a = runs.SETS["test_a"]
    return methods, rate_share, test_a, with_unit


@app.cell
def fig3_folds(methods, with_unit):
    # Every summary is over the five cross-validation folds (Methods, Statistics):
    # every method was retrained on each.
    cv_folds = runs.RESULTS.cross_validation.folds
    _cv = with_unit(runs.cv_scores("per_recording"))
    cv_methods = [m for m in methods if m in set(_cv["method"])]
    by_fold = _cv[_cv["method"].isin(cv_methods)].reset_index(drop=True)

    def fold_tables(names, kinds=("breaths", "false_positives", "lags", "windows")):
        """The analysis tables of each method on each fold (the tables analysis),
        labelled by the method and a ``fold`` column; a run not finished is left out,
        as its scores are (``runs.cv_scores``)."""
        out = {k: [] for k in kinds}
        for n in names:
            for f in cv_folds:
                if runs.evaluation(runs.cv_name(n, f)) is None:
                    continue
                t = runs.tables(runs.cv_name(n, f))
                for k in kinds:
                    out[k].append(t[k].assign(method=n, fold=f))
        return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}

    return by_fold, cv_methods, fold_tables


@app.cell
def fig3a(cv_methods, methods):
    # One clip of an animal held out by fold cv_0 and an excerpt of it, picked by hand
    # (picks.STACK_CLIP, picks.STACK_EXCERPT).
    _fold = runs.RUNS[runs.SINGLE].fold
    _names = {n: runs.cv_name(n, _fold) for n in methods if n in cv_methods}
    _stem = picks.STACK_CLIP
    _truth = runs.truth(_stem)
    _preds = {n: runs.prediction(c, _stem) for n, c in _names.items()}
    _origin = float(_preds["Zephyr"]["times"][0])
    _t0, _window = picks.STACK_EXCERPT
    fig3a = plots.trace_stack_panel(
        *size("fig3", "a"), _preds, _truth, _origin, _t0, _window
    )
    plots.save_panel(fig3a, FIGURES / "fig3" / "a.svg")
    mo.vstack([mo.md(f"{_stem}, from {_t0:.1f} s"), fig3a])
    return


@app.cell
def fig3b(by_fold, methods, test_a):
    # Inference speed, frames per second (runs.wall_time): networks on this machine's GPU,
    # the pixel methods and Facemap in numpy on its CPU. Inhale F1 on new animals per
    # fold (dots) and its mean over folds with the 95% interval.
    timing = runs.wall_time(methods).set_index("method")
    on_cpu = [m for m in methods if not timing.loc[m, "device"].startswith("NVIDIA")]
    _a = by_fold[by_fold["set"] == test_a]
    fig3b = plots.accuracy_time_panel(
        *size("fig3", "b"),
        analysis.per_fold(_a, "inhale", by=["method"]),
        analysis.fold_scores(_a, "inhale", by=["method"]),
        timing["frames per s"],
        methods,
        on_cpu=on_cpu,
        label_offsets={
            "Zephyr": (-4, 6),
            "TS-CAN": (3, -7),
            "PhysNet": (4, 3),
            "Pixel (SNR)": (-12, -7),
        },
    )
    plots.save_panel(fig3b, FIGURES / "fig3" / "b.svg")
    fig3b
    return on_cpu, timing


@app.cell
def fig3cde(fold_tables, rate_share, test_a):
    # The networks' clip-by-clip tables on new animals, per fold. c: F1 by rate, mean
    # over folds with its interval; d, e: every fold's new-animal breaths and windows
    # pooled (each animal is new in one fold only).
    nets = ["TS-CAN", "RhythmFormer", "PhysFormer", "PhysNet", "Zephyr"]
    whole = {k: v.query("set == @test_a") for k, v in fold_tables(nets).items()}
    _by_rate = analysis.f1_by_rate(
        whole["breaths"], whole["false_positives"], analysis.RATE_BINS_HZ[1:]
    )
    _rate = analysis.across_folds(_by_rate, "f1", by=["method", "bin"]).merge(
        _by_rate.groupby(["method", "bin"], observed=True)["share"]
        .mean()
        .reset_index(),
        on=["method", "bin"],
    )
    _cv_breaths = whole["breaths"]
    fig3c = plots.rate_band_panel(
        *size("fig3", "c"),
        _rate,
        nets[::-1],
        rate_share(_cv_breaths[_cv_breaths["method"] == "Zephyr"]),
        legend=False,  # colours as in b, where every method is named
    )
    plots.save_panel(fig3c, FIGURES / "fig3" / "c.svg")
    _cv = whole
    _zephyr = _cv["lags"][_cv["lags"]["method"] == "Zephyr"]
    fig3d = plots.lag_rate_panel(*size("fig3", "d"), _zephyr)
    plots.save_panel(fig3d, FIGURES / "fig3" / "d.svg")
    fig3e = plots.bland_altman_panel(*size("fig3", "e"), _cv["windows"], "Zephyr")
    plots.save_panel(fig3e, FIGURES / "fig3" / "e.svg")

    _f1 = _rate.set_index(["method", "bin"])["mean"]
    _breaths = _cv["breaths"][_cv["breaths"]["method"] == "Zephyr"]
    _in_band = _breaths["rate_hz"].between(2, 15, inclusive="left")
    _ms = _zephyr.assign(ms=1000 * _zephyr["lag_s"], method="Zephyr")

    def _lag_ci(lags):
        """The mean lag of each fold, then their mean and 95% bootstrap interval over folds."""
        per_fold = lags.groupby(["method", "fold"], as_index=False)["ms"].mean()
        return analysis.across_folds(per_fold, "ms").iloc[0]

    _lag = _lag_ci(_ms)

    def _lag_at(low: float, high: float) -> str:
        return values.signed(
            _lag_ci(_ms[_ms["rate_hz"].between(low, high, "left")])["mean"]
        )

    _w = _cv["windows"][_cv["windows"]["method"] == "Zephyr"]
    _d = 60 * (_w["pred_hz"] - _w["true_hz"])  # breaths per minute
    _clips = runs.cv_scores("clips").query("method == 'Zephyr' and set == @test_a")
    _middle = _f1.loc["Zephyr"].loc[["6-8", "8-10", "10-12"]]
    v_rate = {
        "f1_slow": values.num(_f1.loc[("Zephyr", "2-3")]),
        "f1_middle": values.span(_middle.min(), _middle.max()),
        "f1_fast": values.num(_f1.loc[("Zephyr", "12-15")]),
        "physnet_f1_slow": values.num(_f1.loc[("PhysNet", "2-3")]),
        "physnet_f1_fast": values.num(_f1.loc[("PhysNet", "12-15")]),
        "physformer_f1_slow": values.num(_f1.loc[("PhysFormer", "2-3")]),
        "breaths": values.count(_in_band.sum()),
        "lags": values.count(len(_zephyr)),
        "lag": values.num(_lag["mean"], 1),
        "lag_ci": values.span(_lag["low"], _lag["high"], 1),
        "lag_slow": _lag_at(2, 3),
        "lag_fast": _lag_at(11, 12),
        "windows": values.count(len(_w)),
        "mae": values.num(_d.abs().mean(), 1),
        "bias": values.signed(_d.mean()),
        "limits_low": values.signed(_d.mean() - 1.96 * _d.std()),
        "limits_high": values.signed(_d.mean() + 1.96 * _d.std()),
        # the error relative to the thermistor's rate, and the windows within 10% of it
        "mape": values.num((_d.abs() / (60 * _w["true_hz"])).mean() * 100, 1),
        "within_10": values.pct((_d.abs() / (60 * _w["true_hz"]) <= 0.10).mean()),
        "clips": values.count(len(_clips)),
        "sessions": values.count(_clips["session"].nunique()),
        "animals": values.count(_clips["subject"].nunique()),
    }
    mo.vstack([fig3c, fig3d, fig3e, mo.ui.table(_rate.round(3), selection=None)])
    return nets, v_rate, whole


@app.cell
def figs5(fold_tables, test_a):
    # Supplementary Fig. 5: why slow breaths score lower. Zephyr's breaths on new
    # animals per rate bin and fold: found within 25 ms, 50 ms and 10% of the breath's
    # cycle; the spread of its lags in ms and as a share of the cycle; breaths found
    # late (beyond 25 ms) against not found. Mean over folds with its interval.
    _t = fold_tables(["Zephyr"], ("breaths", "lags"))
    _t = {k: v.query("set == @test_a") for k, v in _t.items()}
    _timing = analysis.timing_by_rate(
        _t["lags"], _t["breaths"], analysis.RATE_BINS_HZ[1:]
    )
    _measures = [
        "within 25 ms",
        "within 50 ms",
        "within cycle",
        "iqr_ms",
        "iqr_cycle",
        "near",
        "absent",
    ]
    slow_timing = pd.concat(
        analysis.across_folds(
            _timing.rename(columns={m: "value"}), "value", by=["method", "bin"]
        ).assign(measure=m)
        for m in _measures
    )
    _panels = {
        "a": (
            {
                "within 25 ms": "Within 25 ms",
                "within 50 ms": "Within 50 ms",
                "within cycle": "Within 10% of cycle",
            },
            "Breaths found (%)",
            100.0,
        ),
        "b": ({"iqr_ms": "Timing spread"}, "Timing spread (ms)", 1.0),
        "c": ({"iqr_cycle": "Timing spread"}, "Timing spread (% of cycle)", 1.0),
        "d": (
            {"near": "Found 25–100 ms off", "absent": "Not found"},
            "Breaths (%)",
            100.0,
        ),
    }
    _figs = []
    for _panel, (_m, _label, _scale) in _panels.items():
        _fig = plots.rate_lines_panel(
            *size("figs5", _panel),
            slow_timing[slow_timing["measure"].isin(list(_m))],
            _m,
            _label,
            scale=_scale,
        )
        plots.save_panel(_fig, FIGURES / "figs5" / f"{_panel}.svg")
        _figs.append(_fig)
    # e: inhalations the corpus detector adds, per breath of its labels, when its
    # prominence fraction is lowered from 0.2, on the thermistor of every held-out clip
    # (the prominence analysis).
    _added = runs.label_added()
    _settings = {f: f"{f:g}" for f in runs.PROMINENCE}
    label_added = analysis.across_folds(
        analysis.added_by_rate(_added, analysis.RATE_BINS_HZ[1:]),
        "added",
        by=["setting", "bin"],
    ).rename(columns={"setting": "measure"})
    _fig = plots.rate_lines_panel(
        *size("figs5", "e"),
        label_added,
        {v: f"Prominence {v}" for v in _settings.values()},
        "Inhalations added (%)",
        scale=100.0,
    )
    plots.save_panel(_fig, FIGURES / "figs5" / "e.svg")
    _figs.append(_fig)
    _gain = label_added.set_index(["measure", "bin"])["mean"]
    _at = slow_timing.set_index(["measure", "bin"])["mean"]
    v_slow = {
        "added_slow": values.pct(_gain[("0.1", "2-3")], 1),
        "added_mid": values.pct(_gain[("0.1", "6-8")], 1),
        "added_slow_05": values.pct(_gain[("0.05", "2-3")], 1),
        "within_25_slow": values.pct(_at[("within 25 ms", "2-3")]),
        "within_50_slow": values.pct(_at[("within 50 ms", "2-3")]),
        "within_cycle_slow": values.pct(_at[("within cycle", "2-3")]),
        "within_cycle_fast": values.pct(_at[("within cycle", "10-12")]),
        "iqr_ms_slow": values.num(_at[("iqr_ms", "2-3")], 0),
        "iqr_ms_mid": values.num(_at[("iqr_ms", "6-8")], 0),
        "iqr_cycle_slow": values.num(_at[("iqr_cycle", "2-3")], 0),
        "iqr_cycle_mid": values.num(_at[("iqr_cycle", "6-8")], 0),
        "near_slow": values.pct(_at[("near", "2-3")]),
        "absent_slow": values.pct(_at[("absent", "2-3")], 1),
    }
    mo.vstack([mo.hstack(_figs), mo.md(f"{v_slow}")])
    return (v_slow,)


@app.cell
def saliency_data():
    # Supplementary Fig. 6 (the saliency analysis). The face network of each fold on
    # its new animals: for every unit, the clip nearest the unit's median correlation,
    # two 8.5-s windows (analysis.clip_saliency). The side and front networks (fold
    # cv_0) on the same spans, their clips each run's own held-out list.
    saliency = runs.saliency()
    return (saliency,)


@app.cell
def figs6(saliency):
    # Supplementary Fig. 6: a: one new animal of fold cv_0 from each camera, the
    # occlusion map in its crop box (a unit whose maps are close to the average on
    # every camera, with a clean side view); b: maps averaged over the units of each
    # camera, over their mean crop; c: how each inhale's gradient spreads over its
    # frames.
    _session = picks.SALIENCY_SESSION

    def _unit_of(found):
        same = [
            k
            for k, s in enumerate(found["stem"])
            if str(s).rsplit("_span", 1)[0] == _session
        ]
        return same[0] if same else 0

    _views = {}
    for _camera, _found in saliency.items():
        _i = _unit_of(_found)
        _stem = str(_found["stem"][_i])
        if "frame" in _found:
            _image, _box = _found["frame"][_i], tuple(_found["box"][_i])
        else:
            _image, _box = runs.thumbnail(_stem), runs.box(_stem)
        _views[_camera] = (_image, _box, _found["occlusion"][_i])
    fig_s6a = plots.saliency_frames_panel(*size("figs6", "a"), _views)
    plots.save_panel(fig_s6a, FIGURES / "figs6" / "a.svg")
    _kinds = {
        "Occlusion": "occlusion",
        "Gradient": "gradient",
        "Random weights": "random",
    }
    fig_s6b = plots.saliency_grid_panel(
        *size("figs6", "b"),
        {c: np.nanmean(f["crop"], axis=0) for c, f in saliency.items()},
        {
            k: {c: np.nanmean(f[key], axis=0) for c, f in saliency.items()}
            for k, key in _kinds.items()
        },
    )
    plots.save_panel(fig_s6b, FIGURES / "figs6" / "b.svg")
    _profiles, _within = {}, {}
    for _camera, _found in saliency.items():
        _long = pd.DataFrame(
            {
                "fold": np.repeat(_found["fold"], _found["profile"].shape[1]),
                "offset_s": np.tile(_found["offset_s"], len(_found["fold"])),
                "share": 100 * _found["profile"].ravel(),
            }
        )
        _per_fold = _long.groupby(["fold", "offset_s"], as_index=False)["share"].mean()
        _profiles[_camera] = analysis.across_folds(_per_fold, "share", by=["offset_s"])
        _near = _per_fold[_per_fold["offset_s"].abs() <= 0.1]
        _sum = _near.groupby("fold", as_index=False)["share"].sum().assign(all="all")
        _within[_camera] = analysis.across_folds(_sum, "share", by=["all"]).iloc[0]
    fig_s6c = plots.attribution_profile_panel(*size("figs6", "c"), _profiles)
    plots.save_panel(fig_s6c, FIGURES / "figs6" / "c.svg")
    _face = _within["Face camera"]
    v_saliency = {
        "units": values.count(len(saliency["Face camera"]["unit"])),
        "within_100": values.num(_face["mean"], 0),
        "within_100_ci": values.span(_face["low"], _face["high"], 0),
        **{
            f"within_100_{c.split()[0].lower()}": values.num(w["mean"], 0)
            for c, w in _within.items()
            if c != "Face camera"
        },
        **{
            f"units_{c.split()[0].lower()}": values.count(len(f["unit"]))
            for c, f in saliency.items()
            if c != "Face camera"
        },
    }
    mo.vstack([fig_s6a, fig_s6b, fig_s6c, mo.md(f"{v_saliency}")])
    return (v_saliency,)


@app.cell
def figs7(with_unit):
    # Supplementary Fig. 7: the all-camera network on a different behavioural task (dynamic
    # foraging, zephyr-df-dataset), scored on each left-out animal before and after it
    # was refined on the other two (one run per animal, pooled). a: the experiment's two
    # views of a left-out animal with their crops; b: a few seconds of one of its clips
    # from both views, the thermistor and the refined network; c: each session's scores
    # before and after; d: what refining cost on the VR test sets of cv_0, after minus
    # before per animal x rig unit (the three refined networks pooled).
    _parent = runs.cv_name("Zephyr, all cameras", "cv_0")
    _on_df, _refined = (
        runs.pooled("Zephyr, all cameras, on DF"),
        runs.pooled("Zephyr, all cameras, refined on DF"),
    )
    _cameras = {"side_camera_right": "Side view", "bottom_camera": "Bottom view"}
    _before = pd.concat([runs.clip_scores(n) for n in _on_df], ignore_index=True)
    _after = pd.concat([runs.clip_scores(n) for n in _refined], ignore_index=True)
    _df = _after[_after["set"] == "df_test_a"]
    _metrics = ["correlation", "inhale"]

    # c: each session and camera, the mean of its clips before and after.
    _table = pd.concat(
        pd.DataFrame(
            {
                "Before": _before.groupby(["camera", "session"])[m].mean(),
                "After": _df.groupby(["camera", "session"])[m].mean(),
            }
        )
        .reset_index()
        .assign(metric=m, camera=lambda d: d["camera"].map(_cameras))
        for m in _metrics
    )
    fig_s7c = plots.before_after_panel(
        *size("fig4", "i"), _table, _metrics, list(_cameras.values())
    )
    plots.save_panel(fig_s7c, FIGURES / "fig4" / "i.svg")

    # d: the VR test sets, every camera, after minus before per unit (paired
    # bootstrap); the animal from the corpus's session metadata.
    _folds = pd.read_parquet(runs.dataset_raw("folds.parquet"))
    _subject = dict(
        zip(_folds["session_id"], _folds["subject_id"].astype(str), strict=True)
    )
    _vr_sets = ["test_a", "test_b"]
    _vr = pd.concat(
        [
            runs.clip_scores(_parent).assign(method="Before"),
            _after.assign(method="After"),
        ],
        ignore_index=True,
    )
    _vr = _vr[_vr["set"].isin(_vr_sets)]
    _vr = with_unit(_vr.assign(subject=_vr["session"].map(_subject)))
    _cost = []
    for _m in _metrics:
        _b = analysis.bootstrap(_vr, _m, reference="Before", unit="unit")
        _b = _b[_b["method"] == "After"]
        _cost.append(
            pd.DataFrame(
                {
                    "metric": _m,
                    "set": _b["set"],
                    "mean": -_b["Before minus"],
                    "low": -_b["diff high"],
                    "high": -_b["diff low"],
                    "units": _b["units"],
                }
            )
        )
    refine_cost = pd.concat(_cost, ignore_index=True)
    _set_names = {
        "test_a": "A: new animals",
        "test_b": "B: new sessions",
    }
    fig_s7d = plots.cost_panel(
        *size("figs7", "a"),
        refine_cost.assign(set=refine_cost["set"].map(_set_names)),
        _metrics,
        list(_set_names.values()),
    )
    plots.save_panel(fig_s7d, FIGURES / "figs7" / "a.svg")

    # a, b: the session whose refined correlation, averaged over both cameras, is
    # nearest the median of the held-out animal's sessions; its first span.
    _by_session = _df.groupby("session")["correlation"].mean()
    _session = (_by_session - _by_session.median()).abs().idxmin()
    _views, _rows = {}, {}
    for _camera, _label in _cameras.items():
        _stem = f"{_session}_span1_{_camera}"
        _views[_label] = (runs.thumbnail(_stem), runs.box(_stem))
        _truth = runs.truth(_stem)
        _p = runs.prediction(runs.run_of(_refined, _stem), _stem)
        _times = _p["times"]
        _n = len(_times)
        _rows[_label] = {
            "times": _times,
            "truth": _truth["signal"][:_n],
            "truth_inhale": _truth["inhale"],
            "signal": _p["signal"],
            "head": _p["head"],
            # 12 s around the clip's middle, in s from its start
            "start": round(float(_times[_n // 2] - _times[0]) - 6.0),
        }
    fig_s7a = plots.views_panel(*size("fig4", "g"), _views)
    plots.save_panel(fig_s7a, FIGURES / "fig4" / "g.svg")
    fig_s7b = plots.refined_traces_panel(*size("fig4", "h"), _rows)
    plots.save_panel(fig_s7b, FIGURES / "fig4" / "h.svg")

    _cam = _df.groupby("camera")[_metrics].mean()
    _bef = _before.groupby("camera")[_metrics].mean()
    _c = refine_cost.set_index(["metric", "set"])
    v_refine = {
        "sessions": values.count(_df["session"].nunique()),
        "clips": values.count(len(_df)),
        **{
            f"{k}_{w}_{c.split('_')[0]}": values.num(t.loc[c, m])
            for k, m in (("f1", "inhale"), ("r", "correlation"))
            for w, t in (("after", _cam), ("before", _bef))
            for c in _cameras
        },
        **{
            f"cost_{k}_{s.split('_')[1]}": values.num(_c.loc[(m, s), "mean"], 3)
            for k, m in (("f1", "inhale"), ("r", "correlation"))
            for s in _vr_sets
        },
        **{
            f"cost_{k}_{s.split('_')[1]}_ci": values.span(
                _c.loc[(m, s), "low"], _c.loc[(m, s), "high"], 3
            )
            for k, m in (("f1", "inhale"), ("r", "correlation"))
            for s in _vr_sets
        },
        "threshold_after": values.span(
            min(map(runs.head_threshold, _refined)),
            max(map(runs.head_threshold, _refined)),
            1,
        ),
        "threshold_before": values.span(
            min(map(runs.head_threshold, _on_df)),
            max(map(runs.head_threshold, _on_df)),
            1,
        ),
    }
    mo.vstack(
        [
            mo.hstack([fig_s7a, fig_s7b]),
            mo.hstack([fig_s7c, fig_s7d]),
            mo.md(f"{v_refine}"),
            mo.ui.table(refine_cost.round(4), selection=None),
        ]
    )
    return (v_refine,)


@app.function
def application_task(session: str) -> dict:
    """One session without a thermistor on the grid of its prediction: ``grid``,
    ``speed``, ``rate`` (from its inhalations) and its odor sites (``onsets``: ``odor``,
    ``leave``, ``kind``, ``reward``), beside :func:`runs.application`'s own keys."""
    s = runs.application(session)
    grid = s["times"]
    sites = s["sites"]
    odor = sites[
        (sites["site_label"] == "RewardSite") & sites["odor_onset_time"].notna()
    ]
    onsets = pd.DataFrame(
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
    return s | {
        "grid": grid,
        "speed": np.interp(grid, s["speed"].index.to_numpy(), s["speed"].to_numpy()),
        "rate": analysis.breathing_rate(s["head"], grid),
        "onsets": onsets,
    }


@app.cell
def fig7(qc_windows):
    # Fig. 5: respiration in task sessions without a thermistor, from the face video
    # alone (the manifest's application sessions, predicted by the all-camera network
    # of cv_0). a: their quality-control scores over 30-s windows, pooled, beside sets
    # where the network works and known failures, with the lowest window on head vs
    # trace F1 marked; b: that window whole, with frames of its face video; c: 15 s of
    # the example session; d, e:
    # respiration rate and speed around odor onset per kind of site, the example and
    # the mean over sessions; f, g: respiration rate against speed, the same.
    _sessions = runs.RESULTS.application.session
    _entry = next(e for e in _sessions if e.example)
    _tasks = {e.session: application_task(e.session) for e in _sessions}
    _s = _tasks[_entry.session]
    _grid, _speed, _rate, _onsets = _s["grid"], _s["speed"], _s["rate"], _s["onsets"]
    _threshold = runs.head_threshold(
        runs.cv_name(runs.RESULTS.application.network, runs.RESULTS.application.fold)
    )

    # a, b: the scores, and what a low and a median window look like
    _label = next(q.label for q in runs.RESULTS.quality.set if q.role == "application")
    _apps = qc_windows[qc_windows["set"] == _label].assign(row=_label)
    _measure = next(iter(style.QC_MEASURES))
    # only the lowest window, shown whole with frames of the face video
    _marked = analysis.example_windows(_apps, _measure).iloc[:1]
    _marked = _marked.assign(
        mark=["i"], session=_marked["stem"].map(runs.application_session)
    )
    _windows = pd.concat(
        [
            qc_windows[~qc_windows["role"].isin(["application", "camera"])].assign(
                row=lambda d: d["set"]
            ),
            _apps,
        ]
    )
    _rows = [
        ("Held-out animals", "working"),
        ("Dynamic foraging, refined", "working"),
        (_label, "application"),
        ("Dynamic foraging, unrefined", "null"),
        ("Noise 32", "null"),
    ]
    fig_5a = plots.quality_sets_panel(
        *size("fig7", "a"), _windows, _rows, truth=None, marks=_marked, per_row=2
    )
    plots.save_panel(fig_5a, FIGURES / "fig7" / "a.svg")
    # Frames of the window, s from its start (frame i at i / the clip's frame rate, as
    # the prediction), chosen by eye (picks.APPLICATION_SHOTS_S; the application
    # analysis extracted them).
    _shots_s = list(picks.APPLICATION_SHOTS_S)
    _w = _marked.iloc[0]
    # the 10 s of the window centred on the frames, in s from the window's start
    _shown_s = 10.0
    _from = (_shots_s[0] + _shots_s[-1]) / 2 - _shown_s / 2
    _t = _tasks[_w.session]
    _offset = _t["video_time"] - _w.start
    _keep = (_offset >= _from) & (_offset <= _from + _shown_s)
    _times = _t["times"][_keep]
    _head = _t["head"]
    # the snippet's time axis: s from the window's start, on the Harp clock
    _zero = _times[0] - _offset[_keep][0]
    _frames, _box = runs.application_frames(
        _w.session, [_w.start + _s for _s in _shots_s]
    )
    _examples = [
        {
            "title": _w.session,
            "time": _times - _zero,
            "signal": _t["signal"][_keep],
            "onset": _t["onset"][_keep],
            "head": _head[(_head >= _times[0]) & (_head <= _times[-1])] - _zero,
            "frames": [
                {"time": _s, "image": _f}
                for _s, _f in zip(_shots_s, _frames, strict=True)
            ],
            "box": _box,
        }
    ]
    fig_5b = plots.window_examples_panel(
        *size("fig7", "b"), _examples, _threshold, style.color("Zephyr")
    )
    plots.save_panel(fig_5b, FIGURES / "fig7" / "b.svg")

    # c: a stretch of the example session
    _window = 15.0
    _, _start = task_window(_onsets, _grid[0], _grid[-1], _window)
    _keep = (_grid >= _start) & (_grid <= _start + _window)
    _signal = _s["signal"][_keep]
    _shown = _onsets[(_onsets["leave"] > _start) & (_onsets["odor"] < _start + _window)]
    fig_5c = plots.task_stretch_panel(
        *size("fig7", "c"),
        pd.DataFrame(
            {
                "time": _grid[_keep] - _start,
                "speed": _speed[_keep],
                "rate": _rate[_keep],
            }
        ),
        pd.DataFrame(
            {
                "time": _grid[_keep] - _start,
                "signal": (_signal - _signal.mean()) / _signal.std(),
            }
        ),
        _shown.assign(odor=_shown["odor"] - _start, leave=_shown["leave"] - _start),
        _s["licks"] - _start,
        _onsets["reward"].dropna().to_numpy() - _start,
        label="Respiration\n(z)",
        colour=style.color("Zephyr"),
        inhalations=_s["head"] - _start,
        speed_on_rate=True,
        onset=pd.DataFrame({"time": _grid[_keep] - _start, "p": _s["onset"][_keep]}),
    )
    plots.save_panel(fig_5c, FIGURES / "fig7" / "c.svg")

    # d-g: the task, in the example session and over every session
    _lags = np.arange(-2.0, 5.0 + 1e-9, 1 / 60)
    _bins = plots.rate_speed_bins(60.0)
    _aligned_all, _counts_all = {}, {}
    for _name, _t in _tasks.items():
        _aligned_all[_name] = analysis.odor_aligned(
            _t["grid"], {"rate": _t["rate"], "speed": _t["speed"]}, _t["onsets"], _lags
        ).assign(unit=_name)
        _counts_all[_name] = analysis.rate_speed_counts(
            pd.DataFrame({"speed": _t["speed"], "rate": _t["rate"]}), *_bins
        )
    _aligned = _aligned_all[_entry.session]
    _across = analysis.aligned_across_units(pd.concat(_aligned_all.values()))
    _one = f"Mouse {_s['subject']}"
    _every = f"{len({t['subject'] for t in _tasks.values()})} animals"
    fig_5d = plots.aligned_panel(
        *size("fig7", "d"), _aligned, title=_one, legend_columns=2
    )
    plots.save_panel(fig_5d, FIGURES / "fig7" / "d.svg")
    fig_5e = plots.aligned_panel(
        *size("fig7", "e"), _across, legend=False, title=_every
    )
    plots.save_panel(fig_5e, FIGURES / "fig7" / "e.svg")
    fig_5f = plots.share_heat_panel(
        *size("fig7", "f"),
        analysis.rate_given_speed(_counts_all[_entry.session]),
        *_bins,
        title=_one,
    )
    plots.save_panel(fig_5f, FIGURES / "fig7" / "f.svg")
    fig_5g = plots.share_heat_panel(
        *size("fig7", "g"),
        analysis.rate_given_speed_across(list(_counts_all.values())),
        *_bins,
        title=_every,
    )
    plots.save_panel(fig_5g, FIGURES / "fig7" / "g.svg")

    _n = _aligned[(_aligned["signal"] == "rate") & (_aligned["lag"] == _lags[0])]
    _n_all = _across[(_across["signal"] == "rate") & (_across["lag"] == _lags[0])]
    v_application = {
        "sessions": values.count(len(_sessions)),
        "animals": values.count(len({t["subject"] for t in _tasks.values()})),
        "animal": _s["subject"],
        "minutes": values.count((_grid[-1] - _grid[0]) / 60),
        "windows": values.count(len(_apps)),
        "shown_s": values.count(_shown_s),
        **{
            f"sites_{k.lower()}": values.count(n)
            for k, n in _n.set_index("kind")["n"].items()
        },
        **{
            f"all_sites_{k.lower()}": values.count(n)
            for k, n in _n_all.set_index("kind")["sites"].items()
        },
    }
    mo.vstack(
        [
            fig_5a,
            fig_5b,
            fig_5c,
            mo.hstack([fig_5d, fig_5e, fig_5f, fig_5g]),
            mo.ui.table(_marked[["mark", "stem", "start", *style.QC_MEASURES]]),
            mo.md(f"{v_application}"),
        ]
    )
    return (v_application,)


@app.cell
def figs8(qc_windows):
    # Supplementary Fig. 8: the sessions without a thermistor from every camera: each
    # camera's quality-control scores over 30-s windows, beside sets where the network
    # works and known failures.
    _cams = [q for q in runs.RESULTS.quality.set if q.role in ("application", "camera")]
    _camera = {q.label: q.camera for q in _cams}
    # the held-out animals split by the camera of their clips (the all-camera network
    # scores each)
    _held_by = {c.split()[0] + "Camera": c for c in style.CAMERAS if "camera" in c}
    _held_cam = qc_windows["stem"].astype(str).str.rsplit("_", n=1).str[-1]
    _held = (qc_windows["set"] == "Held-out animals").to_numpy()
    _row = qc_windows["set"].map(_camera).fillna(qc_windows["set"])
    _row[_held] = "Held-out animals, " + _held_cam[_held].map(_held_by).str.lower()
    _windows = qc_windows.assign(row=_row)
    _held_rows = [
        r
        for c in _held_by.values()
        if (r := f"Held-out animals, {c.lower()}") in set(_windows["row"])
    ]
    _rows = [
        *[(r, "working") for r in _held_rows],
        ("Dynamic foraging, refined", "working"),
        *[(q.camera, q.role) for q in _cams],
        ("Dynamic foraging, unrefined", "null"),
        ("Noise 32", "null"),
    ]
    _colours = {q.camera: style.CAMERAS[q.camera] for q in _cams}
    fig_s8a = plots.quality_sets_panel(
        *size("figs8", "a"), _windows, _rows, truth=None, colours=_colours
    )
    plots.save_panel(fig_s8a, FIGURES / "figs8" / "a.svg")
    fig_s8a
    return


@app.cell
def figs9(qc_windows):
    # Supplementary Fig. 9: every measure a quality-control report holds, over the
    # 30-s windows of the sets of Fig. 6e (rhythm measures in a, onset-head measures
    # in b, with the true inhale F1).
    from zephyr.quality_control.metrics import MEASURES as _MEASURES

    _sets = [
        s for s in runs.RESULTS.quality.set if s.role not in ("application", "camera")
    ]
    _rows = [("Validation", "validation"), *[(s.label, s.role) for s in _sets]]
    _family = {
        f: {k: label for k, (fam, label, _) in _MEASURES.items() if fam == f}
        for f in ("rhythm", "head")
    }
    fig_s9a = plots.quality_sets_panel(
        *size("figs9", "a"), qc_windows, _rows, truth=None, measures=_family["rhythm"]
    )
    plots.save_panel(fig_s9a, FIGURES / "figs9" / "a.svg")
    fig_s9b = plots.quality_sets_panel(
        *size("figs9", "b"), qc_windows, _rows, measures=_family["head"]
    )
    plots.save_panel(fig_s9b, FIGURES / "figs9" / "b.svg")
    mo.vstack([fig_s9a, fig_s9b])
    return


@app.cell
def table1(by_fold, cv_methods, fold_tables, methods, on_cpu, test_a, timing):
    # Table 1 (benchmark, new animals) and Supplementary Table (new sessions of known
    # animals, test set B): per method, each score averaged within each unit, then over
    # the units of a fold, then over folds (mean and 95% interval). Rate metrics from 10-s
    # windows. Written as CSV for main.typ.
    _windows_all = fold_tables(methods, ("windows",))["windows"]

    def _cell(row, digits):
        mean = 0.0 if abs(row["mean"]) < 0.5 * 10**-digits else row["mean"]
        if pd.isna(row["low"]):
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} ({row['low']:.{digits}f}–{row['high']:.{digits}f})"

    def _benchmark(set_name: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
        """The table's rows, each column's summary over folds, and the recordings."""
        recs = by_fold[by_fold["set"] == set_name]
        windows = _windows_all[_windows_all["set"] == set_name].copy()
        unit = recs.drop_duplicates(["fold", "session"]).set_index(["fold", "session"])[
            "unit"
        ]
        windows["unit"] = [
            unit.get((f, st.rsplit("_span", 1)[0]))
            for f, st in zip(windows["fold"], windows["stem"], strict=True)
        ]
        rates = analysis.rate_agreement(
            windows.dropna(subset=["unit"]), by=["method", "fold", "unit"]
        ).reset_index()
        rates[["MAE (Hz)", "RMSE (Hz)"]] *= 60  # breaths per minute
        columns = {
            "MAE (Hz)": ("MAE", 1, rates),
            "RMSE (Hz)": ("RMSE", 1, rates),
            "MAPE (%)": ("MAPE", 1, rates),
            "r": ("rate_r", 2, rates),
            "correlation": ("trace_r", 2, recs),
            "inhale": ("inhale_f1", 2, recs),
            "exhale_f1": ("exhale_f1", 2, recs),
        }
        summary = {
            c: analysis.fold_scores(t, c, by=["method"]).set_index("method")
            for c, (_, _, t) in columns.items()
        }
        rows = []
        for m in methods:
            row = {
                "method": plots.DISPLAY.get(m, m),
                "splits": str(int(summary["inhale"].loc[m, "folds"])),
                "seconds": f"{timing.loc[m, 's per clip (median)']:.2f}"
                + ("*" if m in on_cpu else ""),
            }
            for c, (name, digits, _) in columns.items():
                row[name] = _cell(summary[c].loc[m], digits).replace("-0.", "−0.")
            rows.append(row)
        return pd.DataFrame(rows), summary, recs

    benchmark, _summary, _a = _benchmark(test_a)
    benchmark_b, _summary_b, _b = _benchmark(runs.SETS["test_b"])
    (FIGURES.parent / "tables").mkdir(exist_ok=True)
    benchmark.to_csv(FIGURES.parent / "tables" / "benchmark.csv", index=False)
    benchmark_b.to_csv(FIGURES.parent / "tables" / "benchmark_b.csv", index=False)
    benchmark_diff = {
        c: _summary[c][["mean", "low", "high", "minus ref", "diff low", "diff high"]]
        for c in ("inhale", "correlation")
    }
    benchmark_b_diff = {
        c: _summary_b[c][["mean", "low", "high", "minus ref", "diff low", "diff high"]]
        for c in ("inhale", "correlation")
    }

    def _counts(recs: pd.DataFrame) -> dict:
        ref = recs[recs["method"] == runs.REFERENCE]
        return {
            "folds": int(ref["fold"].nunique()),
            "animals": int(ref["subject"].nunique()),
            "units": int(ref["unit"].nunique()),
        }

    cv_counts = {**_counts(_a), "cv methods": cv_methods}
    cv_counts_b = _counts(_b)

    def _ours_minus(summary: pd.DataFrame, other: str, digits: int) -> tuple[str, str]:
        """The reference minus *other* on the same resamples, and its interval."""
        row = summary.loc[other]
        return values.num(-row["minus ref"], digits), values.span(
            -row["diff high"], -row["diff low"], digits
        )

    _ref, _f1, _r = runs.REFERENCE, _summary["inhale"], _summary["correlation"]
    _pixels = ["Pixel (motion)", "Pixel (SNR)"]
    _others = ["TS-CAN", "RhythmFormer", "PhysFormer"]
    _dsp = analysis.fold_scores(_a, "inhale_f1", by=["method"]).set_index("method")
    v_fig3 = {
        "folds": values.count(cv_counts["folds"]),
        "units_a": values.count(cv_counts["units"]),
        "animals_a": values.count(cv_counts["animals"]),
        "units_b": values.count(cv_counts_b["units"]),
        "animals_b": values.count(cv_counts_b["animals"]),
        "motion_r": values.num(_r.loc["Pixel (motion)", "mean"]),
        "pixel_r_max": values.num(_r.loc[["Pixel (SNR)"], "mean"].max()),
        "pixel_f1": values.span(
            _f1.loc[_pixels, "mean"].min(), _f1.loc[_pixels, "mean"].max()
        ),
        "facemap_r": values.num(_r.loc["Facemap", "mean"]),
        "facemap_f1": values.num(_f1.loc["Facemap", "mean"]),
        "f1": values.num(_f1.loc[_ref, "mean"]),
        "f1_ci": values.span(_f1.loc[_ref, "low"], _f1.loc[_ref, "high"]),
        "physnet_f1": values.num(_f1.loc["PhysNet", "mean"]),
        "r": values.num(_r.loc[_ref, "mean"]),
        "r_ci": values.span(_r.loc[_ref, "low"], _r.loc[_ref, "high"]),
        "physnet_r": values.num(_r.loc["PhysNet", "mean"]),
        "others_r": values.span(
            _r.loc[_others, "mean"].min(), _r.loc[_others, "mean"].max()
        ),
        "dsp_f1": values.num(_dsp.loc[_ref, "mean"]),
        "exhale_f1": values.num(_summary["exhale_f1"].loc[_ref, "mean"]),
        "mae_units": values.num(_summary["MAE (Hz)"].loc[_ref, "mean"], 1),
        "seconds": values.num(timing.loc[_ref, "s per clip (median)"]),
        "physnet_seconds": values.num(timing.loc["PhysNet", "s per clip (median)"]),
        "fps": values.count(round(timing.loc[_ref, "frames per s"], -2)),
        "physnet_fps": values.count(round(timing.loc["PhysNet", "frames per s"], -2)),
        "transformer_fps": "–".join(
            values.count(round(v, -2))
            for v in timing.loc[["PhysFormer", "RhythmFormer"], "frames per s"].agg(
                ["min", "max"]
            )
        ),
        # how much higher Zephyr's frame rate is than PhysNet's and the transformers'
        "physnet_fps_gain": values.pct(
            timing.loc[_ref, "frames per s"] / timing.loc["PhysNet", "frames per s"] - 1
        ),
        "transformer_fps_gain": values.span(
            *(
                timing.loc[_ref, "frames per s"]
                / timing.loc[["PhysFormer", "RhythmFormer"], "frames per s"].agg(
                    ["max", "min"]
                )
                * 100
                - 100
            ),
            0,
        ),
        # the share of PhysNet's inference time Zephyr saves, and the transformers' time
        # as a multiple of Zephyr's
        "physnet_saved": values.pct(
            1
            - timing.loc[_ref, "s per clip (median)"]
            / timing.loc["PhysNet", "s per clip (median)"]
        ),
        "transformer_times": values.num(
            timing.loc[["PhysFormer", "RhythmFormer"], "s per clip (median)"].min()
            / timing.loc[_ref, "s per clip (median)"],
            0,
        ),
        "transformer_seconds": values.span(
            timing.loc[["PhysFormer", "RhythmFormer"], "s per clip (median)"].min(),
            timing.loc[["PhysFormer", "RhythmFormer"], "s per clip (median)"].max(),
            1,
        ),
        "r_b": values.num(_summary_b["correlation"].loc[_ref, "mean"]),
        "f1_b": values.num(_summary_b["inhale"].loc[_ref, "mean"]),
        "physnet_r_b": values.num(_summary_b["correlation"].loc["PhysNet", "mean"]),
    }
    v_fig3["physnet_f1_diff"], v_fig3["physnet_f1_diff_ci"] = _ours_minus(
        _f1, "PhysNet", 3
    )
    v_fig3["physnet_r_diff"], v_fig3["physnet_r_diff_ci"] = _ours_minus(
        _r, "PhysNet", 2
    )
    mo.vstack(
        [
            mo.md(f"Test A: {cv_counts}"),
            mo.ui.table(benchmark, selection=None),
            *(mo.ui.table(v.round(3).reset_index()) for v in benchmark_diff.values()),
            mo.md(f"Test B: {cv_counts_b}"),
            mo.ui.table(benchmark_b, selection=None),
            *(mo.ui.table(v.round(3).reset_index()) for v in benchmark_b_diff.values()),
        ]
    )
    return benchmark_b_diff, benchmark_diff, cv_counts, cv_counts_b, v_fig3


@app.cell
def fig4_md():
    mo.md("""
    ## Fig. 4: generalization and its limits

    New animals against new sessions and animal by animal; F1 as the onset tolerance is
    relaxed; a clip of near misses and the clip at the 10th percentile of F1 (e, the
    curve against training size, waits for its runs).
    """)
    return


@app.cell
def fig4ab(by_fold, test_a):
    # a: new animals against new sessions, mean over folds with its interval.
    # b: every new-animal unit of the five folds (each animal is new in one fold).
    _shown = ["Facemap", "TS-CAN", "RhythmFormer", "PhysFormer", "PhysNet", "Zephyr"]
    _sets = [test_a, runs.SETS["test_b"]]
    _ab = by_fold[by_fold["set"].isin(_sets)]
    fig4a = plots.sets_panel(
        *size("fig4", "a"),
        pd.concat(
            analysis.fold_scores(_ab, m, by=["set", "method"]).assign(metric=m)
            for m in ("correlation", "inhale")
        ),
        _shown,
        ["correlation", "inhale"],
        _sets,
        label_on="correlation",
    )
    plots.save_panel(fig4a, FIGURES / "fig4" / "a.svg")
    fig4b = plots.animals_panel(
        *size("fig4", "b"),
        by_fold,
        ["Zephyr"],
        "inhale",
        test_a,
        unit="unit",
        unit_label="Animal × rig",
        by_animal="subject",
    )
    plots.save_panel(fig4b, FIGURES / "fig4" / "b.svg")
    mo.hstack([fig4a, fig4b])
    return


@app.cell
def fig4e(whole):
    # e: F1 and the errors left (misses and false inhalations) as the tolerance is
    # relaxed, per fold, then over folds.
    _nets = ["Zephyr"]
    _mine = {k: v[v["method"].isin(_nets)] for k, v in whole.items()}
    _curve = analysis.tolerance_curve(
        _mine["lags"],
        _mine["breaths"],
        _mine["false_positives"],
        np.arange(0, 100.1, 2.5),
    )
    _by = ["method", "tolerance_ms"]
    tolerance = analysis.across_folds(_curve, "f1", by=_by).merge(
        _curve.groupby(_by)["errors"].mean().reset_index(), on=_by
    )
    # Misses and false inhalations apart, each over folds with its interval.
    for _k in ("misses", "false"):
        _apart = analysis.across_folds(_curve, _k, by=_by).rename(
            columns={"mean": _k, "low": f"{_k} low", "high": f"{_k} high"}
        )
        tolerance = tolerance.merge(_apart.drop(columns="folds"), on=_by)
    fig4c = plots.tolerance_panel(*size("fig6", "a"), tolerance, _nets)
    plots.save_panel(fig4c, FIGURES / "fig6" / "a.svg")
    _at = tolerance.set_index("tolerance_ms")
    v_tolerance = {
        "f1_12": values.num(_at.loc[12.5, "mean"]),
        "f1_25": values.num(_at.loc[25.0, "mean"]),
        "f1_50": values.num(_at.loc[50.0, "mean"]),
        "f1_100": values.num(_at.loc[100.0, "mean"]),
        "errors_25": values.pct(_at.loc[25.0, "errors"], 1),
        "errors_50": values.pct(_at.loc[50.0, "errors"], 1),
        "errors_100": values.pct(_at.loc[100.0, "errors"], 1),
        "breaths": values.count(len(_mine["breaths"])),
    }
    fig4c
    return tolerance, v_tolerance


@app.cell
def fig5a(test_a, with_unit):
    # Fig. 5a: Zephyr trained on fewer animals (all their sessions), and on fewer
    # sessions of all its animals (open), on the cross-validation fold those runs were
    # trained on, against Zephyr on that fold; scored per unit on its new animals (test
    # A), the same units for every run. Several draws of one size are drawn as dots,
    # the line through their mean.
    _size = runs.available(runs.LEARNING_CURVE)

    def _cv_fold(name):
        fold = runs.RUNS[name].fold or ""
        return next(
            (
                f
                for f in runs.RESULTS.cross_validation.folds
                if fold.startswith(f + "-")
            ),
            None,
        )

    _on_fold = [n for n in _size if _cv_fold(n)]
    _ref = runs.cv_name(runs.REFERENCE, _cv_fold(_on_fold[0]))
    _names = [_ref, *_on_fold]

    def _curve(names, reference):
        trained = {n: runs.training(n) for n in names}
        every = trained[reference]
        recs = with_unit(runs.scores(names, "per_recording"))
        rows = []
        for metric in ["inhale", "correlation"]:
            # One run at a time: each has its own known units.
            ci = pd.concat(
                analysis.bootstrap(
                    recs[(recs["set"] == test_a) & (recs["method"] == n)].assign(
                        set="New animals"
                    ),
                    metric,
                    reference=n,
                    unit="unit",
                )
                for n in names
            )
            for _, r in ci.iterrows():
                tr = trained[r["method"]]
                rows.append(
                    {
                        "group": "New animals",
                        "metric": metric,
                        "method": r["method"],
                        "animals": tr["subject"].nunique(),
                        "sessions": len(tr),
                        "full": len(tr)
                        == every["subject"].isin(set(tr["subject"])).sum(),
                        "units": r["units"],
                        **{k: r[k] for k in ("mean", "low", "high")},
                    }
                )
        return pd.DataFrame(rows), recs, trained

    learning, _recs, _trained = _curve(_names, _ref)
    # Draws: several runs of one size below the fold's own, each on all the sessions
    # its split gave its animals (so drawn as the curve's dots).
    _sizes = learning.groupby("method")["animals"].first()
    _repeated = _sizes.map(_sizes.value_counts()) > 1
    _draws = _sizes[_repeated & (_sizes < _sizes[_ref])].index
    learning.loc[learning["method"].isin(_draws), "full"] = True
    # The text's numbers, per condition: a condition is the runs of one size (its
    # draws), and each number is the mean over them. Runs are found by what they
    # trained on, not by their labels.
    _f1 = learning[learning["metric"] == "inhale"].set_index("method")
    _r = learning[learning["metric"] == "correlation"].set_index("method")
    # fewer animals, all their sessions: one condition per number of animals
    _sized = _f1[_f1["full"] & (_f1.index != _ref)]
    _counts = sorted(_sized["animals"].unique())
    _of = {n: sorted(_sized.index[_sized["animals"] == n]) for n in _counts}
    _fewest = _of[_counts[0]]  # the draws of the fewest animals
    _partial = sorted(_f1.index[~_f1["full"]])  # all animals, part of their sessions

    def _mean(table, names, column="mean"):
        return table.loc[names, column].mean()

    def _listing(xs, show):
        xs = list(xs)
        return ", ".join(show(x) for x in xs[:-1]) + " and " + show(xs[-1])

    # The fewest animals against all, paired on the units both were scored on: each
    # unit's score averaged over the draws.
    _pair = _recs[(_recs["set"] == test_a) & _recs["method"].isin([_ref, *_fewest])]
    _pair = _pair.assign(
        method=_pair["method"].where(_pair["method"] == _ref, "fewest")
    )
    _shared = set.intersection(
        *(set(_pair.loc[_pair["method"] == n, "unit"]) for n in (_ref, "fewest"))
    )
    _diff = (
        analysis.bootstrap(
            _pair[_pair["unit"].isin(_shared)], "inhale", reference=_ref, unit="unit"
        )
        .set_index("method")
        .loc["fewest"]
    )
    v_size = {
        "animals": _listing(_counts, values.count),
        "sessions": _listing(
            (round(_mean(_f1, _of[n], "sessions")) for n in _counts), values.count
        ),
        "all_animals": values.count(_f1.loc[_ref, "animals"]),
        "all_sessions": values.count(_f1.loc[_ref, "sessions"]),
        "nested_f1": _listing((_mean(_f1, _of[n]) for n in _counts), values.num),
        "fewest": values.count(_counts[0]),
        "fewest_f1": values.num(_mean(_f1, _fewest)),
        "fewest_r": values.num(_mean(_r, _fewest)),
        "f1": values.num(_f1.loc[_ref, "mean"]),
        "r": values.num(_r.loc[_ref, "mean"]),
        "diff": values.num(_diff[f"{_ref} minus"]),
        "diff_ci": values.span(_diff["diff low"], _diff["diff high"]),
        "partial_sessions": values.count(round(_mean(_f1, _partial, "sessions"))),
        "partial_f1": values.num(_mean(_f1, _partial)),
        "partial_r": values.num(_mean(_r, _partial)),
        "draw_sessions": values.span(
            _f1.loc[_fewest, "sessions"].min(), _f1.loc[_fewest, "sessions"].max(), 0
        ),
        "draws_f1": _listing(_f1.loc[_fewest, "mean"], values.num),
        "draws_r": _listing(_r.loc[_fewest, "mean"], values.num),
        "draw_animals": values.count(_counts[0]),
        "draws": values.count(len(_fewest)),
        "fold": _ref.split(" | ")[-1],
        "units": values.count(_f1.loc[_ref, "units"]),
    }
    # Rig coverage (Methods): new animals (test A) by whether the run trained on their
    # rig. Unpaired: the units in each group differ.
    _tested = _recs[_recs["set"] == test_a]
    _per_unit = (
        _tested.join(analysis.familiarity(_tested, _trained))
        .groupby(["method", "set", "rig_trained", "unit"], as_index=False)[
            ["inhale", "correlation"]
        ]
        .mean()
    )
    rig_coverage = _per_unit.groupby(["method", "rig_trained"], as_index=False).agg(
        units=("unit", "size"), inhale=("inhale", "mean"), r=("correlation", "mean")
    )
    fig5a = plots.learning_curve_panel(
        *size("fig4", "c"),
        learning,
        {"New animals": style.color(test_a)},
    )
    plots.save_panel(fig5a, FIGURES / "fig4" / "c.svg")
    mo.vstack(
        [
            fig5a,
            mo.ui.table(learning.round(3), selection=None),
            mo.md("New animals (test A), by whether the rig was in training"),
            mo.ui.table(rig_coverage.round(3), selection=None),
        ]
    )
    return (v_size,)


@app.cell
def fig4df(test_a):
    # A test-A clip that is not an outlier: among clips whose correlation is at least
    # the median, the one with most near misses (predictions 25-100 ms off), around its
    # densest near misses (b).
    _one = runs.SINGLE
    _clips = (
        runs.scores([_one])
        .query("set == @test_a")
        .loc[lambda d: d["stem"].map(runs.is_local)]
        .sort_values("inhale")
        .set_index("stem")
    )
    _lags = runs.tables(_one)["lags"].query("set == @test_a")
    _far = (
        _lags[_lags["lag_s"].abs() > plots.TOLERANCE_MS / 1000].groupby("stem").size()
    )
    _good = _clips.index[_clips["correlation"] >= _clips["correlation"].median()]
    _near = _far.reindex(_good).idxmax()
    _low = _clips[_clips["correlation"] <= _clips["correlation"].quantile(0.1)].drop(
        _near, errors="ignore"
    )

    _cases = {"b": _clips.loc[_near]}
    low_r = {
        "clips": len(_low),
        "below r": float(_clips["correlation"].quantile(0.1)),
        "median F1": float(_low["inhale"].median()),
        "share F1 >= 0.9": float((_low["inhale"] >= 0.9).mean()),
        "spearman r, F1": float(
            _clips[["correlation", "inhale"]].corr("spearman").iloc[0, 1]
        ),
    }
    _shown = []
    for _panel, _row in _cases.items():
        _stem = _row.name
        # not .query("stem == @_stem"): marimo renames a cell's "_" names
        _errors = runs.set_errors(_one, test_a)
        _errors = _errors[_errors["stem"] == _stem]
        _p = runs.prediction(_one, _stem)
        _origin = float(_p["times"][0])
        _truth = runs.truth(_stem)
        _t = _errors.loc[_errors["near_miss"], "time"].to_numpy()
        _at = float(_t[np.argmax([np.sum(np.abs(_t - x) < 1.5) for x in _t])])
        _fig = plots.failure_panel(
            *size("fig6", _panel),
            _truth,
            _p,
            _errors,
            _origin,
            _at - _origin,
            [],
            half=1.0,
            legend=_panel == "b",
        )
        plots.save_panel(_fig, FIGURES / "fig6" / f"{_panel}.svg")
        _shown += [
            mo.md(
                f"**{_panel}**: {_stem}, inhale F1 {_row['inhale']:.3f}, "
                f"r {_row['correlation']:.3f}"
            ),
            _fig,
        ]
    v_clips = {
        "clips": values.count(len(_clips)),
        "spearman": values.num(low_r["spearman r, F1"]),
        "low_clips": values.count(low_r["clips"]),
        "low_r": values.num(low_r["below r"]),
        "low_f1": values.num(low_r["median F1"]),
        "low_f1_quartile": values.num(_low["inhale"].quantile(0.75)),
        "near_r": values.num(_cases["b"]["correlation"]),
        "near_f1": values.num(_cases["b"]["inhale"]),
        "threshold": values.num(runs.head_threshold(_one)),
    }
    mo.vstack(
        [
            mo.md(f"Lowest tenth by correlation: {low_r}"),
            *_shown,
        ]
    )
    return (v_clips,)


@app.cell
def fig6_md():
    mo.md("""
    ## Fig. 6: scores of predictions without a thermistor

    The quality-control scores the results manifest's `[quality]` names, joined with
    their truth by the quality analysis: the all-camera network's validation windows
    (its reference), sets where it works, and known failures; made with `zephyr
    quality-control reference` and `run`, and the noise null with `zephyr-benchmarks
    probe`.
    """)
    return


@app.cell
def fig6efg():
    # Fig. 6e-g: quality control as scores, never a verdict. g: each measure over the
    # 30-s windows of the network's validation animals, sets where it works and known
    # failures; e: one dynamic-foraging clip predicted before and after refining, the
    # clip whose refined true inhale F1 is nearest the median; f: one held-out clip of
    # the noise null at every level, the clip whose own true inhale F1 is nearest the
    # median.
    qc_windows = runs.quality_windows().assign(row=lambda d: d["set"])
    _sets = [
        s for s in runs.RESULTS.quality.set if s.role not in ("application", "camera")
    ]
    _rows = [("Validation", "validation"), *[(s.label, s.role) for s in _sets]]
    fig6g = plots.quality_sets_panel(*size("fig6", "e"), qc_windows, _rows)
    plots.save_panel(fig6g, FIGURES / "fig6" / "e.svg")

    _noise, _ = runs.quality_noise()
    _own = _noise[_noise["sigma"] == 0].set_index("stem")["truth_inhale_f1"]
    _noisy_stem = (_own - _own.median()).abs().idxmin()
    _clip_truth = runs.truth(_noisy_stem)
    _levels = runs.quality_noise_traces(_noisy_stem)
    _scored = _noise[_noise["stem"] == _noisy_stem].set_index("sigma")
    _rows_f, _corner, _colours = {}, {}, {}
    for _sigma, _trace in _levels.items():
        if _sigma not in (0, *style.NOISE):
            continue
        _name = "No noise" if _sigma == 0 else f"Noise {_sigma:g}"
        _times = _trace["time"].to_numpy()
        _n = min(len(_times), len(_clip_truth["signal"]))
        _rows_f[_name] = {
            "times": _times[:_n],
            "truth": _clip_truth["signal"][:_n],
            "truth_inhale": _clip_truth["inhale"],
            "signal": _trace["signal"].to_numpy()[:_n],
            "head": _times[_trace["inhale"].to_numpy(bool)],
            "start": round(float(_times[_n // 2] - _times[0]) - 6.0),
        }
        _colours[_name] = style.NOISE.get(_sigma, style.color("Zephyr, all cameras"))
        _corner[_name] = "F1 {:.2f}, r {:.2f}".format(
            _scored.loc[_sigma, "truth_inhale_f1"],
            _scored.loc[_sigma, "truth_correlation"],
        )
    fig6f = plots.refined_traces_panel(
        *size("fig6", "d"), _rows_f, colours=_colours, corner=_corner, ncols=2
    )
    plots.save_panel(fig6f, FIGURES / "fig6" / "d.svg")

    _before, _after = "Dynamic foraging, unrefined", "Dynamic foraging, refined"
    _f1 = qc_windows[qc_windows["set"] == _after].groupby("stem")["truth_inhale_f1"]
    _f1 = _f1.mean()
    _stem = (_f1 - _f1.median()).abs().idxmin()
    _truth = runs.truth(_stem)
    _rows_e, _notes = {}, {}
    for _label, _variant in (
        (_before, "Zephyr, all cameras, on DF"),
        (_after, "Zephyr, all cameras, refined on DF"),
    ):
        _trace = runs.quality_traces(_label, _stem)
        _times = _trace["time"].to_numpy()
        _n = min(len(_times), len(_truth["signal"]))
        _rows_e[_variant] = {
            "times": _times[:_n],
            "truth": _truth["signal"][:_n],
            "truth_inhale": _truth["inhale"],
            "signal": _trace["signal"].to_numpy()[:_n],
            "head": _times[_trace["inhale"].to_numpy(bool)],
            # 12 s around the clip's middle, in s from its start
            "start": round(float(_times[_n // 2] - _times[0]) - 6.0),
        }
        _clip = qc_windows[
            (qc_windows["set"] == _label) & (qc_windows["stem"] == _stem)
        ]
        _notes[_variant] = (
            "head vs waveform {:.2f}, breaks {:.2f}, peak {:.2f}, background {:.2f}; "
            "true F1 {:.2f}"
        ).format(*_clip[[*style.QC_MEASURES, "truth_inhale_f1"]].median())
    fig6e = plots.refined_traces_panel(
        *size("fig6", "c"),
        _rows_e,
        labels={
            "Zephyr, all cameras, on DF": "Before\nrefining",
            "Zephyr, all cameras, refined on DF": "After\nrefining",
        },
        notes=_notes,
    )
    plots.save_panel(fig6e, FIGURES / "fig6" / "c.svg")

    _median = qc_windows.groupby("set")[
        [*style.QC_MEASURES, "truth_inhale_f1"]
    ].median()
    _key = {
        "Validation": "validation",
        "Held-out animals": "held_out",
        "Dynamic foraging, refined": "df_refined",
        "Dynamic foraging, unrefined": "df_unrefined",
        "Noise 16": "noise16",
        "Noise 32": "noise32",
        "Noise 64": "noise64",
        "Sessions without a thermistor": "sessions",
        "Sessions without a thermistor, side camera": "sessions_side",
        "Sessions without a thermistor, front camera": "sessions_front",
    }
    v_quality = {
        f"{_key[k]}_{m}": values.num(v, 2)
        for k, row in _median.iterrows()
        for m, v in row.items()
        if pd.notna(v)
    } | {
        "validation_windows": values.count((qc_windows["set"] == "Validation").sum()),
        "held_out_windows": values.count(
            (qc_windows["set"] == "Held-out animals").sum()
        ),
        "df_clips": values.count(
            qc_windows.loc[qc_windows["set"] == _after, "stem"].nunique()
        ),
    }
    mo.vstack([fig6e, fig6f, fig6g, mo.ui.table(_median.round(3).reset_index())])
    return qc_windows, v_quality


@app.cell
def fig6a(test_a):
    # The share of our network's errors on new animals that are near misses, per fold,
    # then over folds.
    _folds = runs.RESULTS.cross_validation.folds
    _names = {f: runs.cv_name(runs.REFERENCE, f) for f in _folds}
    _errors = pd.concat(
        runs.set_errors(n, test_a).assign(fold=f) for f, n in _names.items()
    )
    _profile = analysis.error_profile(_errors)
    error_profile = analysis.across_folds(_profile, "share", by=["kind", "row"]).merge(
        _profile.groupby(["kind", "row"])["n"].sum().reset_index(), on=["kind", "row"]
    )
    _p = error_profile.set_index(["kind", "row"])
    _kinds = {"misses": "missed (FN)", "false": "false (FP)"}
    v_errors = {
        **{k: values.count(_p.loc[(v, "near miss"), "n"]) for k, v in _kinds.items()},
        **{
            f"{k}_near": values.pct(_p.loc[(v, "near miss"), "mean"])
            for k, v in _kinds.items()
        },
    }
    mo.ui.table(error_profile.round(3), selection=None)
    return (v_errors,)


@app.cell
def tables_md():
    mo.md("""
    ## Supplementary tables

    The sets of each cross-validation fold, the comparison networks' settings (from each
    run's own `config.json`); written as CSV for main.typ. The 359-Hz sessions each fold
    holds out (its test set C) are not scored in the paper.
    """)
    return


@app.cell
def supplementary_tables(folds):
    _tables = FIGURES.parent / "tables"

    # The sets of each cross-validation fold, as the capsule's experiment trained and
    # scored them (experiments/face.toml: each set's sessions), with a check that each
    # fold's test A animals are those its runs were scored on.
    _experiment = tomllib.loads(
        (runs.MANUSCRIPT / "experiments" / "face.toml").read_text(encoding="utf8")
    )
    _subject = dict(
        zip(folds["session_id"], folds["subject_id"].astype(str), strict=True)
    )

    def _sessions(fold: dict, name: str) -> list[str]:
        """The sessions of set *name* (``train``, ``val``, ``test_a``-``c``) of *fold*."""
        if name == "train":
            return [g for part in fold["train"] for g in part["groups"]]
        if name == "val":
            return list(fold["val"]["groups"])
        return list(fold["test"][name]["groups"])

    _names = {
        "train": "Training",
        "val": "Validation",
        "test_a": "Held-out animals",
        "test_b": "Held-out sessions",
        "test_c": "Not scored (359 Hz)",
    }
    _rows, _checked = [], {}
    _scored = runs.cv_scores("per_recording")
    for _f in runs.RESULTS.cross_validation.folds:
        (_fold,) = [f for f in _experiment["fold"] if f["name"] == _f]
        for _set, _label in _names.items():
            _in = _sessions(_fold, _set)
            _rows.append(
                {
                    "fold": _f,
                    "set": _label,
                    "animals": len({_subject[s] for s in _in}),
                    "sessions": len(_in),
                }
            )
        _a = _scored[
            (_scored["fold"] == _f)
            & (_scored["method"] == runs.REFERENCE)
            & (_scored["set"] == runs.SETS["test_a"])
        ]
        _want = {_subject[s] for s in _sessions(_fold, "test_a")}
        _checked[_f] = set(_a["subject"].astype(str)) == _want
    fold_sizes = pd.DataFrame(_rows).pivot(
        index="set", columns="fold", values=["animals", "sessions"]
    )
    _wide = pd.DataFrame(index=list(_names.values()))
    for _f in runs.RESULTS.cross_validation.folds:
        _wide[_f] = [
            f"{fold_sizes.loc[_l, ('animals', _f)]} / {fold_sizes.loc[_l, ('sessions', _f)]}"
            for _l in _wide.index
        ]
    _wide.rename_axis("set").reset_index().to_csv(_tables / "cv_folds.csv", index=False)

    # The networks' settings, from each run's config.json, with their size and
    # inference time per 5-min clip on the GPU.
    _nets = ["Zephyr", "TS-CAN", "PhysNet", "PhysFormer", "RhythmFormer"]
    _gpu = runs.wall_time(_nets).set_index("method")
    _rows = []
    for _m in _nets:
        _path = runs.run_file(_m, "config.json")  # its run on the first fold
        _tp = json.loads(_path.read_text())["fold"]["train_params"]
        _rows.append(
            {
                "method": _m,
                "window": _tp["window"],
                "batch": _tp["batch_size"],
                "steps": _tp["steps_per_epoch"],
                "lr": _tp["lr"],
                "infer_window": _tp["infer_window"],
                # only Zephyr and PhysNet read the scales; the others' are an unused
                # default
                "scales": (
                    "/".join(map(str, _tp["scales"]))
                    if _m in ("Zephyr", "PhysNet")
                    else "—"
                ),
                "params_m": f"{_gpu.loc[_m, 'parameters (M)']:.2f}",
                "gpu_s": f"{_gpu.loc[_m, 's per clip (median)']:.2f}",
            }
        )
    comparison_nets = pd.DataFrame(_rows)
    comparison_nets.to_csv(_tables / "comparison_nets.csv", index=False)
    mo.vstack(
        [
            mo.md(f"Folds match their runs: {_checked}"),
            mo.ui.table(_wide.reset_index(), selection=None),
            mo.ui.table(comparison_nets, selection=None),
        ]
    )
    return comparison_nets, fold_sizes


@app.cell
def figs1(by_fold, test_a):
    # Supplementary Fig. 1: why the unit is an animal on one rig. Our network on new
    # animals (every fold's test A, each animal new once): how far apart two units
    # score when they share the animal but not the rig, and when they share the rig
    # but not the animal, with intervals from resampling animals.
    _a = by_fold[(by_fold["set"] == test_a) & (by_fold["method"] == runs.REFERENCE)]
    _pairs = {m: analysis.rig_animal_pairs(_a, m) for m in ("inhale", "correlation")}
    _summaries = {m: analysis.rig_animal_bootstrap(p) for m, p in _pairs.items()}
    figs1a = plots.rig_animal_panel(
        *size("figs1", "a"),
        {m: (_pairs[m], _summaries[m]) for m in _pairs},
        runs.REFERENCE,
    )
    plots.save_panel(figs1a, FIGURES / "figs1" / "a.svg")
    rig_animal = pd.concat(
        [v.assign(metric=k) for k, v in _summaries.items()], ignore_index=True
    )
    mo.vstack([figs1a, mo.ui.table(rig_animal.round(4), selection=None)])
    return (rig_animal,)


@app.cell
def figs2(by_fold, rate_share, test_a, with_unit):
    # Supplementary Fig. 2: our network fed 30-Hz video (every other frame), still
    # predicting on the 60-Hz grid, against the same network on 60-Hz video, on every
    # fold. a: the difference per test set, paired by fold; b: inhale F1 on new animals
    # by breathing rate (each fold's breaths pooled, then over folds).
    _ref = by_fold[by_fold["method"] == runs.REFERENCE]
    _recs = pd.concat(
        [_ref, with_unit(runs.cv_scores(variants=True))], ignore_index=True
    )
    _recs = _recs[_recs["set"].isin([runs.SETS["test_a"], runs.SETS["test_b"]])]
    _name = next(iter(runs.CV_VARIANTS))
    _folds, _summaries = [], []
    for _m in ("correlation", "inhale", "exhale_f1"):
        _f = analysis.per_fold(_recs, _m)
        _p = _f[_f["method"] == _name].merge(
            _f[_f["method"] == runs.REFERENCE],
            on=["set", "fold"],
            suffixes=("", "_ref"),
        )
        _p = _p.assign(value=_p["value"] - _p["value_ref"], metric=_m)
        _folds.append(_p[["metric", "set", "fold", "value"]])
        _summaries.append(
            analysis.across_folds(_p, "value", by=["set"]).assign(metric=_m)
        )
    input_rate_folds = pd.concat(_folds, ignore_index=True)
    input_rate = pd.concat(_summaries, ignore_index=True)
    figs2a = plots.fold_difference_panel(
        *size("figs2", "a"), input_rate_folds, input_rate, _name, runs.REFERENCE
    )
    plots.save_panel(figs2a, FIGURES / "figs2" / "a.svg")
    _tables = {"breaths": [], "false_positives": []}
    for _n in (runs.REFERENCE, _name):
        for _f in runs.RESULTS.cross_validation.folds:
            _t = runs.tables(runs.cv_name(_n, _f))
            for _k, _parts in _tables.items():
                _parts.append(_t[_k].query("set == @test_a").assign(method=_n, fold=_f))
    _by_rate = analysis.f1_by_rate(
        pd.concat(_tables["breaths"], ignore_index=True),
        pd.concat(_tables["false_positives"], ignore_index=True),
        analysis.RATE_BINS_HZ[1:],
    )
    input_rate_by_rate = analysis.across_folds(
        _by_rate, "f1", by=["method", "bin"]
    ).merge(
        _by_rate.groupby(["method", "bin"], observed=True)["share"]
        .mean()
        .reset_index(),
        on=["method", "bin"],
    )
    _breaths = pd.concat(_tables["breaths"], ignore_index=True)
    figs2b = plots.rate_band_panel(
        *size("figs2", "b"),
        input_rate_by_rate,
        [runs.REFERENCE, _name],
        rate_share(_breaths[_breaths["method"] == runs.REFERENCE]),
        ylim=(0.8, 1.0),
    )
    plots.save_panel(figs2b, FIGURES / "figs2" / "b.svg")
    mo.vstack(
        [
            figs2a,
            figs2b,
            mo.ui.table(input_rate.round(4), selection=None),
            mo.ui.table(input_rate_by_rate.round(3), selection=None),
        ]
    )
    return input_rate, input_rate_by_rate


@app.cell
def figs3(rate_share, test_a, with_unit):
    # Supplementary Fig. 3 and Table S5: our network trained on each camera's video, on
    # the cross-validation folds every camera has so far, scored on the recording spans
    # every camera has, all against the same thermistor events (the face clips').
    _runs = {
        "Face camera": runs.REFERENCE,
        "Side camera": "Zephyr, side camera",
        "Front camera": "Zephyr, front camera",
    }
    _cameras = list(_runs)
    _folds = [
        f
        for f in runs.RESULTS.cross_validation.folds
        if all(runs.evaluation(runs.cv_name(r, f)) is not None for r in _runs.values())
    ]
    _sets = [test_a, runs.SETS["test_b"]]
    _clips = pd.concat(
        [runs.cv_scores("clips"), runs.cv_scores("clips", variants=True)],
        ignore_index=True,
    )
    _clips = _clips[
        _clips["fold"].isin(_folds)
        & _clips["set"].isin(_sets)
        & _clips["method"].isin(_runs.values())
    ]
    _clips = with_unit(
        _clips.assign(
            method=_clips["method"].map({r: c for c, r in _runs.items()}),
            span=_clips["stem"].map(runs._span),
        )
    )
    # Only the spans every camera has, in each fold.
    _n = _clips.groupby(["fold", "span"])["method"].nunique()
    _shared = set(_n[_n == len(_cameras)].index)
    camera_clips = _clips[
        [k in _shared for k in zip(_clips["fold"], _clips["span"], strict=True)]
    ]
    # Fig. 5b: each camera on new animals and new sessions; table: every score per set.
    _ci = {
        m: analysis.bootstrap(camera_clips, m, reference=_cameras[0], unit="unit")
        for m in ("correlation", "inhale", "exhale_f1")
    }
    plots.save_panel(
        plots.sets_panel(
            *size("fig4", "d"),
            pd.concat(_ci[m].assign(metric=m) for m in ("correlation", "inhale")),
            _cameras,
            ["correlation", "inhale"],
            _sets,
            labels=False,
            legend="lower right",
        ),
        FIGURES / "fig4" / "d.svg",
    )
    # Fig. 5c: one network trained on all three cameras against the network of each
    # camera, on each camera's clips of the same spans of new animals.
    _pooled = runs.cv_scores("clips", variants=True).query(
        "method == 'Zephyr, all cameras' and set == @test_a"
    )
    _camera_of = {"FaceCamera": "Face", "SideCamera": "Side", "FrontCamera": "Front"}
    _pooled = with_unit(
        _pooled[_pooled["fold"].isin(_folds)].assign(
            span=_pooled["stem"].map(runs._span),
            camera=_pooled["stem"].map(lambda s: _camera_of[s.rsplit("_", 1)[1]]),
        )
    )
    _pooled = _pooled[
        [k in _shared for k in zip(_pooled["fold"], _pooled["span"], strict=True)]
    ]
    _single = camera_clips[camera_clips["set"] == test_a]
    _both = pd.concat(
        [
            _single.assign(camera=_single["method"].str.split().str[0]),
            _pooled.assign(method="All cameras"),
        ],
        ignore_index=True,
    )
    _both.loc[_both["method"] != "All cameras", "method"] = "One camera"
    _pool_ci = {
        m: analysis.bootstrap(
            _both.assign(set=_both["camera"]), m, reference="One camera", unit="unit"
        )
        for m in ("correlation", "inhale")
    }
    _pool = {m: t.set_index(["set", "method"]) for m, t in _pool_ci.items()}

    def _gap(m: str, digits: int) -> str:
        """Pooled minus single-camera, its range over the three cameras."""
        d = -_pool[m].xs("All cameras", level="method")["One camera minus"]
        return values.span(d.min(), d.max(), digits)

    v_pooled = {
        **{
            f"{k}_{c.lower()}": values.num(_pool[m].loc[(c, "All cameras"), "mean"])
            for k, m in (("f1", "inhale"), ("r", "correlation"))
            for c in ("Face", "Side", "Front")
        },
        "diff_f1": _gap("inhale", 3),
        "diff_r": _gap("correlation", 2),
        # each camera's gain (pooled minus its own network) with its 95% interval
        **{
            f"gain_{k}_{c.lower()}": values.num(
                -_pool[m].loc[(c, "All cameras"), "One camera minus"], 3
            )
            for k, m in (("f1", "inhale"), ("r", "correlation"))
            for c in ("Face", "Side", "Front")
        },
        **{
            f"gain_{k}_{c.lower()}_ci": values.span(
                -_pool[m].loc[(c, "All cameras"), "diff high"],
                -_pool[m].loc[(c, "All cameras"), "diff low"],
                3,
            )
            for k, m in (("f1", "inhale"), ("r", "correlation"))
            for c in ("Face", "Side", "Front")
        },
        "units": values.count(_pooled["unit"].nunique()),
    }
    # a: each new-animal unit, ranked by the face camera, as Fig. 4b.
    figs3a = plots.animals_panel(
        *size("figs3", "a"),
        camera_clips,
        _cameras,
        "inhale",
        test_a,
        unit="unit",
        unit_label="Animal × rig",
    )
    plots.save_panel(figs3a, FIGURES / "figs3" / "a.svg")
    # b: inhale F1 on new animals by breathing rate, over the shared spans.
    _tables = {k: [] for k in ("breaths", "false_positives", "windows")}
    for _camera, _run in _runs.items():
        for _f in _folds:
            _t = runs.tables(runs.cv_name(_run, _f))
            for _k, _parts in _tables.items():
                _x = _t[_k].assign(method=_camera, fold=_f)
                _keep = [(_f, runs._span(s)) in _shared for s in _x["stem"].astype(str)]
                _parts.append(_x[_keep])
    _tables = {k: pd.concat(v, ignore_index=True) for k, v in _tables.items()}
    _by_rate = analysis.f1_by_rate(
        _tables["breaths"].query("set == @test_a"),
        _tables["false_positives"].query("set == @test_a"),
        analysis.RATE_BINS_HZ[1:],
    )
    camera_by_rate = analysis.across_folds(_by_rate, "f1", by=["method", "bin"]).merge(
        _by_rate.groupby(["method", "bin"], observed=True)["share"]
        .mean()
        .reset_index(),
        on=["method", "bin"],
    )
    figs3b = plots.rate_band_panel(
        *size("fig4", "e"),
        camera_by_rate,
        _cameras,
        legend=False,  # colours as in Fig. 3d
    )
    plots.save_panel(figs3b, FIGURES / "fig4" / "e.svg")
    # Fig. 5c: inhale F1 by rate, as Fig. 3c, of each camera's network (solid) and of
    # the network trained on all three cameras (dashed) on that camera's clips, over the
    # same shared spans of new animals.
    _pooled_tables = {k: [] for k in ("breaths", "false_positives")}
    for _f in _folds:
        _t = runs.tables(runs.cv_name("Zephyr, all cameras", _f))
        for _k, _parts in _pooled_tables.items():
            _x = _t[_k].assign(fold=_f)
            _stems = _x["stem"].astype(str)
            _x = _x.assign(
                method=_stems.map(
                    lambda s: f"{_camera_of[s.rsplit('_', 1)[1]]} camera, all cameras"
                )
            )
            _parts.append(_x[[(_f, runs._span(s)) in _shared for s in _stems]])
    _pooled_tables = {
        k: pd.concat(v, ignore_index=True) for k, v in _pooled_tables.items()
    }
    _pooled_rate = analysis.f1_by_rate(
        _pooled_tables["breaths"].query("set == @test_a"),
        _pooled_tables["false_positives"].query("set == @test_a"),
        analysis.RATE_BINS_HZ[1:],
    )
    pooled_by_rate = analysis.across_folds(
        _pooled_rate, "f1", by=["method", "bin"]
    ).merge(
        _pooled_rate.groupby(["method", "bin"], observed=True)["share"]
        .mean()
        .reset_index(),
        on=["method", "bin"],
    )
    # Top: the pooled network over all three cameras' breaths; bottom: per camera, the
    # pooled network minus that camera's own.
    _over = analysis.f1_by_rate(
        _pooled_tables["breaths"].query("set == @test_a").assign(method="All cameras"),
        _pooled_tables["false_positives"]
        .query("set == @test_a")
        .assign(method="All cameras"),
        analysis.RATE_BINS_HZ[1:],
    )
    pooled_overall = analysis.across_folds(_over, "f1", by=["method", "bin"])
    _single = camera_by_rate.set_index(["method", "bin"])["mean"]
    pooled_delta = pooled_by_rate.assign(
        camera=pooled_by_rate["method"].str.removesuffix(", all cameras")
    ).assign(
        delta=lambda d: (
            d["mean"].to_numpy()
            - _single.reindex(list(zip(d["camera"], d["bin"], strict=True))).to_numpy()
        )
    )
    fig5c = plots.pooled_rate_panel(
        *size("fig4", "f"), pooled_overall, pooled_delta, _cameras
    )
    plots.save_panel(fig5c, FIGURES / "fig4" / "f.svg")
    # c: every new-animal clip, each other camera against the face camera.
    _wide = (
        camera_clips[camera_clips["set"] == test_a]
        .pivot_table(index=["fold", "span"], columns="method", values="inhale")
        .reset_index()
    )
    figs3c = plots.paired_scatter_panel(
        *size("figs3", "b"), _wide, _cameras[0], _cameras[1:], "inhale"
    )
    plots.save_panel(figs3c, FIGURES / "figs3" / "b.svg")
    # Table S5: per set and camera, every score over units (mean, 95% bootstrap
    # interval), with the breathing-rate MAE of 10-s windows (breaths per minute).
    _w = _tables["windows"]
    _unit = camera_clips.drop_duplicates("stem").set_index("stem")["unit"]
    _w = _w.assign(unit=_w["stem"].map(_unit)).dropna(subset=["unit"])
    _mae = analysis.rate_agreement(_w, by=["method", "set", "unit"]).reset_index()
    _mae["mae"] = 60 * _mae["MAE (Hz)"]
    _ci["mae"] = analysis.bootstrap(_mae, "mae", reference=_cameras[0], unit="unit")

    def _cell(row, digits: int) -> str:
        return (
            f"{values.num(row['mean'], digits)} "
            f"({values.span(row['low'], row['high'], digits)})"
        )

    _rows = []
    for _set in _sets:
        for _camera in _cameras:
            _r = {
                m: t.set_index(["set", "method"]).loc[(_set, _camera)]
                for m, t in _ci.items()
            }
            _rows.append(
                {
                    "set": {
                        runs.SETS["test_a"]: "Held-out animals",
                        runs.SETS["test_b"]: "Held-out sessions",
                    }[_set],
                    "camera": _camera.split()[0],
                    "units": values.count(_r["inhale"]["units"]),
                    "trace_r": _cell(_r["correlation"], 2),
                    "inhale_f1": _cell(_r["inhale"], 2),
                    "exhale_f1": _cell(_r["exhale_f1"], 2),
                    "mae": _cell(_r["mae"], 1),
                }
            )
    cameras_table = pd.DataFrame(_rows)
    cameras_table.to_csv(FIGURES.parent / "tables" / "cameras.csv", index=False)
    _a = camera_clips[camera_clips["set"] == test_a]
    _b = camera_clips[camera_clips["set"] == runs.SETS["test_b"]]
    _on_a = {m: t.set_index(["set", "method"]) for m, t in _ci.items()}
    v_cameras = {
        **{
            f"{k}_{c.split()[0].lower()}": values.num(_on_a[m].loc[(test_a, c), "mean"])
            for k, m in (("f1", "inhale"), ("r", "correlation"))
            for c in _cameras
        },
        "folds": values.count(len(_folds)),
        "clips_a": values.count(len(_a) / len(_cameras)),
        "clips_b": values.count(len(_b) / len(_cameras)),
        "units_a": values.count(_a["unit"].nunique()),
        "units_b": values.count(_b["unit"].nunique()),
        "animals_a": values.count(_a["subject"].nunique()),
        "breaths": values.count(
            (_tables["breaths"].query("set == @test_a")["method"] == _cameras[0]).sum()
        ),
    }
    mo.vstack(
        [
            mo.hstack([figs3a, figs3b, figs3c]),
            mo.ui.table(cameras_table, selection=None),
            mo.md(f"{v_cameras}"),
            fig5c,
            mo.md(f"{v_pooled}"),
            mo.ui.table(pd.concat(_pool_ci.values()).round(3), selection=None),
        ]
    )
    return v_cameras, v_pooled


@app.cell
def figs4():
    # Supplementary Fig. 4: what the three cameras see, in a representative subset of
    # sessions (picks.VIEW_SESSIONS), its three cameras side by side with the crop box
    # each network reads (the corpus analysis): face, the corpus's thumbnail and clip
    # list; side and front, the first frame of the clip their clip lists' boxes are on.
    _sessions = [
        {"label": _s, "views": runs.camera_views(_s)} for _s in picks.VIEW_SESSIONS
    ]
    figs4a = plots.camera_views_panel(LAYOUT["figs4"]["width_mm"], _sessions)
    plots.save_panel(figs4a, FIGURES / "figs4" / "a.svg")
    v_views = {"sessions": values.count(len(_sessions))}
    figs4a
    return (v_views,)


@app.cell
def manuscript_values(
    v_cameras,
    v_clips,
    v_corpus,
    v_errors,
    v_fig1,
    v_fig3,
    v_rate,
    v_rates,
    v_cameras_missing,
    v_selection,
    v_size,
    v_tolerance,
    v_pooled,
    v_application,
    v_quality,
    v_refine,
    v_saliency,
    v_slow,
    v_views,
):
    # Every number the text cites, as printed, for main.typ (#v("section.key")).
    _cv = [
        runs.head_threshold(runs.cv_name(runs.REFERENCE, f))
        for f in runs.RESULTS.cross_validation.folds
    ]
    v_threshold = {
        "folds": values.span(min(_cv), max(_cv)),
    }
    manuscript = {
        "benchmark": v_fig3,
        "cameras": v_cameras,
        "clips": v_clips,
        "corpus": v_corpus,
        "errors": v_errors,
        "rate": v_rate,
        "rates": v_rates,
        "selection": v_selection,
        "optional_cameras": v_cameras_missing,
        "size": v_size,
        "task": v_fig1,
        "threshold": v_threshold,
        "tolerance": v_tolerance,
        "pooled": v_pooled,
        "application": v_application,
        "quality": v_quality,
        "refine": v_refine,
        "saliency": v_saliency,
        "slow": v_slow,
        "views": v_views,
    }
    values.write(
        FIGURES.parent / "values.toml", manuscript, "notebooks/manuscript_figures.py"
    )
    mo.ui.table(
        [
            {"value": f"{s}.{k}", "printed": v}
            for s, keys in manuscript.items()
            for k, v in keys.items()
        ],
        selection=None,
    )
    return


if __name__ == "__main__":
    app.run()
