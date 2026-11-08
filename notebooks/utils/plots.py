"""Reusable figure components. Every colour and size comes from :mod:`.style`.

Times on a clip's traces are seconds from the clip's first output sample.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from zephyr.benchmarks import analysis

from . import style

FIGURES: Path = Path(__file__).resolve().parents[1] / "figures"
CROP_EDGE: str = "#d55e00"
"""Outline of the crop the network sees, on the whole frame."""
METRIC_LABELS: dict[str, str] = {
    "correlation": "Correlation",
    "inhale": "Inhale F1",
    "exhale_f1": "Exhale F1 (DSP)",
    "kl_ibi": "KL-IBI (lower is better)",
}
TOLERANCE_MS = 25.0
"""The scorer's event tolerance at 60 Hz (1.5 frames), drawn on timing plots."""


def save(fig: Figure, name: str) -> None:
    """Write *fig* as PNG and SVG into ``notebooks/figures``."""
    FIGURES.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{ext}")


def save_panel(fig: Figure, path: Path) -> None:
    """Write a figure panel as SVG (text kept as text) at exactly the size it was made
    (:func:`utils.style.panel`): never cropped to fit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches=None would fall back to the style's "tight", which crops
    with mpl.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(path.with_suffix(".svg"))


def mark_events(
    ax: Axes, times: NDArray, y: float, color: str, marker: str = "v", size: float = 3
) -> None:
    """Mark event times along height *y* of an axes."""
    ax.plot(
        times,
        np.full(len(times), y),
        marker,
        color=color,
        markersize=size,
        clip_on=False,
        lw=0,
    )


def _inside(x: NDArray, t0: float, t1: float) -> NDArray:
    return (x >= t0) & (x <= t1)


def _crop_image(ax: Axes, image: NDArray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=7, weight="normal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


# --- one clip ---------------------------------------------------------------------------


# --- manuscript panels ------------------------------------------------------------------
# Drawn at their printed size by utils.style.panel; panel letters are added where the
# figure is assembled, never here.


def prediction_panel(
    width_mm: float,
    height_mm: float,
    truth: dict,
    prediction: dict,
    origin: float,
    t0: float,
    duration: float,
    thumbnail: NDArray | None = None,
    box: tuple[float, float, float, float] | None = None,
) -> Figure:
    """The session's face frame with the crop the network sees (left, when
    *thumbnail* is given), then Zephyr's respiration trace over the thermistor (top) and
    its onset probability (bottom), with the thermistor's and the head's inhalations."""
    if thumbnail is None:
        fig, ax = style.panel(
            width_mm, height_mm, [["trace"], ["onset"]], height_ratios=[2, 1]
        )
    else:
        fig, ax = style.panel(
            width_mm,
            height_mm,
            [["frame", "trace"], ["frame", "onset"]],
            height_ratios=[2, 1],
            width_ratios=[1, 3.6],
        )
        _crop_image(ax["frame"], thumbnail, "")
        if box is not None:
            ax["frame"].add_patch(
                Rectangle(
                    box[:2],
                    box[2],
                    box[3],
                    fill=False,
                    edgecolor=CROP_EDGE,
                    linewidth=0.8,
                )
            )
    top, bottom = ax["trace"], ax["onset"]
    bottom.sharex(top)
    top.tick_params(labelbottom=False, bottom=False)
    t1 = t0 + duration
    tt, tp = truth["times"] - origin, prediction["times"] - origin
    kt, kp = _inside(tt, t0, t1), _inside(tp, t0, t1)
    true_on, head = truth["inhale"] - origin, prediction["head"] - origin
    true_on, head = true_on[_inside(true_on, t0, t1)], head[_inside(head, t0, t1)]
    top.plot(tt[kt], truth["signal"][kt], color=style.TRUTH, label="Thermistor")
    top.plot(
        tp[kp], prediction["signal"][kp], color=style.color("Zephyr"), label="Zephyr"
    )
    for events, color, marker, y in (
        (true_on, style.TRUTH, "v", 1.12),
        (head, style.color("Zephyr"), "^", 1.02),
    ):
        top.plot(
            events,
            np.full(len(events), y),
            marker,
            color=color,
            markersize=2.5,
            lw=0,
            clip_on=False,
            transform=top.get_xaxis_transform(),
        )
    top.set_ylabel("Respiration (z)")
    top.legend(loc="lower right", bbox_to_anchor=(1, 1.08), ncol=2)
    bottom.plot(tp[kp], prediction["onset"][kp], color=style.ONSET)
    bottom.set_ylim(0, 1.05)
    bottom.set_yticks([0, 1])
    bottom.set_ylabel("Onset (p)")
    bottom.set_xlim(t0, t1)
    bottom.set_xlabel("Time (s)")
    return fig


def _site_legend(ax: Axes) -> None:
    """The kinds of odor site (:data:`style.SITES`), in a row above *ax*."""
    ax.legend(
        handles=[
            Line2D([], [], color=c, lw=0.9, label=k) for k, c in style.SITES.items()
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=len(style.SITES),
        columnspacing=1.0,
        fontsize=5.5,
        handlelength=1,
        borderaxespad=0.2,
    )


def task_summary_panel(
    width_mm: float,
    height_mm: float,
    aligned: pd.DataFrame,
    samples: pd.DataFrame,
    grid_hz: float,
    speed_range: tuple[float, float, float] = (-5.0, 60.0, 5.0),
    rate_hz: tuple[float, float] = (1.0, 11.0),
) -> Figure:
    """Sniffing in the task. Top, breathing rate and velocity around odor onset per kind
    of site (*aligned*: ``site``, ``signal`` (``rate`` or ``speed``), ``lag``,
    ``mean`` with its 95% bootstrap interval ``low``, ``high``). Bottom, breathing rate against speed over every sample of
    *samples* (``speed``, ``rate``): the share of each speed bin at each rate (each
    speed bin sums to 1). Velocity bins are ``(low, high, step)`` of *speed_range*; rate bins
    over *rate_hz* hold whole numbers of intervals of the *grid_hz* grid the rate's
    events lie on (:func:`frame_rate_bins`)."""
    fig, ax = style.panel(
        width_mm,
        height_mm,
        [["avg_rate"], ["avg_speed"], ["heat"]],
        height_ratios=[1, 0.55, 1.5],
    )
    a, v = ax["avg_rate"], ax["avg_speed"]
    v.sharex(a)
    _site_legend(a)
    for site in style.SITES:
        colour = style.SITES[site]
        for target, signal in ((a, "rate"), (v, "speed")):
            g = aligned[(aligned["site"] == site) & (aligned["signal"] == signal)]
            if g.empty:
                continue
            target.fill_between(
                g["lag"],
                g["low"],
                g["high"],
                color=colour,
                alpha=0.25,
                lw=0,
            )
            target.plot(g["lag"], g["mean"], color=colour, lw=0.9)
    for target in (a, v):
        target.axvline(0, color="0.3", lw=0.5)
    a.set_ylabel("Respiration rate\n(Hz)")
    a.tick_params(labelbottom=False)
    v.set_ylabel("Velocity\n(cm/s)")
    v.set_xlabel("From odor onset (s)")
    v.set_xlim(aligned["lag"].min(), aligned["lag"].max())
    h = ax["heat"]
    _rate_speed(fig, h, samples, grid_hz, speed_range, rate_hz)
    fig.align_ylabels([a, v, h])
    return fig


def _rate_speed(
    fig: Figure,
    h: Axes,
    samples: pd.DataFrame,
    grid_hz: float,
    speed_range: tuple[float, float, float],
    rate_hz: tuple[float, float],
) -> None:
    """Respiration rate against velocity on *h* (see :func:`task_summary_panel`)."""
    speed_bins, rate_bins = rate_speed_bins(grid_hz, speed_range, rate_hz)
    counts = analysis.rate_speed_counts(samples, speed_bins, rate_bins)
    _share_mesh(fig, h, analysis.rate_given_speed(counts), speed_bins, rate_bins)


def rate_speed_bins(
    grid_hz: float,
    speed_range: tuple[float, float, float] = (-5.0, 60.0, 5.0),
    rate_hz: tuple[float, float] = (1.0, 11.0),
) -> tuple[NDArray, NDArray]:
    """Velocity bins, ``(low, high, step)`` of *speed_range*, and respiration-rate bins over
    *rate_hz* that hold whole numbers of intervals of the *grid_hz* grid
    (:func:`frame_rate_bins`)."""
    lo, hi, step = speed_range
    return np.arange(lo, hi + step / 2, step), frame_rate_bins(grid_hz, *rate_hz)


def _share_mesh(
    fig: Figure, h: Axes, share: NDArray, speed_bins: NDArray, rate_bins: NDArray
) -> None:
    """*share* (speed bins x rate bins, each speed bin summing to 1) on *h*."""
    image = h.pcolormesh(speed_bins, rate_bins, share.T, cmap="Blues", rasterized=True)
    h.set_xlabel("Velocity (cm/s)")
    h.set_ylabel("Respiration rate (Hz)")
    h.set_ylim(rate_bins[0], rate_bins[-1])
    h.grid(False)
    bar = fig.colorbar(image, ax=h, fraction=0.06, pad=0.02, aspect=15)
    bar.ax.tick_params(labelsize=5.5)
    bar.outline.set_visible(False)


def share_heat_panel(
    width_mm: float,
    height_mm: float,
    share: NDArray,
    speed_bins: NDArray,
    rate_bins: NDArray,
    title: str = "",
) -> Figure:
    """Respiration rate against velocity from a precomputed *share* (speed bins x rate
    bins, :func:`analysis.rate_given_speed`)."""
    fig, ax = style.panel(width_mm, height_mm, [["heat"]])
    _share_mesh(fig, ax["heat"], share, speed_bins, rate_bins)
    if title:
        ax["heat"].set_title(title, fontsize=6.5)
    return fig


def aligned_panel(
    width_mm: float,
    height_mm: float,
    aligned: pd.DataFrame,
    legend: bool = True,
    title: str = "",
    legend_columns: int | None = None,
) -> Figure:
    """Respiration rate (top) and running velocity (bottom) around odor onset per kind of
    site (*aligned*: ``kind``, ``signal`` (``rate`` or ``speed``), ``lag``, ``mean``,
    ``low``, ``high``, from :func:`analysis.odor_aligned` or
    :func:`analysis.aligned_across_units`), the band its interval."""
    fig, ax = style.panel(
        width_mm, height_mm, [["rate"], ["speed"]], height_ratios=[1, 0.6], sharex=True
    )
    for kind, colour in style.SITES.items():
        for signal in ("rate", "speed"):
            g = aligned[(aligned["kind"] == kind) & (aligned["signal"] == signal)]
            if g.empty:
                continue
            a = ax[signal]
            a.fill_between(
                g["lag"], g["low"], g["high"], color=colour, alpha=0.25, lw=0
            )
            a.plot(g["lag"], g["mean"], color=colour, lw=0.9)
    for a in ax.values():
        a.axvline(0, color="0.3", lw=0.5)
        a.xaxis.set_major_locator(ticker.MultipleLocator(2))
    if legend:
        ax["rate"].legend(
            handles=[
                Line2D([], [], color=c, lw=0.9, label=k) for k, c in style.SITES.items()
            ],
            loc="lower left",
            bbox_to_anchor=(-0.05, 1.0),
            ncol=legend_columns or len(style.SITES),
            columnspacing=0.5,
            handlelength=0.8,
            handletextpad=0.3,
            fontsize=5,
            borderaxespad=0.1,
        )
    if title:
        rows = -(-len(style.SITES) // (legend_columns or len(style.SITES)))
        ax["rate"].set_title(title, fontsize=6.5, pad=3 + 8 * rows if legend else 3)
    ax["rate"].set_ylabel("Respiration rate\n(Hz)")
    ax["speed"].set_ylabel("Velocity\n(cm/s)")
    ax["speed"].set_xlabel("From odor onset (s)")
    ax["speed"].set_xlim(aligned["lag"].min(), aligned["lag"].max())
    fig.align_ylabels(list(ax.values()))
    return fig


def rate_speed_panel(
    width_mm: float,
    height_mm: float,
    samples: pd.DataFrame,
    grid_hz: float,
    speed_range: tuple[float, float, float] = (-5.0, 60.0, 5.0),
    rate_hz: tuple[float, float] = (1.0, 11.0),
) -> Figure:
    """The bottom of :func:`task_summary_panel` alone: respiration rate against speed
    over every sample of *samples*, each speed bin's column summing to 1."""
    fig, ax = style.panel(width_mm, height_mm, [["heat"]])
    _rate_speed(fig, ax["heat"], samples, grid_hz, speed_range, rate_hz)
    return fig


def task_stretch_panel(
    width_mm: float,
    height_mm: float,
    example: pd.DataFrame,
    thermistor: pd.DataFrame,
    sites: pd.DataFrame,
    licks: NDArray,
    rewards: NDArray,
    label: str = "Thermistor\n(a.u.)",
    colour: str = style.TRUTH,
    inhalations: NDArray | None = None,
    onset: pd.DataFrame | None = None,
    speed_on_rate: bool = False,
    thumbnail: NDArray | None = None,
    box: tuple[float, float, float, float] | None = None,
) -> Figure:
    """A stretch of one session (*example*: ``time`` (s), ``speed``, ``rate``, the
    respiration rate; *thermistor*: ``time``, ``signal``, the respiration trace, from
    the thermistor or a network as *label* says, drawn in *colour*; *sites*: ``odor``,
    ``leave``; *licks*, *rewards*, *inhalations*: times; *onset*: ``time``, ``p``, a
    network's onset probability): running speed (its own row, or on a right axis over
    the rate if *speed_on_rate*), the trace with its *inhalations* as ticks above, the
    onset probability when given, then the respiration rate, with licks (ticks), reward
    deliveries (triangles) and odor sites (shaded); the session's *thumbnail* (with the
    crop *box*) on the left when given."""
    names = [
        *([] if speed_on_rate else ["speed"]),
        "signal",
        *(["onset"] if onset is not None else []),
        "rate",
    ]
    ratios = {"speed": 0.6, "signal": 1.1, "onset": 0.6, "rate": 1.0}
    if thumbnail is None:
        layout, widths = [[n] for n in names], None
    else:
        layout, widths = [["frame", n] for n in names], [1, 2.1]
    fig, ax = style.panel(
        width_mm,
        height_mm,
        layout,
        height_ratios=[ratios[n] for n in names],
        width_ratios=widths,
    )
    traces = [ax[n] for n in names]
    for a in traces[:-1]:
        a.sharex(traces[-1])
        a.tick_params(labelbottom=False)
    w, r = ax["signal"], ax["rate"]
    t0, t1 = example["time"].iloc[0], example["time"].iloc[-1]
    if thumbnail is not None:
        _crop_image(ax["frame"], thumbnail, "")
        ax["frame"].set_axis_off()
        if box is not None:
            ax["frame"].add_patch(
                Rectangle(
                    box[:2], box[2], box[3], fill=False, edgecolor=CROP_EDGE, lw=0.8
                )
            )
    if onset is not None:
        o = ax["onset"]
        o.plot(onset["time"], onset["p"], color=style.ONSET, lw=0.6)
        o.set_ylim(0, 1.05)
        o.set_yticks([0, 1])
        o.set_ylabel("Onset (p)")
    for a in traces:
        for site in sites.itertuples():
            a.axvspan(site.odor, site.leave, color=SITE_SHADE, alpha=0.18, lw=0)
    if speed_on_rate:
        s = r.twinx()
        s.spines["right"].set_visible(True)
        s.grid(False)
        s.set_ylabel("Velocity (cm/s)", color=style.SPEED)
        s.tick_params(axis="y", colors=style.SPEED)
    else:
        s = ax["speed"]
        s.set_ylabel("Velocity\n(cm/s)")
    s.plot(example["time"], example["speed"], color=style.SPEED, lw=0.7)
    keep = _inside(thermistor["time"].to_numpy(), t0, t1)
    w.plot(
        thermistor["time"].to_numpy()[keep],
        thermistor["signal"].to_numpy()[keep],
        color=colour,
        lw=0.4,
    )
    if inhalations is not None:
        shown = inhalations[_inside(inhalations, t0, t1)]
        w.plot(
            shown,
            np.full(len(shown), 1.02),
            "v",
            color=colour,
            ms=1.6,
            lw=0,
            clip_on=False,
            transform=w.get_xaxis_transform(),
        )
    w.set_yticks([])
    w.set_ylabel(label)
    w.spines["left"].set_visible(False)
    r.plot(example["time"], example["rate"], color=colour, lw=0.7)
    shown = licks[_inside(licks, t0, t1)]
    r.plot(
        shown,
        np.full(len(shown), 1.04),
        "|",
        color=style.SITES["Rewarded"],
        ms=2.5,
        mew=0.5,
        clip_on=False,
        transform=r.get_xaxis_transform(),
        label="Licks",
    )
    given = rewards[_inside(rewards, t0, t1)]
    r.plot(
        given,
        np.full(len(given), 1.13),
        "v",
        color=style.REWARD,
        ms=2.5,
        lw=0,
        clip_on=False,
        transform=r.get_xaxis_transform(),
        label="Reward",
    )
    handles = [
        *r.get_legend_handles_labels()[0],
        Rectangle((0, 0), 1, 1, color=SITE_SHADE, alpha=0.3, lw=0, label="Odor site"),
    ]
    traces[0].legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1, 1.04 if inhalations is not None else 1.0),
        ncol=3,
        fontsize=5.5,
        handlelength=1,
        borderaxespad=0.2,
    )
    r.set_xlim(t0, t1)
    r.set_ylim(0, 12)
    r.set_ylabel(
        "Respiration rate\n(Hz)", **({"color": colour} if speed_on_rate else {})
    )
    r.set_xlabel("Time (s)")
    fig.align_ylabels(traces)
    return fig


SITE_SHADE = "#7f7f7f"
"""Odor sites in a stretch of a session, whatever their outcome."""


def frame_rate_bins(
    grid_hz: float, low_hz: float, high_hz: float, ratio: float = 1.15
) -> NDArray:
    """Rate bin edges (Hz, rising) about *ratio* apart, each holding whole numbers of
    *grid_hz* intervals: events on an output grid give only rates ``grid_hz / k``, so
    even bins would alias (some hold two of them, others none)."""
    n = int(np.ceil(np.log(high_hz / low_hz) / np.log(ratio))) + 1
    frames = np.unique(np.round(np.geomspace(grid_hz / high_hz, grid_hz / low_hz, n)))
    edges = np.r_[frames - 0.5, frames[-1] + 0.5]
    return np.sort(grid_hz / edges)


def site_kind(stopped: bool, rewarded: bool) -> str:
    """An odor site's key in :data:`style.SITES`."""
    if not stopped:
        return "Skipped"
    return "Rewarded" if rewarded else "Unrewarded"


def rate_distribution_panel(
    width_mm: float,
    height_mm: float,
    rates: dict[str, NDArray],
    low: float = 2.0,
    high: float = 15.0,
    step: float = 0.25,
) -> Figure:
    """The distribution of breathing rate in each set (share of its breaths per Hz),
    overlaid, from *low* to *high* Hz; sets named as in :data:`style.SETS`. Breath
    intervals are whole samples of the thermistor, so rates take discrete values; bin
    edges sit halfway between them, about *step* Hz apart, so no bin is empty by
    construction."""
    fig, ax = style.panel(width_mm, height_mm, [["rates"]])
    a = ax["rates"]
    every = np.concatenate(list(rates.values()))
    sample = float(np.min(np.diff(np.unique(np.round(1.0 / every, 6)))))
    samples = np.round(1.0 / (np.arange(low, high + step / 2, step) * sample) - 0.5)
    bins = np.unique(1.0 / ((samples + 0.5) * sample))
    for name, values in rates.items():
        counts, _ = np.histogram(values, bins)
        density = 100 * counts / len(values) / np.diff(bins)
        a.stairs(
            density,
            bins,
            color=style.color(name),
            linewidth=0.9,
            label=style.set_label(name),
        )
    a.set_xlim(low, high)
    a.set_ylim(bottom=0)
    a.set_xlabel("Respiration rate (Hz)")
    a.set_ylabel("Cycles (% per Hz)")
    a.legend(loc="upper right", fontsize=6, handlelength=1)
    return fig


def corpus_rate_panel(
    width_mm: float,
    height_mm: float,
    rates: dict[str, NDArray],
    low: float = 2.0,
    high: float = 15.0,
    step: float = 0.25,
) -> Figure:
    """The distribution of breathing rate over the whole corpus (black) and of each
    animal (grey), as a share of breaths per Hz, binned as in
    :func:`rate_distribution_panel`."""
    fig, ax = style.panel(width_mm, height_mm, [["rates"]])
    a = ax["rates"]
    every = np.concatenate(list(rates.values()))
    sample = float(np.min(np.diff(np.unique(np.round(1.0 / every, 6)))))
    samples = np.round(1.0 / (np.arange(low, high + step / 2, step) * sample) - 0.5)
    bins = np.unique(1.0 / ((samples + 0.5) * sample))

    def density(values: NDArray) -> NDArray:
        counts, _ = np.histogram(values, bins)
        return 100 * counts / len(values) / np.diff(bins)

    for values in rates.values():
        a.stairs(density(values), bins, color="0.8", linewidth=0.4)
    a.stairs(density(every), bins, color=style.TRUTH, linewidth=1.0)
    a.plot([], [], color="0.8", lw=0.6, label=f"Each animal ({len(rates)})")
    a.plot([], [], color=style.TRUTH, lw=1.0, label="All")
    a.set_xlim(low, high)
    a.set_ylim(bottom=0)
    a.set_xlabel("Respiration rate (Hz)")
    a.set_ylabel("Cycles (% per Hz)")
    a.legend(loc="upper right", fontsize=6, handlelength=1)
    return fig


def corpus_animals_panel(
    width_mm: float, height_mm: float, sessions: pd.DataFrame
) -> Figure:
    """Sessions per animal, stacked by how many rigs the animal was recorded on; the
    corpus' totals written in the corner. *sessions* has a row per session with
    ``subject_id``, ``rig`` and ``date``."""
    fig, ax = style.panel(width_mm, height_mm, [["animals"]])
    a = ax["animals"]
    per = sessions.groupby("subject_id").agg(
        n=("rig", "size"),
        rigs=("rig", "nunique"),
        first=("date", "min"),
        last=("date", "max"),
    )
    groups = {
        "1 rig": per["rigs"] == 1,
        "2 rigs": per["rigs"] == 2,
        "3+ rigs": per["rigs"] >= 3,
    }
    bins = np.arange(0, per["n"].max() + 4, 3)
    shades = ["#c6dbef", "#6baed6", "#2171b5"]
    a.hist(
        [per.loc[k, "n"] for k in groups.values()],
        bins,
        stacked=True,
        color=shades,
        label=list(groups),
        edgecolor="white",
        linewidth=0.3,
    )
    span = (pd.to_datetime(per["last"]) - pd.to_datetime(per["first"])).dt.days
    a.text(
        0.97,
        0.6,
        f"{len(sessions)} sessions\n{len(per)} animals\n{sessions['rig'].nunique()} rigs\n"
        f"median {per['n'].median():g} sessions\nand {span.median():.0f} days per animal",
        transform=a.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
    )
    a.set_xlabel("Sessions per animal")
    a.set_ylabel("Animals")
    a.legend(loc="upper right", fontsize=5.5, handlelength=0.8)
    return fig


DISPLAY: dict[str, str] = {
    "Pixel (motion)": "Motion energy",
    "Pixel (PCA)": "Pixel PCA",
    "Pixel (SNR)": "Pixel SNR",
}
"""Method names as printed in the manuscript, where they differ from the run's."""


def _shown(name: str) -> str:
    return DISPLAY.get(name, name)


def trace_stack_panel(
    width_mm: float,
    height_mm: float,
    predictions: dict[str, dict],
    truth: dict,
    origin: float,
    t0: float,
    duration: float,
) -> Figure:
    """Every method's trace (one row each, in the given order) over the thermistor;
    the thermistor's inhales on top."""
    fig, ax = style.panel(width_mm, height_mm, [["stack"]])
    a = ax["stack"]
    t1 = t0 + duration
    tt = truth["times"] - origin
    kt = _inside(tt, t0, t1)
    step = 3.4
    a.plot(tt[kt], truth["signal"][kt], color=style.TRUTH, linewidth=0.8)
    a.text(t0 - 0.02 * duration, 0, "Thermistor", ha="right", va="center")
    for row, name in enumerate(predictions, start=1):
        y = -step * row
        p = predictions[name]
        tp = p["times"] - origin
        kp = _inside(tp, t0, t1)
        a.plot(tt[kt], truth["signal"][kt] + y, color="0.85", linewidth=0.6)
        a.plot(tp[kp], p["signal"][kp] + y, color=style.color(name), linewidth=0.8)
        a.text(t0 - 0.02 * duration, y, _shown(name), ha="right", va="center")
    onsets = truth["inhale"] - origin
    onsets = onsets[_inside(onsets, t0, t1)]
    for x in onsets:
        a.axvline(x, color="0.9", linewidth=0.5, zorder=0)
    a.set_xlim(t0, t1)
    # Room for a z-scored trace's full swing above the first row and below the last.
    a.set_ylim(-step * len(predictions) - 3.5, 3.5)
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.set_xticks([t0, t0 + 1])
    a.set_xticklabels(["", "1 s"])
    a.tick_params(axis="x", length=0)
    a.spines["bottom"].set_bounds(t0, t0 + 1)
    a.grid(False)
    return fig


def accuracy_time_panel(
    width_mm: float,
    height_mm: float,
    folds: pd.DataFrame,
    summary: pd.DataFrame,
    speed: pd.Series,
    names: Sequence[str],
    metric: str = "inhale",
    on_cpu: Sequence[str] = (),
    label_offsets: dict[str, tuple[float, float]] | None = None,
) -> Figure:
    """*metric* against inference speed (*speed*: frames per second), per method:
    each fold (dots, *folds*:
    ``method`` and ``value``) and the mean over folds with its 95% interval (*summary*:
    ``method``, ``mean``, ``low``, ``high``; no interval for a method with one split);
    open markers for the *on_cpu* methods."""
    fig, ax = style.panel(width_mm, height_mm, [["acc"]])
    a = ax["acc"]
    rng = np.random.default_rng(0)
    for method in names:
        if method not in speed or method not in set(summary["method"]):
            continue
        x = float(speed[method]) / 1000
        values = folds.loc[folds["method"] == method, "value"].to_numpy()
        colour = style.color(method)
        if len(values) > 1:
            a.scatter(
                x * np.exp(rng.uniform(-0.06, 0.06, len(values))),
                values,
                s=2.5,
                color=colour,
                alpha=0.45,
                lw=0,
            )
        row = summary[summary["method"] == method].iloc[0]
        a.errorbar(
            x,
            row["mean"],
            yerr=_interval(row),
            fmt="o",
            ms=3.2,
            color=colour,
            mfc="white" if method in on_cpu else colour,
            mew=0.9,
            elinewidth=0.9,
            capsize=0,
            zorder=4,
        )
        a.annotate(
            _shown(method),
            (x, row["mean"]),
            xytext=(label_offsets or {}).get(method, (4, 0)),
            textcoords="offset points",
            va="center",
            fontsize=5.5,
            color=colour,
        )
    a.set_xscale("log")
    a.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    a.xaxis.set_minor_formatter(ticker.NullFormatter())
    a.set_xticks([2, 5, 10, 20])
    a.set_xlim(2.2, 30)
    a.set_ylim(0, 1.02)
    a.set_xlabel("Inference (10³ frames/s)")
    a.set_ylabel(f"{METRIC_LABELS[metric]} (held-out animals)")
    return fig


def _interval(row: pd.Series) -> list[list[float]]:
    """``yerr`` of one summary row; none when it has no interval (one split)."""
    if pd.isna(row["low"]):
        return [[0.0], [0.0]]
    return [[row["mean"] - row["low"]], [row["high"] - row["mean"]]]


def lag_rate_panel(
    width_mm: float,
    height_mm: float,
    lags: pd.DataFrame,
    low: float = 2.0,
    high: float = 12.0,
    step_hz: float = 1.0,
    widest_ms: float = 50.0,
) -> Figure:
    """Onset lag (predicted minus true inhale) against the breath's rate: the share of
    each rate bin's lags at each lag (each column sums to 1), with each bin's mean lag
    (line) and its 95% bootstrap interval over folds (band; *lags* has a ``fold`` column);
    dotted, the scorer's tolerance."""
    fig, ax = style.panel(width_mm, height_mm, [["heat"]])
    a = ax["heat"]
    lag = lags["lag_s"].to_numpy() * 1000
    rate = lags["rate_hz"].to_numpy()
    keep = (
        np.isfinite(rate) & (rate >= low) & (rate < high) & (np.abs(lag) <= widest_ms)
    )
    lag, rate = lag[keep], rate[keep]
    x_bins = np.arange(low, high + step_hz / 2, step_hz)
    y_bins = np.arange(-widest_ms, widest_ms + 0.1, 2.5)
    counts, _, _ = np.histogram2d(rate, lag, [x_bins, y_bins])
    share = counts / counts.sum(axis=1, keepdims=True)
    image = a.pcolormesh(x_bins, y_bins, share.T, cmap="Blues", rasterized=True)
    centres = (x_bins[:-1] + x_bins[1:]) / 2
    per_fold = (
        pd.DataFrame(
            {
                "bin": np.digitize(rate, x_bins) - 1,
                "fold": lags["fold"].to_numpy()[keep],
                "lag": lag,
            }
        )
        .groupby(["bin", "fold"])["lag"]
        .mean()
        .groupby("bin")
        .apply(
            lambda v: pd.Series(
                dict(
                    zip(
                        ("mean", "low", "high"),
                        (v.mean(), *analysis.bootstrap_mean(v.to_numpy())),
                        strict=True,
                    )
                )
            )
        )
        .unstack()
    )
    x = centres[per_fold.index]
    a.fill_between(
        x,
        per_fold["low"],
        per_fold["high"],
        color=style.TRUTH,
        alpha=0.25,
        lw=0,
    )
    a.plot(x, per_fold["mean"], color=style.TRUTH, lw=0.9)
    for y in (-TOLERANCE_MS, TOLERANCE_MS):
        a.axhline(y, color="0.4", linewidth=0.6, linestyle=":")
    a.set_xlabel("Respiration rate (Hz)")
    a.set_ylabel("Onset lag (ms)")
    a.grid(False)
    bar = fig.colorbar(image, ax=a, fraction=0.06, pad=0.03, aspect=25)
    bar.set_label("Share of the rate bin")
    bar.outline.set_visible(False)
    return fig


def bland_altman_panel(
    width_mm: float,
    height_mm: float,
    windows: pd.DataFrame,
    name: str,
    min_hz: float = 2.0,
) -> Figure:
    """Breathing rate per window, predicted minus true against their mean, with the
    bias (solid) and 95% limits of agreement (dotted)."""
    fig, ax = style.panel(width_mm, height_mm, [["ba"]])
    a = ax["ba"]
    w = windows[(windows["method"] == name) & (windows["true_hz"] >= min_hz)]
    mean = (w["true_hz"] + w["pred_hz"]) / 2
    diff = w["pred_hz"] - w["true_hz"]
    bias, limit = float(diff.mean()), 1.96 * float(diff.std())
    lim = max(1.0, 1.5 * limit)
    # Counted rates come in steps of 1 / window, so bin the differences by that step.
    step = (
        float(np.min(np.diff(np.unique(np.round(diff, 6))))) if len(diff) > 1 else 0.1
    )
    y_bins = np.arange(-lim - step / 2, lim + step, step)
    x_bins = np.arange(np.floor(mean.min()), np.ceil(mean.max()) + 0.25, 0.25)
    *_, image = a.hist2d(
        mean,
        diff,
        [x_bins, y_bins],
        cmap="Blues",
        norm=mpl.colors.LogNorm(),
        rasterized=True,
    )
    for y, ls in ((bias, "-"), (bias - limit, ":"), (bias + limit, ":")):
        a.axhline(y, color="0.3", linewidth=0.6, linestyle=ls)
    a.set_ylim(-lim, lim)
    a.grid(False)
    bar = fig.colorbar(image, ax=a, fraction=0.08, pad=0.03, aspect=25)
    bar.set_label("Windows")
    bar.outline.set_visible(False)
    a.set_xlabel("Mean rate (Hz)")
    a.set_ylabel("Video − thermistor (Hz)")
    return fig


def sets_panel(
    width_mm: float,
    height_mm: float,
    summary: pd.DataFrame,
    names: Sequence[str],
    metrics: Sequence[str],
    sets: Sequence[str],
    labels: bool = True,
    legend: str | None = None,
    xlabel: str = "Held out",
    label_on: str | None = None,
) -> Figure:
    """Per metric (side by side, named on its y axis), each method's mean on each set
    with its 95% interval (*summary*: ``metric``, ``set``,
    ``method``, ``mean``, ``low``, ``high``), joined by a line, the method named at the
    last set if *labels* (on the *label_on* metric only, when given), or in a legend at
    *legend* (a location) on the last metric."""
    # Each metric's axes as wide as its x range, so the set names have the same room
    # whether or not the method names are written beside the last set.
    span = [
        len(sets) + (0.65 if labels and label_on in (None, m) else 0.0) for m in metrics
    ]
    fig, ax = style.panel(
        width_mm, height_mm, [list(metrics)], sharey=True, width_ratios=span
    )
    for metric in metrics:
        a = ax[metric]
        d = summary[summary["metric"] == metric].set_index(["set", "method"])
        ends = []
        for method in names:
            rows = [
                d.loc[(s_, method)] if (s_, method) in d.index else None for s_ in sets
            ]
            y = [r["mean"] if r is not None else np.nan for r in rows]
            colour = style.color(method)
            a.plot(range(len(sets)), y, "-", color=colour, lw=0.8, label=_shown(method))
            for i, r in enumerate(rows):
                if r is not None:
                    a.errorbar(
                        i,
                        r["mean"],
                        yerr=_interval(r),
                        fmt="o",
                        ms=2.5,
                        color=colour,
                        elinewidth=0.7,
                        capsize=0,
                    )
            ends.append((y[-1], method))
        # Names at the last set, Zephyr on top and the rest below it, nudged apart so
        # they do not overlap.
        ends.sort(key=lambda e: (e[1] != "Zephyr", -e[0]))
        placed: list[float] = []
        named = labels and label_on in (None, metric)
        for y, method in ends if named else []:
            y_text = min(y, placed[-1] - 0.045) if placed else y
            placed.append(y_text)
            a.text(
                len(sets) - 1 + 0.12,
                y_text,
                _shown(method),
                va="center",
                fontsize=6,
                color=style.color(method),
            )
        a.set_xticks(range(len(sets)), [style.set_label(s_, short=True) for s_ in sets])
        a.set_xlim(-0.25, len(sets) - 1 + (0.9 if named else 0.25))
        a.set_xlabel(xlabel)
        a.set_ylabel(METRIC_LABELS[metric])
    if legend is not None:
        ax[metrics[-1]].legend(
            loc=legend, fontsize=5.5, handlelength=1, labelspacing=0.2, borderpad=0.2
        )
    return fig


def camera_views_panel(
    width_mm: float,
    sessions: Sequence[dict],
    per_row: int = 2,
    gap_mm: float = 1.0,
    block_gap_mm: float = 5.0,
    label_mm: float = 3.5,
) -> Figure:
    """One block per session (*sessions*: ``label`` and ``views``, camera -> ``(image,
    box)`` with the box ``(x, y, w, h)`` in the image's pixels), its cameras side by
    side, *per_row* blocks to a row; each camera's crop box outlined in its colour, the
    camera named over the first row's blocks. Its height follows from *width_mm*."""
    cameras = list(sessions[0]["views"])
    first = sessions[0]["views"][cameras[0]][0]
    aspect = first.shape[0] / first.shape[1]
    columns = per_row * len(cameras)
    image_w = (
        width_mm - (per_row - 1) * block_gap_mm - per_row * (len(cameras) - 1) * gap_mm
    ) / columns
    image_h = image_w * aspect
    rows = -(-len(sessions) // per_row)
    height_mm = label_mm + rows * (label_mm + image_h) + (rows - 1) * gap_mm
    fig = Figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    for k, session in enumerate(sessions):
        row, block = divmod(k, per_row)
        top = label_mm + row * (label_mm + image_h + gap_mm) + label_mm
        for c, camera in enumerate(cameras):
            left = block * (len(cameras) * (image_w + gap_mm) - gap_mm + block_gap_mm)
            left += c * (image_w + gap_mm)
            a = fig.add_axes(
                (
                    left / width_mm,
                    1 - (top + image_h) / height_mm,
                    image_w / width_mm,
                    image_h / height_mm,
                )
            )
            image, (x, y, w, h) = session["views"][camera]
            a.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
            a.add_patch(
                Rectangle((x, y), w, h, fill=False, lw=0.8, ec=style.color(camera))
            )
            a.set_axis_off()
            if row == 0:
                fig.text(
                    (left + image_w / 2) / width_mm,
                    1 - (label_mm * 0.55) / height_mm,
                    camera.split()[0],
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=style.color(camera),
                )
            if c == 0:
                fig.text(
                    left / width_mm,
                    1 - (top - label_mm * 0.35) / height_mm,
                    session["label"],
                    ha="left",
                    va="center",
                    fontsize=6,
                )
    return fig


def paired_scatter_panel(
    width_mm: float,
    height_mm: float,
    wide: pd.DataFrame,
    reference: str,
    names: Sequence[str],
    metric: str,
) -> Figure:
    """Each row of *wide* (one clip; a column per method) as a dot: each of *names*
    against *reference* on *metric*, one colour per method, with the identity line; the
    share of clips each method scores higher than the reference in the legend."""
    fig, ax = style.panel(width_mm, height_mm, [["pairs"]])
    a = ax["pairs"]
    low = float(np.nanmin(wide[[reference, *names]].to_numpy()))
    lim = (max(0.0, low - 0.02), 1.0)
    a.plot(lim, lim, color="0.6", lw=0.6, ls=":", zorder=0)
    for name in names:
        both = wide[[reference, name]].dropna()
        above = float((both[name] > both[reference]).mean())
        a.plot(
            both[reference],
            both[name],
            "o",
            ms=1.8,
            mew=0,
            alpha=0.6,
            color=style.color(name),
            label=f"{_shown(name)} (higher in {100 * above:.0f}%)",
        )
    a.set_xlim(*lim)
    a.set_ylim(*lim)
    a.set_aspect("equal", adjustable="box")
    a.set_xlabel(f"{METRIC_LABELS[metric]}, {_shown(reference).lower()}")
    a.set_ylabel(f"{METRIC_LABELS[metric]}, other camera")
    a.legend(loc="lower right", fontsize=5.5, handletextpad=0.1, markerscale=1.5)
    return fig


def animals_panel(
    width_mm: float,
    height_mm: float,
    recordings: pd.DataFrame,
    names: Sequence[str],
    metric: str,
    set_name: str,
    unit: str = "subject",
    unit_label: str = "Animal",
    by_animal: str | None = None,
) -> Figure:
    """Each unit (value of the *unit* column) of one set, ranked by the first method,
    one marker per method. With *by_animal* (the column naming each unit's animal),
    an inset shows the first method's units again, one column per animal ranked by
    the animal's mean, its units joined, so the spread within a column is what the
    rig changes and the climb across columns what the animal does."""
    fig, ax = style.panel(width_mm, height_mm, [["animals"]])
    a = ax["animals"]
    rows = recordings[recordings["set"] == set_name]
    table = rows.groupby([unit, "method"])[metric].mean().unstack("method")
    table = table.sort_values(names[0])
    x = np.arange(len(table))
    for method in names:
        a.plot(
            x,
            table[method],
            "o",
            color=style.color(method),
            ms=3 if method == names[0] else 2.2,
            mfc="none" if method != names[0] else style.color(method),
            mew=0.7,
            label=_shown(method),
        )
    ranks = [0, *range(24, len(table), 25)]
    a.set_xticks(ranks, [str(r + 1) for r in ranks])
    a.set_xlim(-1, len(table))
    a.set_xlabel(f"{unit_label} (ranked)")
    a.set_ylabel(f"{METRIC_LABELS[metric]} (held-out animals)")
    if len(names) > 1:
        a.legend(loc="lower right", fontsize=6, handletextpad=0.2)
    if by_animal is not None:
        units = (
            rows[rows["method"] == names[0]]
            .groupby([by_animal, unit])[metric]
            .mean()
            .reset_index()
        )
        order = units.groupby(by_animal)[metric].mean().sort_values().index
        inset = a.inset_axes([0.33, 0.17, 0.64, 0.45])
        colour = style.color(names[0])
        for i, animal in enumerate(order):
            v = units.loc[units[by_animal] == animal, metric].to_numpy()
            if len(v) > 1:
                inset.plot([i, i], [v.min(), v.max()], color=colour, lw=0.6, alpha=0.6)
            inset.plot(np.full(len(v), i), v, "o", color=colour, ms=1.6, mew=0)
        inset.set_xticks([0, len(order) - 1], ["1", str(len(order))])
        inset.set_xlim(-1, len(order))
        inset.set_ylim(a.get_ylim())
        inset.tick_params(labelsize=5, length=2, pad=1)
        inset.set_xlabel("Animal (ranked)", fontsize=5, labelpad=0)
        inset.set_ylabel("")
    return fig


def rig_animal_panel(
    width_mm: float,
    height_mm: float,
    by_metric: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    method: str,
) -> Figure:
    """How far apart two units score when they share the animal but not the rig, and
    when they share the rig but not the animal, per metric (side by side, *by_metric*:
    ``(pairs, summary)`` from ``analysis.rig_animal_pairs`` and
    ``rig_animal_bootstrap``): the share of each kind's pairs in each bin of absolute
    difference (30 bins shared by the two kinds), one line per kind, and above it each
    kind's mean with its 95% interval."""
    metrics = list(by_metric)
    fig, ax = style.panel(width_mm, height_mm, [metrics])
    colours = [style.color(method), CROP_EDGE]
    for metric, (pairs, ci) in by_metric.items():
        a = ax[metric]
        kinds = [k for k in ci["kind"] if k != "difference"]
        summary = ci.set_index("kind")
        edges = np.linspace(0, pairs["difference"].max(), 31)
        top = 0.0
        for kind, colour in zip(kinds, colours, strict=True):
            x = pairs.loc[pairs["kind"] == kind, "difference"].to_numpy()
            share = np.histogram(x, edges)[0] / len(x)
            top = max(top, share.max())
            a.stairs(
                share,
                edges,
                color=colour,
                lw=0.9,
                label=f"{kind[0].upper()}{kind[1:]} ({len(x)})",
            )
        for k, (kind, colour) in enumerate(zip(kinds, colours, strict=True)):
            r = summary.loc[kind]
            level = top * (1.1 + 0.1 * k)
            a.plot([r["low"], r["high"]], [level, level], color=colour, lw=1.0)
            a.plot(r["mean"], level, "o", color=colour, ms=2.5)
        a.set_xlim(0, edges[-1])
        a.set_ylim(0, top * 1.27)
        a.set_xlabel(f"{METRIC_LABELS[metric]} difference")
    first = ax[metrics[0]]
    first.set_ylabel("Share of pairs")
    ax[metrics[-1]].legend(
        loc="upper right", bbox_to_anchor=(1, 0.8), fontsize=6, handlelength=1
    )
    return fig


def rate_band_panel(
    width_mm: float,
    height_mm: float,
    f1: pd.DataFrame,
    names: Sequence[str],
    share: pd.DataFrame | None = None,
    legend: bool = True,
    ylim: tuple[float, float] = (0.4, 1.02),
) -> Figure:
    """Inhale F1 per breathing-rate bin, one line per method (mean over folds, band its
    95% interval; *f1*: ``method``, ``bin``, ``mean``, ``low``, ``high``), over the
    share of breaths in each bin when *share* is given (from
    :func:`analysis.rate_share`: mean over units, band its 95% interval); *ylim*
    bounds the F1 axis."""
    rows = [["f1"], ["share"]] if share is not None else [["f1"]]
    ratios = [3, 1] if share is not None else [1]
    fig, ax = style.panel(width_mm, height_mm, rows, sharex=True, height_ratios=ratios)
    bins = list(dict.fromkeys(f1["bin"].astype(str)))
    x = np.arange(len(bins))
    for method in names:
        g = f1[f1["method"] == method]
        g = g.set_index(g["bin"].astype(str)).reindex(bins)
        colour = style.color(method)
        ax["f1"].plot(
            x, g["mean"], "-o", ms=2, lw=0.8, color=colour, label=_shown(method)
        )
        if g["low"].notna().any():
            ax["f1"].fill_between(
                x, g["low"], g["high"], color=colour, alpha=0.18, lw=0
            )
    bottom = ax["f1"]
    if share is not None:
        s = share.set_index(share["bin"].astype(str)).reindex(bins)
        bottom = ax["share"]
        bottom.plot(x, 100 * s["mean"], "-o", ms=1.6, lw=0.8, color=style.TRUTH)
        bottom.fill_between(
            x, 100 * s["low"], 100 * s["high"], color=style.TRUTH, alpha=0.15, lw=0
        )
        bottom.set_ylim(bottom=0)
        bottom.set_ylabel("Cycles (%)")
        fig.align_ylabels([ax["f1"], bottom])
    bottom.set_xticks(x, [b.split("-")[0] for b in bins])
    bottom.set_xlabel("Respiration rate (Hz)")
    ax["f1"].set_ylabel("Inhale F1\n(held-out animals)")
    ax["f1"].set_ylim(*ylim)
    if legend:
        ax["f1"].legend(loc="lower center", fontsize=6, handlelength=1, ncol=3)
    return fig


def pooled_rate_panel(
    width_mm: float,
    height_mm: float,
    overall: pd.DataFrame,
    delta: pd.DataFrame,
    cameras: Sequence[str],
    ylim: tuple[float, float] = (0.4, 1.02),
) -> Figure:
    """A network trained on several cameras, by breathing-rate bin: its inhale F1 over
    every camera's breaths (top; *overall*: ``bin``, ``mean``, ``low``, ``high``, band its
    interval over folds), and per camera its F1 minus that of the network trained on
    that camera alone (bottom; *delta*: ``camera``, ``bin``, ``delta``); *ylim* bounds
    the F1 axis, as in :func:`rate_band_panel`, so the two compare."""
    fig, ax = style.panel(
        width_mm, height_mm, [["f1"], ["delta"]], sharex=True, height_ratios=[3, 2]
    )
    bins = list(dict.fromkeys(overall["bin"].astype(str)))
    x = np.arange(len(bins))
    g = overall.set_index(overall["bin"].astype(str)).reindex(bins)
    colour = style.color("All cameras")
    ax["f1"].plot(x, g["mean"], "-o", ms=2, lw=0.8, color=colour)
    if g["low"].notna().any():
        ax["f1"].fill_between(x, g["low"], g["high"], color=colour, alpha=0.18, lw=0)
    ax["f1"].set_ylabel("Inhale F1\n(all cameras)")
    ax["f1"].set_ylim(*ylim)
    for camera in cameras:
        d = delta[delta["camera"] == camera]
        d = d.set_index(d["bin"].astype(str)).reindex(bins)
        ax["delta"].plot(
            x, d["delta"], "-o", ms=1.6, lw=0.7, color=style.color(camera), label=camera
        )
    ax["delta"].axhline(0, color="0.5", linewidth=0.5, linestyle=":")
    ax["delta"].set_ylabel("All minus\none camera")
    ax["delta"].set_xticks(x, [b.split("-")[0] for b in bins])
    ax["delta"].set_xlabel("Respiration rate (Hz)")
    fig.align_ylabels([ax["f1"], ax["delta"]])
    return fig


def rate_lines_panel(
    width_mm: float,
    height_mm: float,
    summary: pd.DataFrame,
    measures: dict[str, str],
    ylabel: str,
    scale: float = 1.0,
    legend: bool = True,
) -> Figure:
    """Measures per breathing-rate bin, one line each (mean over folds, band its 95%
    interval; *summary*: ``measure``, ``bin``, ``mean``, ``low``, ``high``), labelled by
    *measures* (measure -> label) and multiplied by *scale* (100 for percentages)."""
    fig, ax = style.panel(width_mm, height_mm, [["rate"]])
    a = ax["rate"]
    bins = list(dict.fromkeys(summary["bin"].astype(str)))
    x = np.arange(len(bins))
    for colour, (measure, label) in zip(
        style.series(2 * len(measures))[::2], measures.items(), strict=True
    ):
        g = summary[summary["measure"] == measure]
        g = g.set_index(g["bin"].astype(str)).reindex(bins)
        a.plot(x, scale * g["mean"], "-o", ms=2, lw=0.8, color=colour, label=label)
        if g["low"].notna().any():
            a.fill_between(
                x, scale * g["low"], scale * g["high"], color=colour, alpha=0.18, lw=0
            )
    a.set_xticks(x, [b.split("-")[0] for b in bins])
    a.set_xlabel("Respiration rate (Hz)")
    a.set_ylabel(ylabel)
    if legend and len(measures) > 1:
        a.legend(fontsize=5.5, handlelength=1, labelspacing=0.2, borderpad=0.2)
    return fig


def learning_curve_panel(
    width_mm: float,
    height_mm: float,
    curve: pd.DataFrame,
    colors: dict[str, str],
    metrics: Sequence[str] = ("inhale", "correlation"),
) -> Figure:
    """Scores against the number of training animals (log axis), one row per metric:
    *curve* has a row per run and ``group`` (e.g. new or known animals) with
    ``animals``, ``metric``, ``mean``, ``low``, ``high`` and ``full`` (whether the run
    used every session of its animals); runs with all sessions are joined by a line,
    the others drawn open; each group in its colour from *colors*. Where several runs
    share a size (draws of different animals), each is a small dot without interval
    and the line passes through their mean."""
    fig, ax = style.panel(width_mm, height_mm, [[m] for m in metrics], sharex=True)
    groups = list(dict.fromkeys(curve["group"]))
    for metric in metrics:
        a = ax[metric]
        for group in groups:
            d = curve[(curve["metric"] == metric) & (curve["group"] == group)]
            color = colors[group]
            repeated = d["full"] & (
                d.groupby(["full", "animals"])["mean"].transform("size") > 1
            )
            draws = d[repeated]
            a.plot(draws["animals"], draws["mean"], "o", color=color, ms=1.8)
            line = pd.concat(
                [
                    d[d["full"] & ~repeated][["animals", "mean"]],
                    draws.groupby("animals", as_index=False)["mean"].mean(),
                ]
            ).sort_values("animals")
            a.plot(line["animals"], line["mean"], color=color, lw=0.9, label=group)
            for full in (True, False):
                e = d[(d["full"] == full) & ~repeated]
                a.errorbar(
                    e["animals"] * (1 if full else 1.12),  # open marker beside
                    e["mean"],
                    yerr=[e["mean"] - e["low"], e["high"] - e["mean"]],
                    color=color,
                    linestyle="none",
                    marker="o",
                    markersize=3,
                    markerfacecolor=color if full else "white",
                    elinewidth=0.6,
                    capsize=0,
                )
        a.set_ylabel(METRIC_LABELS[metric])
    bottom = ax[metrics[-1]]
    bottom.set_xscale("log", base=2)
    ticks = sorted(curve["animals"].unique())
    bottom.set_xticks(ticks, [f"{t:g}" for t in ticks])
    bottom.minorticks_off()
    bottom.set_xlabel("Training animals")
    if len(groups) > 1:
        ax[metrics[-1]].legend(
            loc="center left", fontsize=5.5, handlelength=1, bbox_to_anchor=(0, 0.6)
        )
    return fig


def tolerance_panel(
    width_mm: float,
    height_mm: float,
    curve: pd.DataFrame,
    names: Sequence[str],
    widest_ms: float = 100.0,
) -> Figure:
    """Inhale F1 as the matching tolerance is relaxed, mean over folds with its 95%
    interval (*curve*: ``method``, ``tolerance_ms``, ``mean``, ``low``, ``high`` and
    ``errors``, the mean share of errors left), and below it the first method's misses
    and false inhalations as shares of true breaths (``misses`` and ``false``, each with
    ``low`` and ``high``). The first method's F1 and errors left are written at half the
    scorer's tolerance, at it, twice it and the widest."""
    fig, ax = style.panel(
        width_mm, height_mm, [["tol"], ["err"]], sharex=True, height_ratios=[3, 2]
    )
    a, e = ax["tol"], ax["err"]
    for k, method in enumerate(names):
        c = curve[curve["method"] == method].sort_values("tolerance_ms")
        colour = style.color(method)
        t = c["tolerance_ms"].to_numpy()
        a.plot(t, c["mean"], color=colour, label=_shown(method), lw=0.9)
        if c["low"].notna().any():
            a.fill_between(t, c["low"], c["high"], color=colour, alpha=0.18, lw=0)
        if k == 0:
            lines = []
            for at in (TOLERANCE_MS / 2, TOLERANCE_MS, 2 * TOLERANCE_MS, widest_ms):
                r = c.iloc[int(np.argmin(np.abs(t - at)))]
                a.plot(at, r["mean"], "o", ms=2.5, color=colour, zorder=4)
                share = (
                    f"{r['errors']:.0%}" if r["errors"] >= 0.1 else f"{r['errors']:.1%}"
                )
                lines.append(f"{at:g} ms: F1 {r['mean']:.2f}, {share} errors")
            a.text(
                0.98,
                0.04,
                "\n".join([_shown(method), *lines]),
                transform=a.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.5,
                color=colour,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
                zorder=5,
            )
            for kind, label in (("misses", "missed (FN)"), ("false", "false (FP)")):
                tint = style.ERRORS[label]
                e.plot(t, 100 * c[kind], color=tint, lw=0.9, label=label)
                e.fill_between(
                    t,
                    100 * c[f"{kind} low"],
                    100 * c[f"{kind} high"],
                    color=tint,
                    alpha=0.18,
                    lw=0,
                )
    for axis in (a, e):
        for at in (TOLERANCE_MS, 2 * TOLERANCE_MS):
            axis.axvline(at, color="0.5", linewidth=0.6, linestyle=":")
    e.set_xlabel("Tolerance (ms)")
    e.set_ylabel("Errors (%)")
    wide = c[c["tolerance_ms"] >= 10]
    e.set_ylim(0, 105 * max(wide["misses high"].max(), wide["false high"].max()))
    e.legend(loc="upper right", fontsize=6, handlelength=1)
    a.set_ylabel("Inhale F1")
    a.set_xlim(10, widest_ms)
    shown = curve[curve["method"].isin(names) & (curve["tolerance_ms"] >= 10)]
    a.set_ylim(np.floor(20 * shown["low"].fillna(shown["mean"]).min()) / 20, 1.01)
    if len(names) > 1:
        a.legend(loc="center right", fontsize=6, handlelength=1)
    return fig


def failure_panel(
    width_mm: float,
    height_mm: float,
    truth: dict,
    prediction: dict,
    errors: pd.DataFrame,
    origin: float,
    at: float,
    frames: Sequence[tuple[float, NDArray]],
    half: float = 1.5,
    legend: bool = True,
) -> Figure:
    """A window around one error: the crops around it (top, when *frames* has any), the thermistor and
    Zephyr's trace with true inhales, head events and errors marked, and the onset
    probability; *half* seconds either side, the legend under the trace if *legend*."""
    n = max(len(frames), 1)
    layout = [["trace"] * n, ["onset"] * n]
    ratios = [1.6, 0.8]
    if frames:
        layout, ratios = [[f"f{i}" for i in range(n)], *layout], [1.1, *ratios]
    fig, ax = style.panel(width_mm, height_mm, layout, height_ratios=ratios)
    t0, t1 = at - half, at + half
    tt, tp = truth["times"] - origin, prediction["times"] - origin
    kt, kp = _inside(tt, t0, t1), _inside(tp, t0, t1)
    tr = ax["trace"]
    tr.plot(tt[kt], truth["signal"][kt], color=style.TRUTH, label="Thermistor")
    tr.plot(
        tp[kp], prediction["signal"][kp], color=style.color("Zephyr"), label="Zephyr"
    )
    true_on = truth["inhale"] - origin
    head = prediction["head"] - origin
    mark_events(tr, true_on[_inside(true_on, t0, t1)], 3.6, style.TRUTH, size=2.5)
    mark_events(tr, head[_inside(head, t0, t1)], 3.0, style.color("Zephyr"), "^", 2.5)
    shown = errors[_inside(errors["time"].to_numpy() - origin, t0, t1)]
    for kind, g in shown.groupby("kind"):
        for xi in g["time"].to_numpy() - origin:
            for a in (tr, ax["onset"]):
                a.axvline(xi, color=style.ERRORS[kind], linewidth=0.6, alpha=0.8)
        tr.plot([], [], color=style.ERRORS[kind], lw=0.8, label=kind.split()[-1][1:-1])
    tr.set_xlim(t0, t1)
    tr.set_ylim(-5.2, 3.9)
    tr.set_ylabel("Respiration (z)")
    tr.set_xticklabels([])
    if legend:
        tr.legend(
            loc="lower left",
            ncol=4,
            fontsize=5,
            handlelength=0.8,
            columnspacing=0.8,
            bbox_to_anchor=(0, -0.08),
            frameon=False,
        )
    o = ax["onset"]
    o.plot(tp[kp], prediction["onset"][kp], color=style.ONSET)
    o.axhline(
        prediction.get("threshold", 0.5), color="0.6", linewidth=0.5, linestyle=":"
    )
    o.set_ylim(0, 1.05)
    o.set_yticks([0, 1])
    o.set_ylabel("Onset (p)")
    o.set_xlim(t0, t1)
    o.set_xlabel("Time (s)")
    o.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    for i, (ft, image) in enumerate(frames):
        f = ax[f"f{i}"]
        f.imshow(image, cmap="gray", vmin=0, vmax=255)
        f.set_title(f"{ft - at:+.2f} s", fontsize=6, weight="normal", pad=1.5)
        f.set_axis_off()
    return fig


# --- the benchmark --------------------------------------------------------------------


# --- failure modes --------------------------------------------------------------------


def fold_difference_panel(
    width_mm: float,
    height_mm: float,
    folds: pd.DataFrame,
    summary: pd.DataFrame,
    name: str,
    reference: str = "Zephyr",
) -> Figure:
    """*name* minus *reference* per test set (rows) for each metric (side by side),
    paired by cross-validation fold: each fold's difference (dots; *folds*: ``metric``,
    ``set``, ``fold``, ``value``) and their mean with its 95% bootstrap interval over folds
    (*summary*: ``metric``, ``set``, ``mean``, ``low``, ``high``)."""
    metrics = list(dict.fromkeys(summary["metric"]))
    sets = list(dict.fromkeys(summary["set"]))
    fig, ax = style.panel(width_mm, height_mm, [metrics], sharey=True)
    color = style.color(name)
    for metric in metrics:
        a = ax[metric]
        for i, set_name in enumerate(sets):
            f = folds[(folds["metric"] == metric) & (folds["set"] == set_name)]
            r = summary[(summary["metric"] == metric) & (summary["set"] == set_name)]
            a.plot(f["value"], np.full(len(f), i + 0.22), "o", color="0.6", ms=1.6)
            r = r.iloc[0]
            a.plot([r["low"], r["high"]], [i, i], color=color, lw=1.0)
            a.plot(r["mean"], i, "o", color=color, ms=3)
        a.axvline(0, color="0.3", lw=0.6)
        reach = np.ceil(
            100 * np.nanmax(np.abs(folds[folds["metric"] == metric]["value"]))
        )
        reach = max(reach, 1) / 100
        a.set_xlim(-1.15 * reach, 1.15 * reach)
        a.set_xticks([-reach, 0, reach], [f"{-reach:g}", "0", f"{reach:g}"])
        a.set_ylim(len(sets) - 0.4, -0.6)
        a.set_title(METRIC_LABELS[metric], fontsize=7)
        a.set_xlabel(f"Minus {reference}")
        a.grid(axis="y", visible=False)
    ax[metrics[0]].set_yticks(range(len(sets)), [style.set_label(s) for s in sets])
    return fig


# --- frame embeddings ------------------------------------------------------------------


# --- quality control ----------------------------------------------------------------------


def _family(name: str) -> str:
    from zephyr.quality_control.metrics import MEASURES

    return MEASURES[name][0]


# --- manuscript panels: quality control -------------------------------------------------


# --- where the network looks ---------------------------------------------------------

SALIENCY_CMAP = "magma"


def _heat(a: Axes, heat: NDArray, extent=None) -> Any:
    """*heat* over what *a* shows, scaled from 0 to its 99th percentile, that value
    written in the top right corner beside a swatch of the scale's top colour."""
    top = float(np.nanpercentile(heat, 99))
    shown = a.imshow(
        heat, cmap=SALIENCY_CMAP, alpha=0.65, extent=extent, vmin=0, vmax=top
    )
    text = f"{top:.2f}" if top >= 0.01 else f"{top:.1e}"
    # The swatch in the corner, the value to its left.
    a.add_patch(
        Rectangle(
            (0.90, 0.89),
            0.07,
            0.07,
            transform=a.transAxes,
            color=mpl.colormaps[SALIENCY_CMAP](1.0),
            lw=0,
        )
    )
    a.text(
        0.88,
        0.925,
        text,
        transform=a.transAxes,
        ha="right",
        va="center",
        fontsize=5,
        color="white",
    )
    return shown


def saliency_frames_panel(
    width_mm: float, height_mm: float, views: dict[str, tuple]
) -> Figure:
    """One whole frame per camera (title -> ``(image, box, heat)``), the network's crop
    box outlined in the camera's colour with *heat* (a map on the crop's pixels)
    inside it."""
    names = list(views)
    fig, ax = style.panel(width_mm, height_mm, [names])
    for name in names:
        image, box, heat = views[name]
        a = ax[name]
        a.imshow(image, cmap="gray")
        x, y, w, h = box
        _heat(a, heat, extent=(x, x + w, y + h, y))
        a.add_patch(Rectangle((x, y), w, h, fill=False, ec=style.color(name), lw=0.8))
        a.set_xlim(0, image.shape[1])
        a.set_ylim(image.shape[0], 0)
        a.set_title(name, fontsize=6, color=style.color(name))
        a.set_axis_off()
    return fig


def saliency_grid_panel(
    width_mm: float,
    height_mm: float,
    crops: dict[str, NDArray],
    maps: dict[str, dict[str, NDArray]],
) -> Figure:
    """A grid of maps on the crop's pixels: one column per camera (*crops*: camera ->
    the mean crop the maps are drawn over), one row per kind of map (*maps*: kind ->
    camera -> map), each scaled to its own range."""
    cameras, kinds = list(crops), list(maps)
    layout = [[f"{k}|{c}" for c in cameras] for k in kinds]
    fig, ax = style.panel(width_mm, height_mm, layout)
    for kind in kinds:
        for camera in cameras:
            a = ax[f"{kind}|{camera}"]
            a.imshow(crops[camera], cmap="gray")
            _heat(a, maps[kind][camera])
            a.set_xticks([])
            a.set_yticks([])
            for spine in a.spines.values():
                spine.set_visible(False)
            if kind == kinds[0]:
                a.set_title(camera, fontsize=6, color=style.color(camera))
            if camera == cameras[0]:
                a.set_ylabel(kind, fontsize=6)
    return fig


def attribution_profile_panel(
    width_mm: float,
    height_mm: float,
    profiles: dict[str, pd.DataFrame],
    half_s: float = 0.5,
) -> Figure:
    """Share of an inhale's attribution per frame against the frame's time from the
    inhale, one line per camera (*profiles*: camera -> ``offset_s``, ``mean``, ``low``,
    ``high``, in percent; band where there is an interval), within *half_s* s of it."""
    fig, ax = style.panel(width_mm, height_mm, [["profile"]])
    a = ax["profile"]
    for name, profile in profiles.items():
        p = profile[profile["offset_s"].abs() <= half_s]
        t = 1000 * p["offset_s"]
        colour = style.color(name)
        a.plot(t, p["mean"], color=colour, lw=0.9, label=name)
        if p["low"].notna().any():
            a.fill_between(t, p["low"], p["high"], color=colour, alpha=0.2, lw=0)
    a.axvline(0, color="0.5", lw=0.5, ls=":")
    a.set_xlabel("Frame time from the inhalation (ms)")
    a.set_ylabel("Share of attribution\nper frame (%)")
    a.set_xlim(-1000 * half_s, 1000 * half_s)
    if len(profiles) > 1:
        a.legend(fontsize=5.5, handlelength=1, labelspacing=0.2, borderpad=0.2)
    return fig


# --- refining on a new task and rig ---------------------------------------------------


def views_panel(width_mm: float, height_mm: float, views: dict[str, tuple]) -> Figure:
    """One whole frame per camera (title -> ``(image, box)``), the network's crop box
    outlined in the camera's colour."""
    names = list(views)
    fig, ax = style.panel(width_mm, height_mm, [names])
    for name in names:
        image, box = views[name]
        a = ax[name]
        a.imshow(image, cmap="gray")
        x, y, w, h = box
        a.add_patch(Rectangle((x, y), w, h, fill=False, ec=style.color(name), lw=1.0))
        a.set_title(name, fontsize=6, color=style.color(name))
        a.set_axis_off()
    return fig


def refined_traces_panel(
    width_mm: float,
    height_mm: float,
    rows: dict[str, dict],
    duration: float = 12.0,
    labels: dict[str, str] | None = None,
    notes: dict[str, str] | None = None,
    colours: dict[str, str] | None = None,
    corner: dict[str, str] | None = None,
    ncols: int = 1,
) -> Figure:
    """One row per camera or network (title -> ``times``, ``truth``, ``truth_inhale``,
    ``signal``, ``head``, ``start``): the thermistor and the network's trace (each
    z-scored) over *duration* s from ``start`` (s from the clip's start, the axis's
    time), the thermistor's (black) and the network's (the title's colour) inhales as
    ticks above; *labels* replace a title on its axis, *notes* go above its row,
    *colours* replace its colour and *corner* goes in its top right corner. Rows fill
    *ncols* columns, left to right."""
    names = list(rows)
    grid = [names[k : k + ncols] for k in range(0, len(names), ncols)]
    grid[-1] += ["."] * (ncols - len(grid[-1]))
    fig, ax = style.panel(width_mm, height_mm, grid, sharex=True, empty_sentinel=".")
    for name in names:
        r, a = rows[name], ax[name]
        origin = r["times"][0]
        t0, t1 = r["start"], r["start"] + duration
        t = r["times"] - origin
        keep = _inside(t, t0, t1)
        truth, signal = r["truth"][keep], r["signal"][keep]
        truth = (truth - truth.mean()) / (truth.std() or 1.0)
        signal = (signal - signal.mean()) / (signal.std() or 1.0)
        colour = (colours or {}).get(name) or style.color(name)
        a.plot(t[keep], truth, color=style.TRUTH, lw=0.6, label="Thermistor")
        a.plot(t[keep], signal, color=colour, lw=0.7, label="Zephyr")
        for events, y, c, m in (
            (r["truth_inhale"], 3.4, style.TRUTH, "v"),
            (r["head"], 2.8, colour, "^"),
        ):
            e = events - origin
            e = e[_inside(e, t0, t1)]
            a.plot(e, np.full(len(e), y), m, color=c, ms=1.8, lw=0)
        a.set_ylim(-3.2, 3.9)
        a.set_yticks([])
        a.set_ylabel((labels or {}).get(name, name), fontsize=6, color=colour)
        a.spines["left"].set_visible(False)
        if corner and name in corner:
            a.text(
                1.0,
                1.0,
                corner[name],
                transform=a.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.5,
                color=colour,
            )
        if notes and name in notes:
            a.set_title(notes[name], fontsize=5, loc="left", color=colour, pad=1)
    for name in grid[-1]:
        if name != ".":
            ax[name].set_xlabel("Time (s)")
    ax[names[-1]].set_xlim(t0, t1)
    return fig


def before_after_panel(
    width_mm: float,
    height_mm: float,
    table: pd.DataFrame,
    metrics: Sequence[str],
    cameras: Sequence[str],
) -> Figure:
    """Per metric (side by side), per camera, each session's score before and after
    (*table*: ``camera``, ``session``, ``metric``, ``Before``, ``After``) joined by a
    grey line, with the mean over sessions in each one's colour."""
    fig, ax = style.panel(width_mm, height_mm, [list(metrics)], sharey=True)
    for metric in metrics:
        a = ax[metric]
        for k, camera in enumerate(cameras):
            d = table[(table["camera"] == camera) & (table["metric"] == metric)]
            x = np.array([3 * k, 3 * k + 1])
            for _, row in d.iterrows():
                a.plot(x, [row["Before"], row["After"]], color="0.8", lw=0.5, zorder=1)
            for i, when in enumerate(("Before", "After")):
                a.plot(
                    x[i],
                    d[when].mean(),
                    "o",
                    ms=3.5,
                    color=style.color(when),
                    mec=style.color(camera),
                    mew=0.8,
                    zorder=3,
                )
        a.axhline(0, color="0.6", lw=0.5, ls=":")
        a.set_xticks(
            [3 * k + 0.5 for k in range(len(cameras))],
            [c.split()[0] for c in cameras],
        )
        a.set_xlim(-0.6, 3 * len(cameras) - 1.4)
        a.set_title(METRIC_LABELS[metric], fontsize=7)
    ax[metrics[0]].set_ylabel("Held-out animal")
    handles = [
        Line2D([], [], marker="o", ls="", color=style.color(w), ms=3.5, label=w)
        for w in ("Before", "After")
    ]
    ax[metrics[-1]].legend(handles=handles, loc="lower right", fontsize=5.5)
    return fig


def cost_panel(
    width_mm: float,
    height_mm: float,
    summary: pd.DataFrame,
    metrics: Sequence[str],
    sets: Sequence[str],
) -> Figure:
    """Per metric (side by side), after minus before on each set with its 95% interval
    (*summary*: ``metric``, ``set``, ``mean``, ``low``, ``high``)."""
    fig, ax = style.panel(width_mm, height_mm, [list(metrics)], sharey=True)
    for metric in metrics:
        a = ax[metric]
        d = summary[summary["metric"] == metric].set_index("set").loc[list(sets)]
        x = np.arange(len(sets))
        a.errorbar(
            x,
            d["mean"],
            yerr=[d["mean"] - d["low"], d["high"] - d["mean"]],
            fmt="o",
            ms=3,
            color=style.color("After"),
            elinewidth=0.8,
            capsize=0,
        )
        a.axhline(0, color="0.5", lw=0.6, ls=":")
        a.set_xticks(x, [style.set_label(s, wrap=True) for s in sets])
        a.set_xlim(-0.5, len(sets) - 0.5)
        a.set_xlabel("VR test set")
        a.set_title(METRIC_LABELS[metric], fontsize=7)
    ax[metrics[0]].set_ylabel("After minus before")
    return fig


# --- noisy frames ----------------------------------------------------------------------


# --- quality-control scores -----------------------------------------------------------


def quality_sets_panel(
    width_mm: float,
    height_mm: float,
    windows: pd.DataFrame,
    rows: Sequence[tuple[str, str]],
    truth: str | None = "truth_inhale_f1",
    marks: pd.DataFrame | None = None,
    colours: dict[str, str] | None = None,
    measures: dict[str, str] | None = None,
    per_row: int | None = None,
) -> Figure:
    """Each quality-control measure (*measures*, column -> label; by default
    :data:`style.QC_MEASURES`; then *truth*, when given; *per_row* to a row, all in one
    by default) over the 30-s *windows* of each
    row (*rows*: its value of ``row`` in *windows*, and its role, which colours it): each
    window a small faint point in the row's colour, each session's median (``session``)
    a black dot, and the mean of those medians a black tick under the shaded bar of its
    95% bootstrap interval over sessions (:func:`analysis.unit_medians`). *marks*
    (windows with ``row``, ``mark``, a label, and the measures) are drawn as labelled
    points on their row, labels above and below in turn. *colours* sets a row's colour
    by name instead of its role's."""
    colours = colours or {}
    measures = dict(measures or style.QC_MEASURES)
    if truth is not None:
        measures[truth] = "True inhale F1"
    columns = list(measures)
    per_row = per_row or len(columns)
    grid = [columns[i : i + per_row] for i in range(0, len(columns), per_row)]
    grid[-1] += ["."] * (per_row - len(grid[-1]))
    fig, ax = style.panel(width_mm, height_mm, grid, sharey=True)
    names = [name for name, _ in rows]
    jitter = np.random.default_rng(0)
    for column, label in measures.items():
        a = ax[column]
        for y, (name, role) in enumerate(rows if column in windows else []):
            rows_of = windows[(windows["row"] == name) & windows[column].notna()]
            if not len(rows_of):
                continue
            colour = colours.get(name, style.QC_ROLES[role])
            medians, summary = analysis.unit_medians(rows_of, column)
            a.plot(
                rows_of[column].to_numpy(),
                y + jitter.uniform(-0.25, 0.25, len(rows_of)),
                "o",
                ms=1.0,
                color=colour,
                mec="none",
                alpha=0.25,
                lw=0,
                zorder=1,
            )
            a.plot(
                medians.to_numpy(),
                y + jitter.uniform(-0.12, 0.12, len(medians)),
                "o",
                ms=2.0,
                color="k",
                mec="w",
                mew=0.3,
                lw=0,
                zorder=4,
            )
            if summary["n"] > 1:
                a.add_patch(
                    Rectangle(
                        (summary["low"], y - 0.3),
                        summary["high"] - summary["low"],
                        0.6,
                        facecolor=colour,
                        edgecolor="none",
                        alpha=0.35,
                        zorder=3,
                    )
                )
            a.plot(
                [summary["mean"]] * 2, [y - 0.3, y + 0.3], color="k", lw=1.0, zorder=3
            )
        if marks is not None:
            for i, m in enumerate(marks.itertuples()):
                y = names.index(m.row)
                x = getattr(m, column)
                a.plot(x, y, "o", ms=2.6, mfc="w", mec="k", mew=0.5, zorder=5)
                a.annotate(
                    m.mark,
                    (x, y),
                    xytext=(0, 3.5 if i % 2 == 0 else -3.5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if i % 2 == 0 else "top",
                    fontsize=5,
                )
        a.set_title(label, fontsize=6.5)
        a.tick_params(labelsize=5.5)
        a.set_ylim(len(rows) - 0.5, -0.5)
        a.margins(x=0.05)
    for line in grid:
        first = ax[line[0]]
        first.set_yticks(range(len(rows)))
        first.set_yticklabels(names, fontsize=5.5)
        for (_, role), tick in zip(rows, first.get_yticklabels(), strict=True):
            tick.set_color(
                colours.get(
                    tick.get_text(),
                    "k" if role in ("validation", "working") else style.QC_ROLES[role],
                )
            )
        for column in line[1:]:
            if column != ".":
                ax[column].tick_params(left=False, labelleft=False)
    return fig


def window_examples_panel(
    width_mm: float,
    height_mm: float,
    examples: Sequence[dict],
    threshold: float,
    colour: str,
) -> Figure:
    """Stretches of predictions without a thermistor, two per row (*examples*: each a
    ``title``, ``time`` (s from the stretch's start), the predicted ``signal`` and
    ``onset`` probability, and the ``head`` inhalation times): the trace with its
    inhalations as ticks above, and below it the onset probability with the
    *threshold* they were found at. A single example may carry ``frames`` (``time``,
    gray ``image``) and the crop ``box`` on them: the frames in a row above, each
    numbered at its time on the trace and the onset probability, with the box."""
    n = len(examples)
    cols = min(2, n)
    shots = examples[0].get("frames", []) if n == 1 else []
    if shots:
        # The frames' row as tall as a frame is wide; the rest to the trace and onset.
        k = len(shots)
        side = width_mm / k
        rest = max(height_mm - side, 1.0)
        fig, ax = style.panel(
            width_mm,
            height_mm,
            [[f"f{j}" for j in range(k)], ["t0"] * k, ["p0"] * k],
            height_ratios=[side, rest * 0.7, rest * 0.3],
        )
    else:
        rows = []
        for r in range((n + cols - 1) // cols):
            rows.append([f"t{r * cols + c}" for c in range(cols)])
            rows.append([f"p{r * cols + c}" for c in range(cols)])
        fig, ax = style.panel(
            width_mm, height_mm, rows, height_ratios=[1.5, 0.6] * (len(rows) // 2)
        )
    box = examples[0].get("box")
    for k, shot in enumerate(shots):
        a = ax[f"f{k}"]
        image = shot["image"]
        if box is not None:
            # A square around the box, three times its size (the paws and mouth
            # below the snout in view), inside the frame.
            x, y, w, h = box
            side = min(3 * max(w, h), *image.shape)
            left = int(np.clip(x + w / 2 - side / 2, 0, image.shape[1] - side))
            top = int(np.clip(y + h / 2 - side / 2, 0, image.shape[0] - side))
            image = image[top : top + int(side), left : left + int(side)]
            a.add_patch(
                Rectangle((x - left, y - top), w, h, fill=False, lw=0.6, ec=colour)
            )
        a.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
        a.set_axis_off()
        a.set_title(f"Frame {k + 1}  {shot['time']:.1f} s", fontsize=6, pad=2)
    for i, e in enumerate(examples):
        t, p = ax[f"t{i}"], ax[f"p{i}"]
        p.sharex(t)
        for k, shot in enumerate(shots):
            for a in (t, p):
                a.axvline(shot["time"], color="0.5", lw=0.4, ls=":")
            t.text(
                shot["time"] + 0.1,
                0.97,
                f"Frame {k + 1}",
                ha="left",
                va="top",
                fontsize=6,
                transform=t.get_xaxis_transform(),
            )
        t.tick_params(labelbottom=False)
        z = (e["signal"] - np.mean(e["signal"])) / np.std(e["signal"])
        t.plot(e["time"], z, color=colour, lw=0.5)
        t.plot(
            e["head"],
            np.full(len(e["head"]), 1.03),
            "v",
            color=colour,
            ms=1.1,
            lw=0,
            clip_on=False,
            transform=t.get_xaxis_transform(),
        )
        t.set_ylim(-3, 3)
        t.set_yticks([-3, 0, 3])
        t.set_title(e["title"], fontsize=6, loc="left", pad=5)
        p.plot(e["time"], e["onset"], color=style.ONSET, lw=0.5)
        p.axhline(threshold, color="0.6", lw=0.4, ls=":")
        p.set_ylim(0, 1.05)
        p.set_yticks([0, 1])
        p.set_xlim(e["time"][0], e["time"][-1])
        if i % cols == 0:
            t.set_ylabel("Respiration (z)")
            p.set_ylabel("Onset (p)")
        if i >= n - cols:
            p.set_xlabel("Time in the window (s)" if shots else "Time (s)")
    return fig
