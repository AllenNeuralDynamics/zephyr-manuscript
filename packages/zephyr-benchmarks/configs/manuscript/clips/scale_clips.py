"""Write a copy of a dataset clip list at another working resolution or frame rate.

The dataset's lists place 96x96 boxes in a 360x270 frame (half the cameras' 720x540);
``--factor 2`` multiplies ``target_size`` and every box by 2, so the same crops are made
at the cameras' own resolution (192x192). ``--select-fs 30`` sets the rate of the frames
the CNN sees (zephyr still predicts and is scored at 60 Hz). ``--camera SideCamera`` keeps
only that camera's clips, used (the dataset lists them ``selection = "exclude"``, as it does
every camera but the face's); ``--all-cameras`` keeps every camera, used. Everything else
is kept. The copy's paths are
absolute under the dataset mount, so it may live anywhere (``code/clips/``) and gives the
cache keys a list in the dataset would: ``--root`` is where the source's relative paths
point on Code Ocean, whatever folder the source was read from.

    python scale_clips.py /data/zephyr-dataset/clips.toml clips/clips-192.toml --factor 2
    python scale_clips.py clips.toml clips/clips-30hz.toml --factor 1 --select-fs 30
    python scale_clips.py clips.toml clips/clips-side.toml --factor 1 --camera SideCamera
    python scale_clips.py clips.toml clips/clips-all.toml --factor 1 --all-cameras
"""

import argparse
import tomllib
from pathlib import Path, PurePosixPath

import tomli_w

DATASET = Path("/data/zephyr-dataset")
"""Where Code Ocean mounts the dataset (the capsule's data asset)."""

PATHS = ("video", "timestamps", "thermistor")


def scale(
    data: dict,
    factor: int,
    root: PurePosixPath,
    select_fs: float | None = None,
    camera: str | None = None,
    all_cameras: bool = False,
) -> dict:
    """*data* (a clip list) with ``target_size`` and boxes times *factor*, paths under
    *root*, ``select_fs`` if given, and only *camera*'s clips (used) if given, or every
    camera's (used) with *all_cameras*."""
    out = {**data, "preprocess": {**data["preprocess"]}}
    out["preprocess"]["target_size"] = [
        v * factor for v in data["preprocess"]["target_size"]
    ]
    if select_fs is not None:
        out["preprocess"]["select_fs"] = select_fs
    clips = []
    for clip in data["clip"]:
        if camera is not None:
            if not clip["video"].removesuffix(".mp4").endswith(f"_{camera}"):
                continue
            clip = {k: v for k, v in clip.items() if k != "selection"}
        elif all_cameras:
            clip = {k: v for k, v in clip.items() if k != "selection"}
        clip = dict(clip)
        for key in PATHS:
            if isinstance(clip.get(key), str):
                clip[key] = (root / clip[key]).as_posix()
        if "box" in clip:
            clip["box"] = [v * factor for v in clip["box"]]
        clips.append(clip)
    out["clip"] = clips
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "source", type=Path, help="a dataset clip list (or a local copy of one)"
    )
    p.add_argument("out", type=Path)
    p.add_argument("--factor", type=int, default=2)
    p.add_argument(
        "--select-fs",
        type=float,
        help="the CNN's frame rate, Hz (default: the source's)",
    )
    which = p.add_mutually_exclusive_group()
    which.add_argument(
        "--camera", help="keep only this camera's clips, used (default: all, as listed)"
    )
    which.add_argument(
        "--all-cameras", action="store_true", help="keep every camera's clips, used"
    )
    p.add_argument(
        "--root",
        default=DATASET.as_posix(),
        help="the folder the source's paths are relative to",
    )
    a = p.parse_args()
    with a.source.open("rb") as handle:
        data = tomllib.load(handle)
    if "derive" in data or any(not c.get("video") for c in data["clip"]):
        raise SystemExit(f"{a.source}: only lists naming each clip's files are handled")
    scaled = scale(
        data, a.factor, PurePosixPath(a.root), a.select_fs, a.camera, a.all_cameras
    )
    if not scaled["clip"]:
        raise SystemExit(f"{a.source}: no {a.camera} clips")
    unboxed = [c["video"] for c in scaled["clip"] if "box" not in c]
    if (a.camera is not None or a.all_cameras) and unboxed:
        raise SystemExit(
            f"{len(unboxed)} {a.camera} clips have no box, e.g. {unboxed[0]}"
        )
    w, h = scaled["preprocess"]["target_size"]
    sizes = {tuple(c["box"][2:]) for c in scaled["clip"] if "box" in c}
    rate = f", select_fs {a.select_fs:g} Hz" if a.select_fs is not None else ""
    rate += f", only {a.camera}" if a.camera is not None else ""
    rate += ", every camera used" if a.all_cameras else ""
    header = (
        f"# {a.source.name} with target_size and boxes x{a.factor} ({w}x{h} frame, boxes "
        f"{' '.join('x'.join(map(str, s)) for s in sorted(sizes))}){rate}, paths under {a.root}.\n"
        f"# Written by scale_clips.py (never edit by hand): write it again when the source "
        f"changes.\n\n"
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(header + tomli_w.dumps(scaled))
    print(
        f"wrote {a.out}: {len(scaled['clip'])} clips, {len(sizes)} box size(s) {sorted(sizes)}"
    )


if __name__ == "__main__":
    main()
