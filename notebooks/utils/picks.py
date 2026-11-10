"""The examples the paper's figures show that were picked by hand, in one place: both
notebooks read them, ``analyses.py`` to extract what the figures need of each (frames,
behaviour, views), ``manuscript_figures.py`` to draw it. Each was picked on an earlier
set of runs: look at it again on the runs the paper reports, and change it here (then
run the analyses again) if another example shows the point better."""

TASK_ASSET = "863680_2026-09-08_19-35-38_processed_2026-09-23_03-47-07"
"""Fig. 1b, c: the processed asset (running speed, sites, licks) of the session shown,
one of a new animal (863680 is in test A of ``TASK_FOLD``) whose correlation and head
inhale F1 were nearest test A's medians."""
TASK_FOLD = "cv_2"
"""The fold that holds ``TASK_ASSET``'s animal out, whose network predicts it."""

EXAMPLE_CLIP = "behavior_828425_2026-01-29_21-14-59_span2_FaceCamera"
"""Fig. 1d, e: the example clip, a test-A clip of the single network (``runs.SINGLE``)
whose trace correlation was nearest the median of its test-A clips, kept for the
breathing rates it goes through."""

EXCERPT = (271.0, 4.0)
"""Fig. 1e: the excerpt of the example clip shown, start (s from the clip's start) and
length."""

STACK_CLIP = "behavior_806527_2026-01-12_19-18-42_span1_FaceCamera"
"""Fig. 3a: the clip shown, of an animal held out by fold cv_0, whose correlation
averaged over the methods was nearest the median of its new-animal clips."""

STACK_EXCERPT = (167.0, 6.0)
"""Fig. 3a: the excerpt of ``STACK_CLIP`` shown, start (s from the clip's start) and
length: 6 s where the thermistor's breathing rate halves."""

APPLICATION_SHOTS_S = (3.0, 6.45, 8.9, 10.5)
"""Fig. 5b: frames of the window shown (the lowest window of the sessions without a
thermistor), s from its start, chosen by eye: the face at rest, two paw strokes over the
snout and a lick."""

SALIENCY_SESSION = "behavior_806527_2026-01-22_18-42-08"
"""Fig. S6a: the new animal of fold cv_0 whose maps are shown from each camera (its
maps close to the average on every camera, with a clean side view)."""

VIEW_SESSIONS = (
    "behavior_810750_2025-11-05_20-47-56",
    "behavior_828418_2025-11-12_23-52-42",
    "behavior_811026_2025-11-25_21-25-01",
    "behavior_815104_2025-11-26_22-46-47",
    "behavior_828425_2025-11-28_22-26-13",
    "behavior_828425_2025-12-19_16-43-15",
    "behavior_806527_2026-01-23_19-16-56",
    "807093_2026-02-10_17-02-30",
    "841310_2026-02-19_21-04-00",
    "828424_2026-02-21_00-33-08",
    "828415_2026-02-28_00-20-42",
    "823993_2026-03-02_21-02-53",
    "863680_2026-06-26_16-20-33",
    "863698_2026-07-22_16-58-02",
    "864848_2026-09-18_19-51-33",
    "879294_2026-09-29_20-38-40",
)
"""Fig. S4: a representative subset of sessions, its three cameras side by side."""
