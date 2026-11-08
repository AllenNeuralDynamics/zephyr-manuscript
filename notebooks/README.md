# Notebooks

[marimo](https://marimo.io) notebooks that make the manuscript's figures, tables and
values from the finished runs. They are plain `.py` files, so they diff and review like
code.

```bash
uv run marimo edit notebooks/manuscript_figures.py     # edit interactively
uv run marimo run notebooks/manuscript_figures.py      # serve as a read-only app
```

Every asset the notebooks read or write lives under one bucket, `<bucket>` below:
`bucket` in `[tool.zephyr-manuscript]` of the root `pyproject.toml`
(`s3://aind-scratch-data/vr-foraging`), or `ZEPHYR_BUCKET` when set
(`zephyr.benchmarks.storage`). Moving the assets is changing that one line, provided the
folders under it keep their names. The raw sessions on `aind-open-data` are a source, not
the paper's assets, and stay named as they are.

- `manuscript_figures.py`: every data panel of the manuscript's figures
  (`manuscript/figures/<figure>/<panel>.svg`, at the sizes in
  `manuscript/figures/layout.toml`), the tables it reads (`manuscript/tables/`) and
  every number its text cites (`manuscript/values.toml`, formatted as printed by
  `utils/values.py`; `main.typ` reads them as `#v("section.key")`), and the
  quality-control reports for Fig. 5 (`ZEPHYR_MANUSCRIPT_QC`). Which runs, under which
  labels, is the results manifest `packages/zephyr-benchmarks/configs/results/corpus-v3.toml`
  (`ZEPHYR_RESULTS` names another); the measures are `zephyr.benchmarks.analysis`. The
  runs come from the manifest's `root`, `<bucket>/zephyr-runs/` (read unsigned, no
  credentials: the bucket is public), downloaded once into `cache/runs/`; the clips from
  a local copy of the dataset and its feature cache (`ZEPHYR_TEST_ROOT`). Scores are the
  runs' own `evaluation.json`; traces are the runs' own `traces/` (and the pixel
  methods'), downloaded, where a run saved them, else predicted here (GPU, inference
  only); either way cached in `cache/walkthrough/`, with the tables, wall times and
  saliency maps computed from them. A network's inhales come from its onset head at a
  threshold chosen on its own validation clips, as training now chooses one
  (`ZEPHYR_HEAD_THRESHOLD=default` keeps 0.5); what is chosen, and why, is in
  `cache/walkthrough/head-validation/head-thresholds.json`.
- `checkpoint_analyses.py`: the two analyses of a finished network that no training run
  makes, run on this machine (GPU, inference only) from the all-camera network of cv_0 in
  `<bucket>/zephyr-manuscript-runs/`: the quality-control noise
  null (`zephyr-benchmarks probe` on 24 test-A face clips) and the dynamic-foraging animals
  scored before refining (`zephyr-benchmarks evaluate` on each fold's `df_test_a`). It reads
  everything from S3, and only what scoring reads (no local dataset copy): the clips'
  thermistors and timestamps from their datasets and their entries of the runs' cache
  (`zephyr-cache-v2`), no videos. Each
  runs only when its button is pressed, into `cache/analysis/`, with `analysis.json` and
  `provenance.json`; **Upload** copies them to `zephyr-manuscript-runs/analysis/<name>/`,
  refusing a folder that holds another checkpoint's.
- `corpus.py`: reruns the default `zephyr-dataset` pipeline
  (`packages/zephyr-dataset/configs/vr-foraging-sniff.toml`) up to `select` into a new
  folder, from the corpus's saved DocDB return, and compares the selection with the
  corpus (not `folds`, which would draw a new split: `raw/folds.parquet` is the record).
  It writes only that new folder, and only when its Rebuild button is pressed
  (about an hour; the raw thermistors and camera timestamps come from S3).
- `utils/style.py`: figure sizes, fonts and one fixed colour per method, variant
  and set of clips. Every figure calls `style.use_style()` and takes its colours
  from here.
- `utils/plots.py`: reusable figure components. `utils/runs.py`: downloads the
  manifest's runs, predicts and caches their traces, and caches the
  `zephyr.benchmarks.analysis` tables built from them.

Notebooks only read results (`runs/`, `benchmarks/`, `data/`); they never write
into them. Keep notebooks declarative: logic that a second notebook would need
goes into `utils/`.
