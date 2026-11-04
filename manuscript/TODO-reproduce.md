# Reproducing the paper: retraining and publishing

Every network and experiment the paper reports is written down as TOML files before
anything runs, run from one tagged commit in the Code Ocean capsule, and published as a
small set of assets. Started 2026-11-03.

## 0. Decisions (2026-11-04)

- [x] Repository: split. A paper repository holds the manuscript, notebooks, results
      manifests and zephyr-benchmarks, pinned to a zephyr release; zephyr keeps the
      package, docs and web app; the corpus build stays where it is.
- [x] Bucket: `s3://aind-scratch-data` for now; moved to aind-open-data by hand afterwards.
- [x] Seeds: one per fold (seed 17) for every run.
- [x] PhysFormer and RhythmFormer: retrained on the five folds, like the other methods.
- [x] Variants not in the text (more augmentation, rate-balanced, 192-px crops): dropped.
- [x] 30-Hz input: retrained on the five folds with the rest.
- [x] Dynamic foraging: one fold (cv_0), leave-one-out over its 3 animals (3 refined
      networks, each scored on the animal it left out).
- [x] Training-set size: as Methods says, on cv_4: 3 draws each of 4, 8 and 16 animals and
      of "sessions of 8", and 5 draws of 2 animals (17 runs).
- [x] Learning rates of the comparison networks: no sweep; keep the current values and
      state where each comes from in Supplementary Note 2.
- [x] Statistics: keep the current bootstrap (over folds; over units for single-fold
      analyses).
- [x] No run uses the dev split: the published split goes from the paper.
- [x] Application and quality control: the all-camera network of cv_0.
- [x] Receptive fields: cv_0 networks of the three cameras, computed on this machine.
- [x] Timing: on this machine (i9-13900K, RTX 4060 Ti), as Methods says.
- [x] Test set C (359-Hz sessions): dropped from the paper; the folds are unchanged.
- [x] No ONNX export for the web app as part of the paper.

## 1. Assets

| Asset | Access | Holds |
|---|---|---|
| Corpus | public | `clips/`, `clips.toml` (boxes), `thumbnails/`, `sessions.parquet`, READMEs, and `raw/` (the build record: per-session quality control, `folds.parquet`, the clip selection, `[[review]]` decisions) |
| Other sessions | public | the dynamic-foraging sessions and the VR-foraging sessions without a thermistor: video, clip lists with their checked boxes, thermistor where there is one |
| Trained networks | public | each run's `best.pt` (with its onset threshold), `config.json`, `history.json`, `best_per_clip.json`, `qc-reference.json` and its experiment TOML |
| Results | public | inference outputs (traces, `evaluation.json`, quality-control reports, predictions of the other sessions) and the notebook's outputs (figures, tables, `values.toml`) |
| Caches | private | the preprocessed crops of every camera and frame rate, rebuilt from the corpus and its boxes |

All in `s3://aind-scratch-data` until they move to aind-open-data.

- [ ] Publishing `raw/` changes the rule in AGENTS.md (the dataset's top level lists only
      `clips/`, `clips.toml`, `thumbnails/`, `sessions.parquet` and the READMEs): update it
      and the corpus README.
- [ ] Hand decisions are inputs, kept in the assets, never only on this machine: crop boxes
      of the other sessions (now `D:/ToBePredicted`, `D:/zephyr-df-dataset`), the
      corpus's `[[review]]`, and the frames picked for Fig. 5b (in the notebook).
- [ ] Upstream sources the figures read, named in a results manifest instead of in the
      notebook: the processed assets on aind-open-data of the Fig. 1b–c session (now
      hard-coded) and of the sessions without a thermistor (already in `[application]`).
      Cite them in the paper; the notebook still downloads them to rebuild the figures.
- [ ] The results asset is built with the caches mounted: Fig. 1e, the receptive fields
      and the timing read the frames. Data availability says the caches are rebuilt from
      the corpus, not distributed.
- [ ] Cache keys hold the clips' absolute paths (`/data/zephyr-dataset/...`): anyone
      rebuilding the caches mounts the corpus there or sets `ZEPHYR_PATH_MAP`; say so in
      the corpus README.
- [ ] DOIs and licences, once the assets are on aind-open-data (Nature Methods cites data
      and code with a DOI in the reference list).

## 2. Repository split

- [ ] New paper repository: `manuscript/` (with its template submodule), `notebooks/`,
      `packages/zephyr-benchmarks/` (configs and results manifests included), its tests
      and CI (manuscript PDF build).
- [ ] zephyr keeps `packages/zephyr`, `packages/zephyr-dataset`, `docs/`, `web/`; its
      boundary test and AGENTS.md lose what moved.
- [ ] The paper repository pins a zephyr release; the capsule pins the paper repository.
- [ ] Both AGENTS.md files say what lives where.

## 3. Pre-registration

The pre-registration is the TOML files we already use, written in full and tagged before
the first run:

- [ ] the capsule's matrices (`matrices/*-v5.toml`) and experiment files
      (`code/experiments/`), each copied from the one it repeats, never edited;
- [ ] the results manifest (copied from `corpus-v3.toml` under a new name), listing every
      run the paper reads and every upstream asset, with a comment naming the figure or
      table each feeds;
- [ ] one tag covering the commits of zephyr, the paper repository and the capsule.

## 4. The Code Ocean capsule

Runs are trained and scored by the capsule built from `zephyr-compute-sandbox` (capsule
`bce40323-…`, `uv run launch.py matrices/<matrix>.toml [--smoke] --submit`).

- [ ] Pin the tagged commits. The capsule pins zephyr and zephyr-benchmarks at `a482e19`
      in `environment/pyproject.toml`, 63 commits behind this branch. Move the pin, update
      `environment/uv.lock`, rebuild the image, sync the capsule from git, and check that
      the first lines of a run's log show the new commit.
- [ ] Move into the capsule what runs on this machine today: predictions with quality
      control (held-out animals, dynamic foraging before and after refining, sessions
      without a thermistor on all three cameras), the noise null and the dynamic-foraging
      scores. Scoring, receptive fields, timing and the notebook stay on this machine.
- [ ] Data assets, which you manage: the corpus (`/data/zephyr-dataset`), the face cache
      (`/data/zephyr-cache`), the side, front and 30-Hz caches, and the other sessions.
      Check that each holds the current corpus before the first run.
- [ ] `--smoke` before each full run; read every run's log (Code Ocean reports a crashed
      run as succeeded).
- [ ] Record the seed, GPU and wall time of every run (Methods note "seeds, GPU, time per
      run").
- [ ] The `-v3` matrices scored the onset head at 0.5, and the notebook re-chose each
      threshold on validation; new runs choose it in training. Check that the two give the
      same threshold before comparing new scores with old ones.

## 5. Runs

All on one commit, seed 17. First wave (independent):

| Experiment (paper) | Repeats | Folds | Runs |
|---|---|---|---|
| Zephyr, face camera (Figs. 1–2, Table 1) | `full-gray-cv-v3` | cv_0–cv_4 | 5 |
| PhysNet, TS-CAN | `physnet-cv-v3`, `tscan-cv-v3` | cv_0–cv_4 | 10 |
| PhysFormer, RhythmFormer | `physformer-v3`, `rhythmformer-v3` (dev) | cv_0–cv_4 | 10 |
| Facemap, pixel methods (no training) | `pixel-facemap-cv-v3` | cv_0–cv_4 | 5 |
| 30-Hz input (Fig. S3) | `full-gray-30hz-cv-v3` | cv_0–cv_4 | 5 |
| Side and front cameras (Fig. 3d–e, Fig. S4, Table S4) | `full-gray-{side,front}-720-cv-v4` | cv_0–cv_4 | 10 |
| All cameras (Fig. 3f) | `full-gray-all-cv-v4` | cv_0–cv_4 | 5 |
| Training-set size (Fig. 3c): 4, 8, 16 animals, "sessions of 8" | `full-gray-data-size-cv4-v3` | cv_4, 3 draws each | 12 |
| Training-set size: 2 animals | `full-gray-data-size-small-v3` (dev) | cv_4, 5 draws | 5 |

Second wave (needs the all-camera network of cv_0):

| Experiment (paper) | Repeats | Runs |
|---|---|---|
| Refined on dynamic foraging (Fig. 3g–i, Fig. S8), leave-one-out | `finetune-df-cv-v4` | 3 |
| Quality control: references, reports (held-out animals, dynamic foraging before and after refining), noise null 16/32/64 (Figs. 4–5, S9) | local, `D:/zephyr-qc-probe/v5` | in the capsule |
| Sessions without a thermistor, three cameras (Fig. 5, S10) | local, `D:/ToBePredicted` | in the capsule |

Dropped: `full-gray-sched-v3` and its variants (more augmentation, rate-balanced,
192-px crops), the dev split of every method.

## 6. After the runs

- [ ] Score every run; fill the results manifest.
- [ ] On this machine: receptive fields of the cv_0 face, side and front networks
      (Fig. S7), and timing (Fig. 2b, Table 1, Table S2).
      `notebooks/cache/walkthrough/timing.json` is keyed by method name, so delete it
      first; clear the other caches there too (predictions, error tables, head-threshold
      scores).
- [ ] Table S2 (network settings) and the timing read the dev-split runs (`[[method]]`);
      point them at the fold runs.
- [ ] Examples chosen against the current runs, to choose again:
  - [ ] Fig. 1b–c session (863680, picked as nearest the median scores of the cv_2 run);
  - [ ] Fig. 1e excerpt (269–277 s of the median clip, picked by hand);
  - [ ] Fig. 5b: the lowest window and its four frames, by eye.
- [ ] Text:
  - [ ] remove the "fold cv_0 only so far" notes (side and front cameras, all-camera
        network);
  - [ ] receptive fields: Methods and the Fig. S7 caption say the face network of each
        fold; make it cv_0 for every camera;
  - [ ] dynamic foraging: Methods describes one held-out animal; describe the
        leave-one-out (3 networks);
  - [ ] remove the published split from Data splits, the split-size table ("6 splits"),
        the onset threshold sentence (`threshold.published`), the Table 1 and Table S3
        captions ("Methods run on one split…"), Statistics, and the 2-animal draws
        ("trained on the published split");
  - [ ] Supplementary Note 2: where each learning rate comes from (replaces the note);
  - [ ] Statistics: remove the note on resampling animals.
- [ ] Regenerate `values.toml`, every figure and table into the results asset; check every
      quoted number; fill the `#tbd[...]` placeholders.
- [ ] Compare the new scores with v3 (within the intervals?), after checking no scoring
      rule changed.
- [ ] Data and code availability: the assets of section 1, their DOIs, and what is rebuilt
      rather than distributed (the caches).
