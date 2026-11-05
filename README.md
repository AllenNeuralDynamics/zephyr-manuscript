# zephyr-manuscript

The paper on [zephyr](https://github.com/AllenNeuralDynamics/zephyr), a network that
predicts a mouse's breathing from face video: the manuscript, the notebooks that make
its figures, tables and values, the benchmarks behind them, and the build of its corpus.

| Folder | What it is |
| --- | --- |
| [`manuscript/`](manuscript/) | The paper (Typst), in the Allen Institute preprint template (submodule `manuscript/template`) |
| [`notebooks/`](notebooks/README.md) | marimo notebooks that read the results and draw the figures, tables and `values.toml` |
| [`packages/zephyr-benchmarks/`](packages/zephyr-benchmarks/README.md) | `zephyr.benchmarks`: the comparison methods, the paper's analyses, experiment configs and results manifests |
| [`packages/zephyr-dataset/`](packages/zephyr-dataset/README.md) | `zephyr.corpus`: builds the corpus of VR-foraging sessions from one spec (Allen-internal) |
| [`scripts/`](scripts/) | `demo_video.py`: the demo video of three camera views and their networks' traces, from the results manifest's runs |

zephyr comes from the zephyr repository, pinned to one commit in `pyproject.toml`.

```bash
git clone --recurse-submodules https://github.com/AllenNeuralDynamics/zephyr-manuscript
cd zephyr-manuscript
uv sync --all-packages --extra cpu     # or --extra gpu for CUDA
uv run pytest
```

Build the PDF from the repository root:

```bash
typst compile --font-path manuscript/template/fonts \
  --input version="$(git describe --always --dirty --abbrev=40 --exclude='*')" \
  manuscript/main.typ manuscript.pdf
```
