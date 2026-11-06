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

An editable Word copy (for collaborators who comment in Word) is made from the HTML export
through Pandoc; CI builds it and links it from the web page:

```bash
typst compile --features html --format html --font-path manuscript/template/fonts   manuscript/main.typ manuscript.html
python scripts/build_docx.py manuscript.html manuscript.docx --font-path manuscript/template/fonts
```

It needs `pandoc` on the path. It is a convenience copy: the PDF is the typeset version.
