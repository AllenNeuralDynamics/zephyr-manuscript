// Supplementary Fig. 10: the corpus selection and split. Build:
// typst compile figures/figs10.typ (from manuscript/). Panels: layout.toml [figs10], drawn by
// notebooks/manuscript_figures.py.

#import "lib.typ": assemble

// Panels that exist as figs10/<name>.svg; the rest are drawn as placeholders.
#let files = ("a", "b")

#let body = assemble("figs10", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
