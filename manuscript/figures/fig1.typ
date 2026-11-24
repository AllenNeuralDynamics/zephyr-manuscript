// Fig. 1: The data set. Build: typst compile figures/fig1.typ (from manuscript/).
// Panels: layout.toml [fig1]; data panels from notebooks/manuscript_figures.py, c is
// the corpus table below (numbers from values.toml).

#import "lib.typ": assemble

// Panels that exist as fig1/<name>.svg; the rest are drawn as placeholders.
#let files = ("a", "b", "d")

// The corpus, from the numbers the figure notebook writes.
#let values = toml("../values.toml")
#let v(key) = {
  let (section, name) = key.split(".")
  values.at(section, default: (:)).at(name, default: "[" + key + "]")
}
#let corpus = {
  set text(font: "Arial", size: 6pt)
  set par(justify: false)
  table(
    columns: (auto, 1fr),
    inset: (x: 3pt, y: 2.4pt),
    stroke: none,
    align: (left, left),
    table.hline(stroke: 0.5pt),
    table.header([*Corpus*], []),
    table.hline(stroke: 0.3pt),
    [Animals], [#v("corpus.animals") head-fixed mice],
    [Sessions], [#v("corpus.sessions"), two 300-s clips each],
    [Video], [≈#v("corpus.hours") h per camera],
    [Rigs, span], [#v("corpus.rigs") rigs over #v("corpus.months") months],
    [Cameras], [face, side and front; 240 or 360 Hz, read at 60 Hz],
    [Thermistor], [nasal epithelium, 200–250 Hz],
    [Respiration], [#v("rates.slow")% at 2–4 Hz, #v("rates.medium")% at 4–8 Hz, #v("rates.fast")% at 8–15 Hz],
    table.hline(stroke: 0.3pt),
    [*Each fold*], [],
    [Train], [31–33 animals],
    [Validation], [9 animals],
    [Held out], [9–10 new animals; new sessions of 26–30 training animals],
    table.hline(stroke: 0.5pt),
  )
}

#let body = assemble("fig1", files: files, content: (c: corpus))

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
