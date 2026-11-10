// Manuscript draft, in the Allen Institute preprint template (the submodule template/:
// `git submodule update --init`). Build from the repository root:
//   typst compile --font-path manuscript/template/fonts \
//     --input version="$(git describe --always --dirty --abbrev=40 --exclude='*')" manuscript/main.typ
// Results numbers come from values.toml, which notebooks/manuscript_figures.py writes
// (#v("section.key")): re-run the notebook, never edit them here. Numbers in yellow
// (#tbd) are typed by hand and still to be confirmed; red notes are for the authors
// and are removed before submission.
#import "template/preprint.typ": *
#import "figures/fig1.typ": body as fig1
#import "figures/fig2.typ": body as fig2
#import "figures/fig3.typ": body as fig3
#import "figures/fig4.typ": body as fig4
#import "figures/fig6.typ": body as fig6
#import "figures/fig7.typ": body as fig7
#import "figures/figs1.typ": body as figs1
#import "figures/figs2.typ": body as figs2
#import "figures/figs3.typ": body as figs3
#import "figures/figs4.typ": body as figs4
#import "figures/figs5.typ": body as figs5
#import "figures/figs6.typ": body as figs6
#import "figures/figs7.typ": body as figs7
#import "figures/figs8.typ": body as figs8
#import "figures/figs9.typ": body as figs9

// A figure assembled at its printed width (183 mm), scaled to the text width.
#let fit(body) = layout(size => scale(reflow: true, size.width / 189mm * 100%, body))

// The numbers the figure notebook computes (values.toml, written by
// notebooks/manuscript_figures.py), as printed: #v("section.key").
#let generated = toml("values.toml")
// A key the notebook has not written yet prints as a red placeholder, not an error.
#let v(key) = {
  let (section, name) = key.split(".")
  let found = generated.at(section, default: (:)).at(name, default: none)
  if found == none { text(fill: rgb("#b42318"))[\[#key\]] } else { found }
}

// The benchmark table (Table 1 and its held-out-session twin), from a CSV the figure notebook
// writes: "mean (low–high)" cells put the interval on a smaller line under the mean.
// Rows print in reverse (Zephyr first), each method with its citation, and the best mean
// of each score column (ties included) is bold.
#let method-cite = (
  "Motion energy": (<stringer2019>, <musall2019>),
  "Pixel PCA": (<poh2010>,),
  "Facemap": (<stringer2019>, <syeda2024>),
  "TS-CAN": (<liu2020>,),
  "PhysNet": (<yu2019>,),
  "PhysFormer": (<yu2022>,),
  "RhythmFormer": (<zou2025>,),
)
// One source of citations for both the benchmark tables and the methods table
// (Supplementary Table "Benchmarked methods"); method names match the benchmark CSVs.
#let cites(method) = method-cite.at(method, default: ()).map(l => cite(l)).join[ ]
// Source cell of the methods table: citations, then an optional note; "this work" if neither.
#let source(method, note: none) = {
  let c = cites(method)
  if c == none and note == none { [this work] } else if note == none { c } else if c == none {
    note
  } else [#c, #note]
}
#let benchmark-table(path) = {
  let rows = csv(path).slice(1).rev()
  // Columns 2 on (inference, rate errors, correlations, F1): lower is better for the
  // first four, higher for the rest.
  let lower = (2, 3, 4, 5)
  let mean(c) = float(c.split(" ").at(0).trim("*").replace("−", "-"))
  let best = range(2, 10).map(i => {
    let ms = rows.map(r => mean(r.at(i)))
    if i in lower { calc.min(..ms) } else { calc.max(..ms) }
  })
  let cell(c, i) = {
    let m = c.match(regex("^(\S+) \((.+)\)$"))
    let head = if m == none { c } else { m.captures.at(0) }
    let head = if i >= 2 and mean(c) == best.at(i - 2) { strong(head) } else { head }
    if m == none { [#head] } else {
      [#head \ #text(size: 5.5pt, fill: luma(90))[#m.captures.at(1)]]
    }
  }
  let name(method) = if method == "Zephyr" { [Zephyr (ours)] } else {
    let c = cites(method)
    [#method#if c != none [ #c]]
  }
  set text(size: 7pt)
  table(
    columns: 10,
    align: (left,) + (right,) * 9,
    fill: (_, y) => if y < 2 { colors.table-head } else { colors.table-row },
    toprule,
    table.header(
      [], [], [], table.cell(colspan: 4, align: center)[*Respiration rate (10-s windows)*],
      table.cell(colspan: 3, align: center)[*Waveform and events*],
      [*Method*], [*Splits*], [*Inference (s)*], [*MAE ↓*], [*RMSE ↓*], [*MAPE (%) ↓*],
      [*r ↑*], [*Waveform r ↑*], [*Inhale F1 ↑*], [*Exhale F1 ↑*],
    ),
    midrule,
    ..rows
      .map(r => (name(r.at(0)), [#r.at(1)]) + r.slice(2).enumerate().map(((j, c)) => cell(c, j + 2)))
      .flatten(),
    bottomrule,
  )
}

#let tbd(body) = highlight(fill: rgb("#fff3b0"), body)
#let note(body) = text(fill: rgb("#b42318"), size: 8.5pt)[\[#body\]]

// Figure reference with a panel letter: #fig(<fig:corpus>, "b") gives "Fig. 2b".
#let fig(label, panel) = [#ref(label)#panel]
// Placeholder for a figure that is not drawn yet; `source` says what will draw it.
#let mock(panels, source) = rect(
  width: 100%,
  inset: 10pt,
  fill: luma(246),
  stroke: (paint: luma(160), dash: "dashed"),
)[
  #set text(size: 8.5pt, fill: luma(70))
  #set align(left)
  #panels
  #v(2pt)
  _Source: #source _
]

// The commit the PDF was built from, at the top right of every page, when the build
// passes it (`--input version=...`, the full SHA with "-dirty" for uncommitted changes).
#let version = sys.inputs.at("version", default: none)
#set page(foreground: if version != none {
  place(top + right, dx: -1.5cm, dy: 0.6cm, text(size: 8pt, fill: luma(110))[Built from: #version])
})

// The authors and their contributions, from contributions.csv: one row per author, in
// the order of the byline (the platform first, then alphabetical by surname), one column
// per role. A cell is lead, equal or supporting (or a share in percent); empty is none.
#let contributions = csv("contributions.csv")
#let contribution-roles = contributions.first().slice(1)
#let author-names = contributions.slice(1).map(row => row.first())
#let contribution-level(cell) = {
  let cell = lower(cell.trim())
  let named = (lead: 100, equal: 50, supporting: 20)
  if cell == "" { 0 } else if cell in named { named.at(cell) } else { int(cell) }
}

#show: preprint.with(
  title: [Validated inference of respiration from videography in mice],
  short-title: [Zephyr],
  authors: author-names.map(n => (name: n, affiliations: ("aind",))),
  affiliations: (
    aind: [Allen Institute for Neural Dynamics, Seattle, WA, USA #note[authors] \
      Authors after the VrForaging Platform are listed in alphabetical order.],
  ),
  abstract: [
    Rodents rely on olfaction to survive. Odors guide them to find food, avoid predators,
    and recognize mates and kin. As mice actively sample odors via sniffing, the timing of
    each inhalation shapes odor perception. Ground-truth methods for measuring respiration in mice
    typically rely on sensors implanted in the nasal cavity, which require invasive surgery, and often fail to yield a reliable signal. Here, we developed a ground-truth-validated deep-learning method and show that the respiration waveform and
    the timing of each inhalation can be recovered directly from videography. We also provide a corpus
    of #v("corpus.sessions") curated sessions from #v("corpus.animals") head-fixed mice
    performing an olfactory foraging task, in which behavioral videos are
    synchronized with a nasal epithelial temperature sensor that provided ground-truth measurement of respiration.
    Our method - Zephyr - accurately extracted the timing of inhalation events and recovered the respiration waveform with high fidelity in animals held out from
    training. This performance was robust across different animals, recording conditions, and camera angles. We show that Zephyr can be trained with as few as 4 mice, making it feasible for adoption by standard behavioral and systems neuroscience experiments and labs.
    We propose that Zephyr, along with our curated dataset and quality-control tools, is a viable alternative to existing methods for measuring respiration.
  ],
)

= Introduction

Sniffing is a fundamental form of active sensing. Each inhalation draws odors into the nose, so the
timing of sniffing sets when olfactory information is sampled @kepecs2006 @verhagen2007
@wachowiak2011. Respiration in mice is approximately 2–4 Hz at rest and up to 12–15 Hz
during active sniffing @wesson2008. Olfactory bulb neurons respond at precise phases of this cycle
@shusterman2011. Respiration also reflects an animal's internal state #tbd[(citation needed)] and changes with arousal and emotion @homma2008 @delnegro2018. Neurons
of the breathing rhythm generator themselves promote arousal @yackle2017, and fear-conditioned mice
breathe at a steady 4 Hz while they freeze @bagur2021. The time of inhalation is therefore a
natural reference event for olfactory coding, and respiration would provide valuable state information in many
experiments on awake mice beyond studies of olfaction.

Measuring respiration with this precision in a behaving mouse usually requires an
implanted sensor. The standard methods place either a temperature-sensitive probe or a pressure cannula in
the nasal epithelium or cavity @grimaud2018 @wesson2008 @reisert2014 @mcafee2016. These methods, along
with plethysmography, piezoelectric sensors, and electromyography, each involve trade-offs
in timing precision, freedom of movement, or invasiveness @grimaud2018.
In addition, signals from chronically implanted sensors degrade over time, and are often discarded after inspection
@rafilson2025 @weinreb2024. A measure of respiration from video would spare animals this surgery, a
refinement in the sense of the 3Rs @tannenbaum2015. It would also allow extraction of respiration
signals from head-fixed experiments that already have readily available videography of the face.

The face carries signatures of respiration. Respiration and orofacial movements are coordinated
@moore2013 @kleinfeld2014, and machine vision now allows quantitative measurements of behaviour @mathis2018
@pereira2022 @weinreb2024. Motion from face video has also been leveraged to explain neural activity @stringer2019
@musall2019 @syeda2024. Yet not all facial motion is respiration. The nose, whisker pad, tongue, and jaw
are also involved in various orofacial movements, such as turning, whisking, licking, chewing, and grooming, and the nose can make rhythmic movements independent from
inhalation @kurnikova2017. Current tools that label a "sniffing" component of facial motion do so without
validation against ground-truth sensor data @syeda2024 @hakim2025. Thermal imaging detects temperature changes associated with inhalations in head-restrained mice but
requires specialized thermal cameras @esquivelzeta2017 @mutlu2018, and conventional cameras have
yielded only respiration rate in anaesthetized or sleeping animals @kunczik2019
@breuer2023 @geuther2022. Networks that recover pulse and respiration from human face video
@chen2018 @liu2020 @yu2019 @yu2022 @zou2025 @liu2023toolbox were built for slow, steady
signals, whereas a mouse changes its respiration rate several-fold within a second. To our knowledge, no validated method
has been shown to reconstruct the respiration waveform and timing of each inhalation from
conventional videography in rodents, and no public dataset exists to develop and test such a method.

To fill this gap, we assembled a corpus that pairs multi-view videos with synchronized nasal epithelial temperature
sensor (thermistor) recordings from #v("corpus.sessions") sessions in #v("corpus.animals")
head-fixed mice on #v("corpus.rigs") behavioral rigs. We tested computer-vision and deep learning methods
for recovering respiration from the video. Our network, which we call Zephyr, not only recovered the timing of each inhalation in
held-out animals but was also able to faithfully reconstruct the thermistor waveform. Zephyr performed well when trained across several camera views,
and, after retraining on a small amount of new data, generalized to new experimental conditions.
Finally, we show that curated quality-control scores computed from Zephyr's
output alone can be used to evaluate each prediction, generating confidence metrics of respiration signals extracted from videography in place of surgically implanted sensors. We release the corpus, the
code, as well as trained models, for others to use and refine on their own data.

= Results

== A corpus of orofacial video paired with nasal epithelial temperature sensor recordings

We trained head-fixed mice to perform an olfactory virtual-reality foraging task on a circular
treadmill. Thirsty mice traversed a virtual corridor with odor-cued sites and could stop at a site to
collect a probabilistic reward (#fig(<fig:overview>, "a")). This paradigm allowed us to capture
a wide range of respiration patterns within a single session, from breathing at rest to fast sniffing
(#fig(<fig:overview>, "b,c") and #fig(<fig:corpus>, "b")). Three high-speed cameras simultaneously filmed
the animal from multiple angles, and a thermistor implanted in the nasal cavity recorded respiration (see #link(<sec:methods>)[Methods]).

We curated a high-quality collection of thermistor
data (see #link(<sec:methods>)[Methods]) to ensure reliable model training targets. Of the #v("selection.clips") sessions that passed quality-control checks, we kept
#v("selection.corpus"), capped per animal and rig so that no animal or rig dominates the dataset.

The final corpus holds #v("corpus.sessions") sessions with two 300-s clips each
(≈#v("corpus.hours") h of video per camera), from #v("corpus.animals") mice on
#v("corpus.rigs") rigs over #v("corpus.months") months. Each mouse contributed a median of
#v("corpus.median_sessions") sessions over a median of #v("corpus.median_days") days, and
#v("corpus.multi_rig") were recorded on more than one rig (#fig(<fig:corpus>, "c")).
Respiration from our dataset spanned a wide and representative range for mice: #v("rates.slow")% of cycles were at 2–4 Hz,
#v("rates.medium")% at 4–8 Hz and #v("rates.fast")% at 8–15 Hz (#fig(<fig:corpus>, "b")).

#figure(
  fit(fig1),
  caption: [*Measuring respiration from face video.* The data panels each show one mouse
    held out from training. *a,* The experimental apparatus where data was recorded from, and thumbnails of the 3 available views in the corpus *b,* A representative snippet of the running velocity and respiration signals recorded during a recording session. *c,*
    Respiration rate against binned running velocity for the same example session as in b. Histograms are normalized within velocity bin. *d,*
    Zephyr Network's architecture (#link(<sec:methods>)[Methods]). A frame encoder and a temporal network feed two
    heads: the waveform head predicts the respiration waveform, and the inhalation-onset
    head the probability that an inhalation starts at each time step. *e,* Four seconds
    of a clip ran through Zephyr. The face frame beside it shows the crop the network sees (orange). The
    predicted respiration (blue) is drawn over the thermistor (black), above the inhalation-onset
    probability (green). Ticks mark the inhalation events extracted from the ground-truth or inferred by Zephyr, respectively.],
) <fig:overview>

== Neural networks recover respiration from video

Our goal was to infer respiration from conventional videography alone, so that respiration can be measured in animals without an implanted sensor. A useful method must recover both the respiration waveform and the timing of each inhalation, and must keep doing so in animals it was never trained on.

First, we asked whether the signals that current tools extract from face video can be used to track respiration. Motion energy, the mean absolute difference between frames, is the most common summary of facial movement @stringer2019 @musall2019. Even after fitting a sign and lag on the training clips, it could not predict respiration in held-out mice
($r = #v("benchmark.motion_r")$; #fig(<fig:benchmark>, "a,b")\; @tab:benchmark). A linear readout of
the leading facial motion principal components, as in Facemap @syeda2024, performed better but still recovered only part of the
signal ($r = #v("benchmark.facemap_r")$, inhalation F1 #v("benchmark.facemap_f1")).

Deep-learning methods have been shown to recover periodic physiological signals from human video.
We thus wondered if this class of methods would perform better on held-out mice and sessions.
We started from the state-of-the-art in camera-based physiological measurement methods, four networks developed to recover pulse
and respiration from human face video: TS-CAN @liu2020, PhysNet @yu2019, PhysFormer @yu2022
and RhythmFormer @zou2025. We trained each on the same clips under the same protocol, reading in
a 96 × 96 grayscale crop over the snout and predicting the thermistor waveform (#link(<sec:methods>)[Methods]).
These networks are built for long, slowly varying signals, in particular the two transformers are
large and slow to run.

We therefore propose a smaller and more efficient architecture called Zephyr. Zephyr encodes each frame with a small convolutional neural network (CNN), in series
with a temporal convolutional network (TCN) that models respiration over time. Additionally, we endowed Zephyr with a second head
aimed to predict inhalation onsets directly, whereas the networks we compare against predict only the
waveform and require a separate digital signal processing step to extract inhalation onsets
(#fig(<fig:overview>, "d")\; #link(<sec:methods>)[Methods]).

To measure how well methods generalize, for all analyses, we held out whole animals, and blocks of sessions of training animals, from training (#link(<sec:methods>)[Methods], @sec:data-splits).

We scored each method on held-out animals, never seen in training, in a five-fold
cross-validation in which each of #v("benchmark.animals_a") animals is held out once
(#link(<sec:methods>)[Methods], @tab:split). Scores were averaged within each animal on one rig before
comparison (#v("benchmark.units_a") animal × rig units; #link(<sec:methods>)[Methods], @sec:statistics).
Every network recovered respiration on held-out animals (#fig(<fig:benchmark>, "b")\;
@tab:benchmark\; an example in #fig(<fig:overview>, "e")), so respiration can indeed be learned from
face video in mice. Zephyr and PhysNet showed similar inhalation detection performance. Their F1
scores within 25 ms at 60 Hz were #v("benchmark.f1") (Zephyr; 95% interval
#v("benchmark.f1_ci")) and #v("benchmark.physnet_f1") (PhysNet; paired difference
#v("benchmark.physnet_f1_diff"), #v("benchmark.physnet_f1_diff_ci")). Their waveform
correlations with the thermistor were also similar: #v("benchmark.r") for Zephyr
(95% interval #v("benchmark.r_ci")) and #v("benchmark.physnet_r") for PhysNet (difference
#v("benchmark.physnet_r_diff"), #v("benchmark.physnet_r_diff_ci")); TS-CAN, RhythmFormer, and
PhysFormer reached #v("benchmark.others_r"). Importantly, while PhysNet and Zephyr nearly matched each other in accuracy, Zephyr processed
#v("benchmark.physnet_fps_gain")% more frames per second than the full 3D CNN PhysNet on a consumer GPU,
and #v("benchmark.transformer_fps_gain")% more than the two transformers
(#fig(<fig:benchmark>, "b")\; @tab:nets). Thus, custom neural networks can accurately and efficiently recover respiration waveforms from face video.

Mouse respiration spans a wide range of rates, so we asked whether accuracy held across that
range (#fig(<fig:benchmark>, "c")). Zephyr found almost every inhalation between 6 and
12 Hz (F1 #v("rate.f1_middle")) and slightly fewer above 12 Hz (#v("rate.f1_fast")). It was
least accurate in slow breathing, at 2–3 Hz (#v("rate.f1_slow")). The other networks lost
more accuracy at both ends of the range, a difference we return to below.

The ability to recover inhalations events with precise timing is crucial for behavior and physiology studies. Over the held-out animals of all folds, Zephyr's inhalations lagged the thermistor's by #v("rate.lag") ms on average (95%
interval over folds #v("rate.lag_ci") ms), well within one cycle at the fastest sniffing
rates (#fig(<fig:benchmark>, "d")). The lag hardly changed with respiration rate, from
#v("rate.lag_slow") ms at 2–3 Hz to #v("rate.lag_fast") ms at 11–12 Hz.
We also compared respiration rate, counted in 10-s windows, with the thermistor's, a measurement commonly reported for camera-based respiration in humans @villarroel2019
@liu2023toolbox. Zephyr's rate was off by #v("rate.mae") cycles per minute on average, or
#v("rate.mape")% of the thermistor's rate, with virtually no overall bias
(#v("rate.bias") cycles per minute).

#figure(
  fit(fig3),
  caption: [*Neural network methods faithfully recover respiration dynamics.* *a,*
    Six seconds of one clip of a held-out animal, predicted by each method. The
    thermistor is at the top and each method's waveform is drawn over it in grey. Vertical
    lines mark the thermistor's inhalations. Panels *b–e* show the results from held-out animals across folds, #v("rate.clips") clips from #v("rate.sessions") sessions of #v("rate.animals")
    animals. *b,* Inhalation F1 scores on new animals against
    inference speed. Filled symbols are networks on a consumer GPU and open symbols are methods on a CPU. *c,* Inhalation F1 on held-out animals by respiration rate for each network, mean over folds with its 95% interval. Colours are as in *b*. The bottom panel shows the distribution of respiration rates in the held-out data. *d,* Zephyr's inhalation onset lag (predicted - thermistor), as a function of respiration rate. Every thermistor
    inhalation with a prediction within 50 ms is included, pooled over the new animals of
    every fold (n = #v("rate.lags")). The line shows each rate bin's mean lag, averaged over folds (with 95% CI). Dotted lines mark the default 25-ms tolerance used for inhalation detection. *e,* Respiration rate in 10-s windows, pooled over held-out animals and every fold (n = #v("rate.windows") windows). The difference between Zephyr's inferred rate and the thermistor's is plotted against their mean. The solid line is the bias and the dotted lines are the 95% limits of agreement between the two methods.],
) <fig:benchmark>

#figure(
  kind: table,
  // Written by notebooks/manuscript_figures.py (new animals, over folds, seed 17).
  benchmark-table("tables/benchmark.csv"),
  caption: [*Method comparison on held-out animals.* Each score was averaged within each
    animal × rig unit, then over the held-out-animal units of a split, then over splits. Values
    are the mean with its 95% interval over the #v("benchmark.folds") cross-validation
    folds (#v("benchmark.units_a") units of #v("benchmark.animals_a") animals, each animal
    new in one fold). Methods run on one split show the published split alone
    (#v("benchmark.published_units") units of #v("benchmark.published_animals") animals),
    without an interval. Respiration rate is counted in 10-s windows. MAE and RMSE are in
    cycles per minute, MAPE is relative to the thermistor's rate, and r is the Pearson
    correlation of the windowed rates. Waveform r is the correlation of the predicted
    respiration waveform with the thermistor. F1 counts events within 25 ms. Inference is the
    time in seconds for a 5-min clip from the preprocessed crops on a consumer GPU, or on
    a CPU where marked (\*).],
) <tab:benchmark>

== Zephyr generalizes to new data

Because our goal is to measure respiration without implanted sensors, a useful videography-based method for inferring respiration must generalize to
non-implanted animals absent from its training dataset. We first considered a typical
scenario, in which a laboratory has recorded animals in a similar setup and wishes to apply
a trained network to new data acquired under similar, experimentally controlled conditions.
Besides whole animals, each cross-validation fold held out blocks of consecutive sessions of
its training animals (#link(<sec:methods>)[Methods], @tab:split). All networks recovered respiration from these held-out sessions with very
high waveform correlation and inhalation F1, and, although relatively lower, their scores on
held-out animals remained high (#fig(<fig:general>, "a")\; @tab:benchmark,
@tab:benchmark-b). We emphasize that this generalization depends on both the animal and the
recording setup (@fig:rigs). Thus, for most analyses we treat each animal × rig combination as
one unit of comparison.
Zephyr scored the highest among the networks on both new sessions (inhalation F1
#v("benchmark.f1_b"), waveform correlation #v("benchmark.r_b")) and held-out animals (#v("benchmark.f1")
and #v("benchmark.r")). Inhalation F1 generalized more robustly than waveform correlation: from
new sessions to new animals, Zephyr's F1 fell only from #v("benchmark.f1_b") to #v("benchmark.f1"),
against #v("benchmark.r_b") to #v("benchmark.r") for its waveform correlation, and the vast
majority of held-out samples scored above 0.9 (#fig(<fig:general>, "b")).

Since not all experiments record the same camera view, we next asked whether respiration
can be extracted from the other recorded angles. Each session of our corpus was filmed by
three cameras, allowing us to compare the performance of Zephyr across viewpoints. Besides
the face camera, a side and a front camera film the whole animal from farther away and at
different angles (@fig:views). Trained with the same protocol, Zephyr recovered respiration from both
views on new animals, though less accurately than from the face camera
(#fig(<fig:general>, "d,e")\; @fig:cameras, @tab:cameras) #note[side and front:
  fold cv_0 only so far; the other folds are training].

Networks trained on large and diverse data often generalize better than specialized ones,
as foundation models for vision @oquab2023dinov2 and for animal pose estimation pooled
across many laboratories and setups @ye2024superanimal have shown. We therefore asked
whether one network could learn all three views. Performance of a single network trained on
the pooled dataset of all three cameras was not different from the network trained on
each camera alone, on any of the three cameras (#fig(<fig:general>, "f"))
#note[fold cv_0 only so far; add the numbers once all folds are scored]. 

We next experimented with how much data Zephyr requires to generalize effectively to new animals.
Keeping the held-out animals of one fold fixed, we varied the number of training animals
from #v("size.draw_animals") to #v("size.all_animals"), each with all their sessions
(#fig(<fig:general>, "c")). Despite a clear trend of improvement with more animals,
#v("size.fewest") animals already gave an inhalation F1 of #v("size.fewest_f1") against #v("size.f1") with all #v("size.all_animals"). As
the number of animals decreased to #v("size.draw_animals"), average performance did not drop
dramatically but became more variable across the three draws (F1
#v("size.draws_f1")), suggesting that the space of inputs was not adequately represented in
such a small training set. We also note that the duration of video mattered less than the number of unique animals. All
#v("size.all_animals") animals with a quarter of their sessions matched the full training set (F1 #v("size.partial_f1")).
Together, these results show that Zephyr can be trained from different views and generalizes
effectively to unseen data from similar experimental conditions without requiring extensive training data.

However, whether the network generalizes to data from different experimental setups
remained an open question. We used an independent dataset, "dynamic-foraging task"
@su2026, in
which the faces of head-fixed mice were filmed from the side and from below (#fig(<fig:general>, "g")), to
evaluate how well Zephyr pre-trained on our corpus performed on these sessions. On a held-out animal, it was not able to directly track respiration (inhalation
F1 #v("refine.f1_before_side") from the side and #v("refine.f1_before_bottom") from below). Given our previous success on generalizing to new animals with little data, we next asked
whether a small amount of additional training videos from this new setup could improve
performance by fine-tuning the weights of the network. We retrained the network on the corpus expanded with two animals recorded in the dynamic foraging task and predicted on the third held-out animal. Both
measures improved markedly from both camera angles, to an inhalation F1 of
#v("refine.f1_after_side") and #v("refine.f1_after_bottom") and a waveform correlation of
#v("refine.r_after_side") and #v("refine.r_after_bottom") (#fig(<fig:general>, "h,i")), while
the refined network did not lose performance on the initial dataset. This
result supports the idea that while Zephyr may not generalize to unseen setups out of the box, it
can be fine-tuned with little data to adapt to new experimental conditions.

#figure(
  fit(fig4),
  caption: [*Generalization to new animals, camera views, and experiments.* *a,* Waveform correlation
    and inhalation F1 of each method on held-out animals and held-out sessions
    (#v("benchmark.units_a") and #v("benchmark.units_b") animal × rig units). Colours as in @fig:benchmark. *b,* Inhalation F1 of
    each held-out videos, ranked; the inset ranks animals, and lines join one animal's
    rigs. *c,* F1 scores on the same held-out data for networks trained on increasingly more animals of the same fold. The open circles indicate networks trained on all animals with a quarter of their sessions. *d,* Zephyr
    trained on the face, side or front camera (@fig:cameras). *e,* Inhalation F1 of each
    camera's network by respiration rate (colours as in *d*). *f,* One network trained on all three camera views. Inhalation F1 by respiration rate (top) and its
    difference from each single-camera network (bottom). *g–i,* The network of *f*
    refined on a "dynamic-foraging" experiment. *g* F1 scores of a held-out animal (3 animals in total with "leave-one-out") on its
    side and bottom views with their crops. *h* 12 s of one clip with the ground-truth thermistor
    (black), and the output of the fine-tuned network (colored as *g*). *i* correlation and
    inhalation F1 before and after refining, per held-out animal.],
) <fig:general>

== Quality-control on predictions without ground-truth sensor data

A method that replaces a sensor will be used where no sensor can check it, so its errors
must be predictable and calibrated. We therefore sought to understand the conditions under which Zephyr's predictions are reliable and how to quantify its confidence in the absence of ground-truth data.

By far the most common errors were relatively small timing mismatches between predicted and
thermistor inhalations, rather than missed or hallucinated inhalations. Relaxing the matching
tolerance from 25 to 50 ms raised Zephyr's inhalation F1 from #v("tolerance.f1_25") to
#v("tolerance.f1_50"), and the remaining errors fell from #v("tolerance.errors_25")% to
#v("tolerance.errors_50")% of cycles (#fig(<fig:trust>, "a")).

Timing also explained most of the loss of accuracy in slow breathing
(#fig(<fig:benchmark>, "c")). At 2–3 Hz, Zephyr found #v("slow.within_25_slow")% of the
thermistor's inhalations within 25 ms but #v("slow.within_50_slow")% within 50 ms. Its
timing spread was wider in slow breathing in milliseconds (#v("slow.iqr_ms_slow") ms at
2–3 Hz against #v("slow.iqr_ms_mid") ms at 6–8 Hz) but narrower as a share of the
respiratory cycle (#v("slow.iqr_cycle_slow")% against #v("slow.iqr_cycle_mid")%): with a
tolerance of 10% of each cycle, Zephyr found #v("slow.within_cycle_slow")% of inhalations at
2–3 Hz and #v("slow.within_cycle_fast")% at 10–12 Hz, where 10% of the cycle is shorter
than one video frame. A fixed 25-ms tolerance is thus strictest for slow breathing, which
Zephyr times more precisely than fast breathing relative to the cycle. Some of the remaining
false inhalations in slow breathing may be shallow cycles that the corpus detector does not
label reliability. Relatedly, lowering the ground-truth detector's prominence threshold added #v("slow.added_slow")% more
inhalations at 2–3 Hz and #v("slow.added_mid")% at 6–8 Hz (@fig:slow). In order words, depending on the scientific question, temporal precision may be sacrificed for overall detection sensitivity, especially at lower breathing rates.

Even if sources of most errors are known, we still need ways to assess the quality of the predictions. 
For instance, without having the thermistor raw data, one would like to be able to distinguish between the predictions provided by pre-trained and refined networks on the "dynamic foraging" experiments (add figure).

We calculated metrics (see methods) that we find reliably predict inference quality. Briefly, on each 30-s window of a prediction we computed four measures that need no ground truth: 1. how
well the inhalation onset head's inhalations agree with the peaks of the network's own waveform (head vs
waveform F1); 2. how often inhalations and exhalations fail to alternate; 3. how high the inhalation onset
peaks are; and 4. how high the inhalation onset probability stays between them
(#link(<sec:methods>)[Methods], @tab:qc-measures). Using validation data, we calibrated these scores. Next, we generated surrogate "bad" predictions by systematically adding noise to videos that progressively lower the network's accuracy. Finally, for comparison we also add the scores of the "dynamic foraging" dataset. We found that the scores tracked the true accuracy. With noise of 16 gray levels, inhalation
F1 barely changed (#v("quality.noise16_truth_inhale_f1") against
#v("quality.validation_truth_inhale_f1") on validation animals), and the scores stayed close
to their validation range. At 32 gray levels and beyond, F1 collapsed
(#v("quality.noise32_truth_inhale_f1")), and every score moved away from that range: head vs
waveform F1, for instance, fell from #v("quality.validation_head_trace_f1") to
#v("quality.noise32_head_trace_f1") (#fig(<fig:trust>, "d,e")). The dynamic-foraging
predictions behaved the same way. Before refining, the network found almost none of that
experiment's inhalations (true F1 #v("quality.df_unrefined_truth_inhale_f1")), its head vs
waveform F1 fell to #v("quality.df_unrefined_head_trace_f1") and the inhalation onset probability between
inhalations rose from #v("quality.validation_onset_background") to
#v("quality.df_unrefined_onset_background"). After refining, the same network scored as well on these
videos as it did on the corpus (#fig(<fig:trust>, "c,e")). The proposed metrics can thus separate
predictions that work from known failures without implanting a thermistor.

#figure(
  fit(fig6),
  caption: [*Zephyr predictions can be scored without ground truth.* *a,* Zephyr's inhalation F1 on held-out
    animals as the matching tolerance is relaxed (top), and the misses (FN) and false
    inhalations (FP) left (bottom). *b,* A clip with many near misses, predictions 25–100 ms from
    a thermistor inhalation (r = #v("clips.near_r"), F1 #v("clips.near_f1")). *c,* Twelve seconds of
    one dynamic-foraging clip predicted by the pooled network before and after refining,
    over the thermistor (black). *d,* Twelve seconds of one clip of a held-out animal
    with progressively more Gaussian noise added (std = 16, 32 and 64). *e,* The quality-control scores and the true
    inhalation F1 over 30-s windows of the pooled network's validation animals. Faint points are single windows, black dots each session's median across its windows, black ticks the mean of those medians, and shaded bars its 95% bootstrap interval over sessions. Bands in *a* are
    95% intervals over the #v("benchmark.folds") folds
    (#link(<sec:methods>)[Methods]).],
) <fig:trust>

== Respiration in experiments without a thermistor

Finally, we used the network for its intended purpose - measuring respiration in
experiments without a thermistor. We ran inference on all views from #v("application.animals") animals never implanted with a thermistor, but otherwise running in the same experimental conditions. Over their #v("application.windows") 30-s windows, the quality-control scores lay largely within the range of good predictions and away from
the known failures (#fig(<fig:application>, "a")). Nevertheless, outlier 30s windows were observed occasionally (#fig(<fig:application>, "a")). Anecdotally, these scores were related to individual windows and not entire 1-hour sessions. The example in #fig(<fig:application>, "b") shows how performance in an individual window drops during a grooming bout where the snout is not fully visible. This highlights how the metrics can not only be used to evaluate full sessions, but also reject individual windows with relatively lower confidence.

We then asked whether the predicted respiration carries the same behavioural structure as
the thermistor's. In one session (#v("application.minutes") min), we aligned the predicted
inhalations to the task as we had aligned the thermistor's (#fig(<fig:overview>, "b,c")).
The predicted waveform followed the transitions between running, stopping at odor sites and
licking (#fig(<fig:application>, "c")), and the respiration rate around odor onset
differed by the site's outcome as in the thermistor session, both in this session and on
average over all #v("application.sessions") animals (#fig(<fig:application>, "d,e")).
The joint distribution of respiration rate and running velocity is shown for the same session
and over all sessions in #fig(<fig:application>, "f,g"). Both analyses indicate that the predictions capture the relationship between respiration and behavior reliably.


#figure(
  fit(fig7),
  caption: [*Inferring respiration without ground-truth data.* *a,* Quality-control
    scores over the #v("application.windows") 30-s windows of the
    #v("application.sessions") sessions - from #v("application.animals") animals, beside
    predictions as in #fig(<fig:trust>, "e"). A white circle marks the window that scored
    the lowest on "head vs waveform" F1 metric (i). *b,* #v("application.shown_s") s of the
    window marked in *a*, centred on four frames of its face video with the crop box (top;
    frame 1, the face at rest; frames 2 and 3, paw strokes over the snout; frame 4, a
    lick, its tongue out). Dotted lines mark the frames' times. *c,* Fifteen seconds of
    one animal's session showing (top) predicted waveform, (middle) predicted inhalation onset
    probability, and (bottom) the predicted respiration rate (blue) aligned to features
    of behavior: running velocity (grey), licks (yellow tick), reward (orange) and odor
    inhalation onset (gray shading). *d, e,* Respiration rate and running velocity around odor onset,
    broken down by trail outcome. *f, g,* Respiration rate plotted against running velocity
    (as in #fig(<fig:overview>, "c")), as the distribution of rates in each velocity bin.
    *d, f,* The session of *c* (*d*: n = #v("application.sites_rewarded"),
    #v("application.sites_unrewarded") and #v("application.sites_skipped") sites,
    respectively; mean over sites with its 95% bootstrap interval). *e, g,* All sessions,
    averaged over sessions (*e*: n = #v("application.all_sites_rewarded"),
    #v("application.all_sites_unrewarded") and #v("application.all_sites_skipped") sites;
    mean over sessions of each session's mean, with its 95% bootstrap interval over
    sessions).],
) <fig:application>

= Discussion

We provide a corpus that pairs orofacial video with a nasal epithelial temperature sensor in
behaving mice, and a network Zephyr that successfully recovers inhalation in animals it has never seen. Not only did it recover general features of the thermistor waveform but also did so while providing high accuracy and temporal precision of inhalation onset time events. We show that signals that current behavioural tools extract from facial motion did
not faithfully recover respiration, so a "sniffing" component of facial motion should not be equated with
respiration without a sensor to validate it.

Our method has implications past the immediate context of neuroscience experiments. Reading respiration from video also bears on animal welfare. A nasal implant requires invasive surgery on the snout, can obstruct airflow and
damage the nasal epithelium, and can fail before an experiment even begins @mcafee2016
@grimaud2018. In our own hands, several sessions had to be discarded once the thermistor's signal degraded and was no longer reliable.
Once trained for a rig, the network generalizes to new animals, which then need no
implant. This refines the procedure for those animals and, where implants would otherwise
fail, reduces the number of animals used @tannenbaum2015, and simplifies otherwise complex
experimental procedures. Respiration can be inferred from face videos from prior
experiments without recording any new animals, provided the experimental conditions match those of the training data.

The proposed network architecture Zephyr is a small network trained to predict respiration data from head-fixed rigs and is not intended to be a general model of facial motion. Video from one rig varies little: the camera, lens, lighting and head
position stay the same from session to session, and a small network has the capacity for
it. Zephyr matched larger networks from camera-based physiology measurement tools in accuracy and
with improved inference speeds. The same architecture also performed well on input from the side-view and front-facing cameras, which
see the snout from farther away and at other angles, and a single network trained on all
three cameras matched each single-camera network on its own camera. Accuracy depended
more on the number of training animals than on the amount of video. Nevertheless, we achieved relevant accuracy with as few as 2 or 4 animals. Whether one network can maintain this accuracy across rigs, lenses,
strains and tasks, or each rig needs a network of its own, remains open. Our rigs share
one design and place the cameras in similar positions, yet their lighting and the
cameras' exact position, angle and lens still differ (@fig:views), and the network scored
lower on some rigs and animals than on others (@fig:rigs). The quality-control measures
score each window of a recording without a sensor and, calibrated against scores of predictions
known to work and known to fail, allow the user to set their own thresholds.

Adopting Zephyr therefore requires little beyond the videography many head-fixed laboratories already collect. A laboratory first runs the pre-trained network on its own recordings and uses the quality-control measures to check whether the predictions are reliable for its setup. If they are not, the network can be refined with a small amount of ground-truth data from the laboratory's own rig: a few animals recorded with video and a nasal epithelial temperature sensor (2 to 4 animals sufficed in our experiments). Once refined, Zephyr runs efficiently on a consumer-grade GPU, and no sensor is needed in subsequent animals.

The corpus also limits what we can claim. All mice were head-fixed, so the camera saw the
face from a fixed distance and angle, and the animal could not turn away, rear or groom
freely. Whether respiration can be read from video in freely moving mice remains untested.
Head-mounted cameras, which already film the eyes and face of freely moving mice during
natural behaviour and electrophysiology @meyer2018 @sattler2021 @michaiel2020 @parker2022,
keep the face at a fixed distance and angle, and are a natural setting in which to test
it. The thermistor measures the temperature of the airflow, not the airflow itself, so its
own delay relative to nasal pressure bounds the timing error we can measure.

As opposed to pose-estimation methods, where a human can manually correct and validate labels, annotation of breathing from video is much less straightforward and, at least in our hands, impossible to perform reliably. This presents a challenge since predictions cannot be easily verified against ground truth in cases where the thermistor is not implanted. To address this, our approach relies on quality-control measures that score each window against a calibrated reference, ensuring that only reliable predictions are considered. These are not hard cutoffs but rather provide a continuous measure of confidence that can guide the user in selecting trustworthy segments of the video and should be considered against the scientific context of the experiment.

Future methods may need fewer labels and generalize better to new experimental rigs. Thermistor recordings are costly
to collect, whereas video of the face is recorded in many experiments that measure no
respiration. Self-supervised methods, which in humans recover pulse from unlabeled face
video by exploiting the periodicity of the signal @gideon2021 @sun2022contrast, could
learn from such video and need a sensor only to check the result. In pose estimation,
models trained on data pooled from many laboratories already track animals in new
settings with little or no further labelling @ye2024superanimal, and general-purpose
visual foundation models @oquab2023dinov2 may provide features from which respiration can
be read with fewer labels. Camera physiology in humans advanced on shared datasets and
benchmarks recorded across groups @liu2023toolbox, and networks that excel on one dataset
often fall on another. We release the corpus in a format upon which other labs' recordings can be added, so that a benchmark spanning rigs,
tasks, strains and freely moving animals can be built from recordings that pair a sensor
with video of the face.

#set par.line(numbering: none)

= Data and code availability

We release the corpus (face, side and front video, frame timestamps and the thermistor,
on one clock) with the splits we used, the code, the trained weights, an evaluation that
reproduces the scores reported here, and an application that runs the network on new
recordings: it places a crop box on each recording, predicts respiration and reports the
quality-control measures. The sessions of the dynamic-foraging experiment and the
sessions without a thermistor, which are not part of the corpus, are released with their
video and the network's predictions #note[repository URL, data DOI (DANDI or the Registry of
  Open Data), licence; confirm the corpus can be released].

= Author contribution matrix
// From contributions.csv (see author-names): its rows are the authors, its columns the
// roles; the matrix wants one row per role.
#contribution-matrix(
  author-names.map(n => [#n]),
  contribution-roles.map(r => eval(r, mode: "markup")),
  contribution-roles
    .enumerate()
    .map(((j, _)) => contributions.slice(1).map(row => contribution-level(row.at(j + 1)))),
)

#pagebreak()
#references(bibliography("references.bib"))

#pagebreak()
= Methods <sec:methods>

== Data collection

_Surgery._ All procedures were approved by the Allen Institute Institutional Animal Care
and Use Committee #note[protocol number; mice: strain, sex, age at surgery, housing and
  light cycle; analgesia and post-operative care]. Thermistors were implanted over the
nasal epithelium as described @mcafee2016. Briefly, under isoflurane anaesthesia (1–5%), a
0.5-mm burr hole was thinned through the nasal bone #tbd[3.1] mm anterior to the
frontonasal suture and #tbd[0.5] mm right of the midline, and the thermistor tip was
lowered onto the epithelium without pressing on it. The hole was sealed with silicone
(Kwik-Cast), and the probe and a head bar were fixed with dental cement (Metabond).

_Data acquisition._ Data were acquired on the VrForaging platform (https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging), whose devices are synchronized by the Harp protocol (https://harp-tech.org). For this dataset, a Behavior board (https://github.com/harp-tech/device.behavior)
triggered the 240- or 360-Hz FLIR cameras, a custom-built olfactometer (https://github.com/harp-tech/device.olfactometer) delivered odors and
a sniff-detector board (described below) recorded the thermistor. Video was encoded in real time
with H.264 on the GPU (NVENC, constant quality 18) and re-encoded offline on the CPU
(libx264, CRF 18). Clips were cut from these files by stream copy.

For the "dynamic foraging" experiment, a similar setup to the one described previously @su2026 was used. Animals rested in a metal tube and were asked to lick left or right to collect probabilistic rewards. Cameras were triggered at 504 Hz by a similar Behavior board and respiration data collected with an identical sniff-detector.


== Sniff-detector board
TODO SONYA

== Corpus construction

Sessions were selected with the rules of @tab:qc, in order. Thermistor quality was judged
in 30-s windows after removing large artifacts (and #tbd[2] s around them), and clips were
cut only from stretches of at least 300 s in which every window passed.

#figure(
  table(
    columns: (auto, 1fr, auto, auto),
    align: (left, left, left, right),
    toprule,
    table.header([*Stage*], [*Rule*], [*Threshold*], [*Removed*]),
    midrule, [Window (30 s)], [Signal amplitude (5th–95th percentile range)], [≥ #tbd[100] raw units],
    [#tbd[2,038]], [], [Clipped samples], [#tbd[0]% at ≤ #tbd[20] or ≥ #tbd[4,090]],
    [—], [], [Cycles outside #tbd[2–15] Hz], [≤ #tbd[10]%],
    [—], [], [Repeated events (I→I or E→E)], [≤ #tbd[5]%],
    [—], [Session], [Repeated events over the session], [≤ #tbd[0.3]%],
    [#tbd[257]], [], [Longest run of good windows], [≥ #tbd[300] s],
    [#tbd[229]], [], [Event amplitude within expected range], [robust $z <= #tbd[3]$],
    [#tbd[1]], [], [Dead sensor (low SNR)], [modes ≥ #tbd[3] dB apart],
    [—], [Clips], [Room for two clips in good stretches], [#tbd[2] × #tbd[300] s],
    [#tbd[73]], [Cameras], [A camera missing], [],
    [#tbd[3]], [], [Timestamp alignment issues], [],
    [#tbd[13]], [], [Low exposure], [],
    [#tbd[6]], [Manual review], [], [],
    [#tbd[4]], midrule, [Front camera], [Not recorded in the session],
    [], [#v("optional_cameras.front_not_recorded")], [], [Not on the animal (manual review)],
    [], [#v("optional_cameras.front_reviewed")], bottomrule,
  ),
  caption: [*Corpus selection.* Rules are listed in the order they were applied,
    with the number of sessions each removed. Of #v("selection.recorded") recorded
    sessions, #v("selection.thermistor") passed the thermistor rules and
    #v("selection.clips") the clip and camera rules. Window rules mark 30-s windows as good
    or bad and remove no session on their own ("—"). The exception is amplitude, which is
    also applied to the session as a whole and removed #tbd[2,038] sessions. A session
    without a long enough run of good windows is removed. A repeated event is an
    inhalation followed by another inhalation, or an exhalation by another exhalation.
    Timestamps with a single one-frame rollback were repaired (#tbd[73]) rather than
    removed. Low exposure and unplugged sync cables were found after the sets were drawn,
    and those sessions were removed from them, leaving the #tbd[560] sessions of the
    corpus. The front camera is optional. A session without its video, because it was
    never recorded or pointed away from the animal, keeps its face and side clips, so the
    last two rows count front-camera videos left out, not sessions. The face and side
    cameras were recorded in every session #note[check the dead-sensor and window-rule
      counts].],
) <tab:qc>

_Thermistor events._ Inhalation and exhalation onsets were detected on the thermistor at
its own rate (200-250 Hz) with an adaptive peak detector. The signal was filtered (zero-phase
second-order Butterworth, high-pass #tbd[1] Hz, low-pass #tbd[40] Hz) and z-scored over
the recording. Maxima at least #tbd[50] ms apart were candidate inhalation onsets, and
minima candidate exhalation onsets. A candidate was kept when its prominence was at least
#tbd[0.2] of the local range of the signal (5th to 95th percentile over a centred
#tbd[1.5]-s window), so that a shallow cycle next to deep ones is kept while small
ripples on a flat stretch are not. These events were used for training, quality control
and scoring.

The additional sessions from the "dynamic foraging" experiment, and the sessions recorded
without a thermistor (Sessions without a thermistor), are available for download, but are
not part of the corpus.

== Data splits <sec:data-splits>

To test generalization to held-out data, for each fold, we split the data into training, validation and test.
Test data was further split into two categories: held-out sessions (sessions from animals seen during training but on different days) and held-out animals (animals never seen during training). We drew five splits that remained constant throughtout the manuscript for all methods/networks. The cross-validation folds are described in Supplementary Note 1.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, right, right),
    toprule,
    table.header([*Set*], [*Held out as*], [*Animals*], [*Sessions*]),
    midrule, [Train], [—], [#tbd[31–33]],
    [#tbd[213–295]], [Validation], [whole animals], [#tbd[9]],
    [#tbd[67–144]], [Held-out animals], [whole animals], [#tbd[9–10]],
    [#tbd[75–128]], [Held-out sessions], [sessions of training animals], [#tbd[26–30]],
    [#tbd[5]], midrule, [Total], [],
    [#tbd[50]], [#tbd[560]], bottomrule,
  ),
  caption: [*Split sizes.* Values are the minimum and maximum over the #tbd[6]
    splits, the published split and the #tbd[5] cross-validation folds. In the folds,
    each of the #tbd[48] animals outside the #tbd[359]-Hz group is a held-out animal once.
    Validation includes #tbd[2] sessions of the #tbd[359]-Hz animals. Animals with held-out
    sessions are also training animals, so animals do not add up across sets.],
) <tab:split>

== Preprocessing

For the face camera, frames were downsampled to 360 × 270 pixels, cropped to a 96 × 96-pixel box over the
snout (one per session), converted to gray and selected on a 60-Hz grid by their
timestamps, and standardized over the training clips. We note that the network also
performs well at 30 Hz (@fig:input-rate) but with slightly reduced accuracy. For the side and front cameras,
which film the animal from farther away, frames were kept at the native 720 × 540 resolution
instead, with the same 96 × 96-pixel box over the snout. The target was the thermistor
filtered as above but high-passed at #tbd[0.2] Hz, resampled to 60 Hz and z-scored per
clip.

== Network architecture

A convolutional encoder, whose first layer spans three frames, embeds each crop
(@tab:arch); embeddings are interpolated by their timestamps onto a uniform 60-Hz grid, and
a non-causal temporal convolutional network (receptive field #tbd[4.2] s) outputs the
respiration waveform and an inhalation onset logit at each step (#tbd[0.86] million parameters).

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    toprule,
    table.header([*Stage*], [*Layer*], [*Output*]),
    midrule, [Input], [gray crop, three consecutive frames],
    [3 × 96 × 96], [Encoder], [Conv3D 3×5×5, stride (1, 2, 2), BN, GELU],
    [32 × 48 × 48], [], [Conv 3×3, stride 2, BN, GELU],
    [64 × 24 × 24], [], [Conv 3×3, stride 2, BN, GELU],
    [96 × 12 × 12], [], [Conv 3×3, stride 2, BN, GELU],
    [128 × 6 × 6], [], [average pool to 2×2, flatten, linear, GELU],
    [128 per frame], [Resampling], [linear interpolation by frame timestamps],
    [128 at 60 Hz], [Temporal], [Conv 1×1],
    [128], [], [6 residual blocks of 2 × (Conv 3, dilation $d$, BN, GELU) and dropout; $d = 1, 2, …, 32$],
    [128], [Heads], [Conv 1×1 (waveform), Conv 1×1 (inhalation onset logit)],
    [1 + 1], bottomrule,
  ),
  caption: [*Network architecture.* BN, batch normalization.],
) <tab:arch>

== Training

The loss was $1 - r$ between the predicted and target waveforms, averaged over three time
scales (the waveform, and copies average-pooled by 4 and 16 samples), plus #tbd[0.5] times a
binary cross-entropy on inhalation onsets against Gaussians (s.d. #tbd[20] ms) centred on the
thermistor inhalations, with positives weighted by the inverse of their share in each
batch. Other settings are in @tab:train; the checkpoint with the highest mean
correlation on the validation clips was kept #note[seeds, GPU, time per run]. The
comparison networks were trained the same way, each with the loss of its original paper
(Supplementary Note 2).

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    toprule,
    table.header([*Setting*], [*Value*]),
    midrule, [Input],
    [gray, 96 × 96 crop, 60-Hz selection grid], [Window, batch],
    [#tbd[512] samples (#tbd[8.5] s), #tbd[16]], [Optimizer],
    [AdamW, learning rate #tbd[6 × 10#super[−4]], weight decay #tbd[10#super[−4]]], [Schedule],
    [#tbd[200]-step linear warm-up, cosine decay to 1%], [Regularization],
    [dropout #tbd[0.1] (temporal network), gradient clip #tbd[1], weight EMA #tbd[0.999]], [Loss],
    [multi-scale $1 - r$ (scales 1, 4, 16) + #tbd[0.5] × inhalation onset BCE], [Inhalation onset target],
    [Gaussians, s.d. #tbd[20] ms, per-batch positive weight], [Augmentation],
    [frame jitter ±0.25 frame, shift ±#tbd[4] px, zoom ±#tbd[10]%, brightness ±#tbd[15]%, contrast ±#tbd[20]%, noise #tbd[2] grey levels],
    [Selection],

    [mean correlation on validation animals, every #tbd[4] epochs], [Stopping],
    [patience #tbd[12] scorings, #tbd[40]–#tbd[200] epochs of #tbd[100] steps], [Precision],
    [bf16 mixed precision], bottomrule,
  ),
  caption: [*Training settings of Zephyr.*],
) <tab:train>

== Inference and event detection

Clips were predicted in overlapping windows of #tbd[1,024] samples, discarding #tbd[126]
samples (half the receptive field) at each edge, so that the stitched waveform equals a
single pass over the clip. Inhalation onsets were the peaks of the inhalation onset probability above
a threshold, at least #tbd[50] ms apart. The threshold was chosen for each trained network
on its own validation animals, never on a test set, once early stopping had fixed its
weights. Each threshold from #tbd[0.20] to #tbd[0.85] in steps of #tbd[0.05] was scored
by the mean inhalation F1 over the validation clips (@sec:scoring). Thresholds within
#tbd[0.002] of the best F1 tied, and of these the one whose inhalations per minute came
closest to the thermistor's was kept. The network keeps this threshold, and every
prediction it makes uses it. It was
#v("threshold.folds") for the five folds' networks and #v("threshold.published") for the published split's. Exhalations, and the inhalations of methods without
an inhalation onset output ("DSP"), were found on the predicted waveform by the detector used on the
thermistor, with the same settings (the 40-Hz low-pass does not apply at 60 Hz).

== Benchmarked methods

Every method read the same 96 × 96 gray crops on the 60-Hz grid and was scored by the same
rules (Scoring). The three pixel methods compute a waveform from each clip's pixels with no
training, so the waveform has no set polarity or timing relative to respiration. Each waveform was
therefore band-passed to the respiration band (1–15 Hz) and turned the right way up from its
own shape: inhalation is shorter than exhalation, so a correct waveform falls faster than it
rises. A single sign and lag (at most 0.5 s), the same for every clip, were then chosen to
maximise the mean correlation with the thermistor over the training clips, and applied
unchanged to the test clips.

The methods, their sources and what we changed are in @tab:methods. The four networks
were trained in our training loop on the losses of their own papers (Supplementary
Note 2); none has an inhalation onset output, so their inhalations were found on the predicted waveform.

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    toprule,
    table.header([*Method*], [*Source*], [*Details*]),
    midrule, [Motion energy], [#source("Motion energy")],
    [Motion energy: mean absolute
      difference between consecutive frames over the crop],
    [Pixel PCA],
    [#source("Pixel PCA")],

    [Of the top #tbd[10] principal components of the
      band-passed pixels, the one most concentrated in a single spectral peak],
    [Pixel SNR],
    [#source("Pixel SNR")],

    [Band-passed pixels weighted by their share of variance in
      the respiration band, signed to agree, averaged],
    [Facemap],
    [#source("Facemap")],

    [Top #tbd[100] principal components of the
      crop's motion energy and raw frames, at #tbd[11] lags (#tbd[±250] ms); ridge
      readout, strength chosen leaving one training recording out],
    [TS-CAN],
    [#source("TS-CAN")],

    [One gray channel; output band-passed to 1–15 Hz], [PhysNet], [#source("PhysNet")],
    [One gray channel; 96 × 96 crop], [PhysFormer],
    [#source("PhysFormer", note: [authors' code])],
    [One gray channel; crop upsampled to 128
      pixels; rate spectrum over 1–15 Hz],
    [RhythmFormer],
    [#source("RhythmFormer", note: [authors' code])],

    [One gray channel; crop upsampled to 128
      pixels; rate spectrum over 1–15 Hz],
    bottomrule,
  ),
  caption: [*Benchmarked methods.* For methods that are not networks, the details
    give their settings. For the networks, they give only what we changed; otherwise the
    networks follow their papers. All networks read one gray channel instead of colour,
    and a respiration band replaces the heart-rate band.],
) <tab:methods>

_Timing._ Inference time was the wall-clock time from preprocessed crops in memory to the
60-Hz waveform, median over #tbd[8] 5-min clips after a warm-up, at batch size 1 in bf16 on an
NVIDIA RTX 4060 Ti (non-network methods on the CPU, an Intel Core i9-13900K with 24 cores and 64 GB of
RAM).

== Scoring <sec:scoring>

Predictions and thermistor were compared by time, over the span both cover. Waveforms were
scored by their Pearson correlation with the target. By default, a predicted event could match a
thermistor event within 1.5 frames of the prediction (25 ms at 60 Hz). Predicted and thermistor
events were paired one to one by an optimal assignment on their time differences
(Hungarian algorithm), restricted to pairs within the tolerance, which maximizes the
number of matches. Matched pairs were true positives (TP), unmatched predictions false
positives (FP) and unmatched thermistor events false negatives (FN), and
$F_1 = 2"TP" \/ (2"TP" + "FP" + "FN")$ was computed per clip. Respiration rate was counted in 10-s windows.

_Error analysis._ An error was a near miss when an error of the other kind lay within
100 ms. Errors were pooled over each fold's held-out animals (#v("errors.misses") misses
and #v("errors.false") false inhalations). The tolerance curve pools each fold's #v("tolerance.breaths") thermistor
cycles. The near-miss example (#fig(<fig:trust>, "b")) is, among the first fold's clips
of held-out animals with at least the median correlation, the one with the most
predictions 25–100 ms from a thermistor inhalation, shown where these are densest.

== Spatial and temporal receptive fields

For each animal × rig unit of the held-out animals of every
fold, we took the clip nearest the unit's median waveform correlation, two 8.5-s windows of
it, and the face network of that fold, and considered only the thermistor inhalations at
least half a receptive field from a window's edges. For occlusion, a 12 × 12-pixel square
of the 96 × 96-pixel crop, moved in 6-pixel steps, was set to the training mean in every
frame; we recorded the drop in the mean inhalation onset probability at the thermistor's
inhalations, and one minus the correlation of the waveform with the unoccluded one, and
gave each pixel the mean over the squares covering it. For gradients, we took the
absolute gradient of the inhalation onset logit at each thermistor inhalation with respect to every
input pixel, averaged over frames for the spatial map and summed over pixels for the
temporal profile. The same network with its weights re-initialised served as a control.
The temporal profile is the share of an inhalation's gradient carried by
each frame, in 1/60-s bins of the frame's time from the inhalation, averaged over the
inhalations of a unit, then over units and folds (@fig:saliency).

== Generalization experiments

_Training-set size experiments._ We retrained Zephyr on parts of the training set of fold
#v("size.fold") (#fig(<fig:general>, "c")): nested subsets of #v("size.animals") of its
#v("size.all_animals") training animals with all their sessions (#v("size.sessions")
sessions), and #v("size.partial_sessions") sessions drawn from all #v("size.all_animals")
animals. Each experiment was trained 3 times on a different draw of subset of trials. For the 2 animals condition, 5 networks were trained instead to account for the larger variability.

_Refining on a different task._ The pooled network of the first fold was refined on
video of a dynamic-foraging experiment, recorded with a side and a bottom camera and a nasal epithelial
temperature sensor (thermistor), and cut and cropped as the corpus (360 × 270-pixel frames, 96 × 96-pixel
crops). Training started from the pooled network's checkpoint, with half of each batch
from the fold's training clips of all three corpus cameras and half from the 8 sessions
of 2 animals of that experiment, at a learning rate of 1.5 × 10#super[−4], with the other
settings of @tab:train. Early stopping and the inhalation onset threshold used the fold's
validation animals. Both the refined network and the untouched pooled network were
scored on the #v("refine.clips") clips (#v("refine.sessions") sessions) of a third
animal of that experiment, and the refined network also on the fold's test sets. The cost
on the corpus is the refined network minus the pooled one on the same clips, as the
mean over animal × rig units with a paired 95% bootstrap interval (@fig:refine).

== Quality control

Each prediction was cut into 30-s windows, and each window was scored with the measures
of @tab:qc-measures. Each network was calibrated against validation animals by computing the distribution of the metrics against the F1 scores of the validation dataset. A report gives each
window's measures and their percentiles among the reference's windows.

The measures were tested on predictions known to work and known to fail, all from the
pooled network of the first fold, before or after refining. Working predictions were its predictions on that
fold's held-out animals (#v("quality.held_out_windows") windows) and, after refining, on
the dynamic-foraging experiment's held-out animal (#v("quality.df_clips") clips, scored
with the refined network's own reference). Failing predictions were its predictions on
that animal before refining, and on 24 clips of the held-out animals with Gaussian noise
of standard deviation 16, 32 or 64 gray levels (of 255) added to every frame.

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, left),
    toprule,
    table.header([*Measure*], [*What it measures in a window*], [*Better*]),
    midrule, [Inhalations], [Number of inhalations from the inhalation onset output],
    [—], [Respiration rate], [Median rate between consecutive inhalations (Hz)],
    [—],
    [Breaths too fast],
    [Share of breaths faster than a mouse breathes (15 Hz); slower rates are not
      judged, as they change with running, odors and task],

    [Lower],
    [Alternation breaks\*],
    [Share of events (inhalations from the inhalation onset output, exhalations from the waveform, in
      time order) that repeat the kind before them; breathing alternates],

    [Lower],
    [Flat breaths],
    [Share of breaths (inhalation to inhalation) whose waveform swings less than a quarter
      of the clip's median breath: stretches where the waveform sees nothing],

    [Lower],
    [Head vs waveform F1\*],
    [Agreement between the inhalation onset output's inhalations and the peaks of the network's own
      waveform (thermistor's detector, 1.5-frame tolerance): the two outputs should agree],

    [Higher], [Inhalation onset peak height\*], [Median inhalation onset probability at the inhalations: how sure the inhalation onset output is],
    [Higher], [Ambiguous inhalation onset peaks], [Share of inhalation onset-probability peaks above 0.2 that stay below the inhalation onset threshold],
    [Lower],
    [Inhalation onset background\*],
    [Mean inhalation onset probability further than 100 ms from any inhalation: how quiet it is
      between breaths],

    [Lower],
    [Expected precision],
    [Mean probability that an inhalation is real given its peak height, from a
      calibration fitted on the reference's validation clips],

    [Higher], bottomrule,
  ),
  caption: [*Quality-control measures.* Each is computed on every 30-s window from the
    network's waveform and inhalation onset probability alone, without the thermistor or the video.
    Reports give each value and its percentile among the reference's validation
    windows, never a pass or fail. "Better" is the direction a working prediction tends
    to score. \*Shown in @fig:trust e and @fig:application a; every measure is in @fig:qc-all.],
) <tab:qc-measures>

== Sessions without a thermistor

The face videos of #v("application.sessions") task
sessions of #v("application.animals") mice without a thermistor were predicted with the
network trained on all three cameras of the first fold, at its inhalation onset threshold. Other than the thermistor signal, all data acquisition progressed identically to the original corpus. The side and front videos of the same sessions (@fig:application-cameras) were predicted the same way.

== Statistics <sec:statistics>

The unit of analysis is an animal on one rig, whose sessions share the implant, camera
position and lighting; two rigs of one animal differ in score about two-thirds as much as
two animals on one rig (@fig:rigs). Scores were averaged over a unit's clips, then over a fold's units,
and reported as the mean over the #tbd[5] folds with a 95% bootstrap interval over folds; differences between
methods are paired by fold. Analyses of a single fold, or of the published split,
use a percentile bootstrap over units (#tbd[2,000] draws). F1 by rate, F1 against
tolerance and the error profile pool each fold's cycles before averaging over folds; the
inhalation onset lag and rate agreement pool all folds and carry no interval. #note[units of one animal are not independent of each other; check intervals
  with a bootstrap that resamples animals and then units within them]

#pagebreak()
#show: supplement
= Supplementary information

#let csv-table(path, columns, header, align: auto) = {
  let rows = csv(path).slice(1)
  table(
    columns: columns,
    align: if align == auto { (left,) + (right,) * (columns - 1) } else { align },
    toprule,
    table.header(..header.map(h => [*#h*])),
    midrule,
    ..rows.flatten(),
    bottomrule,
  )
}

== Supplementary Note 1: Cross-validation folds

@tab:folds gives the number of animals and sessions in each set of the five
cross-validation folds (#link(<sec:methods>)[Methods], @sec:data-splits).

#figure(
  kind: table,
  csv-table("tables/cv_folds.csv", 6, ([Set], [cv_0], [cv_1], [cv_2], [cv_3], [cv_4])),
  caption: [*Cross-validation folds.* Each cell gives the number of animals / sessions in one
    set of one fold. Animals with held-out sessions are also training animals, so animals
    do not add up across sets.],
) <tab:folds>

== Supplementary Note 2: Comparison networks

TS-CAN @liu2020, PhysNet @yu2019, PhysFormer @yu2022 and RhythmFormer @zou2025 were
reimplemented from their papers and the authors' code, with their published layers, and
trained through the same training loop as Zephyr: the same gray 96 × 96 crops on the
60-Hz grid, optimizer, warm-up and cosine schedule, weight averaging, augmentation
(without the frame jitter of Zephyr) and early stopping on the validation animals.
Each was trained on the loss of its own paper. TS-CAN, whose input is resized to
#tbd[36] × #tbd[36] pixels, is trained with the mean squared error against the
standardized first difference of the target, and PhysNet with the negative Pearson
correlation. PhysFormer and RhythmFormer read the crops upsampled to the 128 pixels they
were built for and are trained with a negative Pearson term plus a cross-entropy over
each window's spectrum (one bin per cycle per minute over 1–15 Hz) against the
thermistor's rate in that window: $0.1 r + b ("CE" + "KL")$, with $b$ rising from 1 to 5
over training, for PhysFormer, and $0.2 r + "CE"$ for RhythmFormer. Their attention spans
a whole window, so they were also run in windows of their training length. Window,
batch, steps per epoch and learning rate are in @tab:nets #note[confirm how the learning
  rates were chosen: on the validation animals, or from the papers (PhysFormer 10#super[−4],
  RhythmFormer 9 × 10#super[−3])].

#figure(
  kind: table,
  csv-table(
    "tables/comparison_nets.csv",
    9,
    ([Network], [Window], [Batch], [Steps], [LR], [Inference window], [Scales], [Params (M)], [GPU (s)]),
  ),
  caption: [*Settings of the neural networks.* Training and inference windows are in samples
    at 60 Hz, and the batch is the number of training windows per optimizer step. Scales
    are the time scales of the correlation loss, used by Zephyr and PhysNet only. The last
    two columns give the number of parameters and the inference time for a 5-min clip on a
    consumer GPU (NVIDIA RTX 4060 Ti; median of #tbd[8] clips; #link(<sec:methods>)[Methods]).],
) <tab:nets>

#figure(
  kind: table,
  // Written by notebooks/manuscript_figures.py (held-out sessions, over folds, seed 17).
  benchmark-table("tables/benchmark_b.csv"),
  caption: [*Method comparison on held-out sessions of training animals.* As in
    @tab:benchmark, but on held-out sessions (#v("benchmark.units_b") animal × rig units of
    #v("benchmark.animals_b") animals). Values are the mean over the
    #v("benchmark.folds") cross-validation folds with its 95% interval. Methods run on one
    split show the published split alone.],
) <tab:benchmark-b>

#figure(
  kind: table,
  // Written by notebooks/manuscript_figures.py (the folds every camera has, seed 17).
  csv-table(
    "tables/cameras.csv",
    7,
    ([Set], [Camera], [Units], [Waveform r], [Inhale F1], [Exhale F1], [Rate MAE]),
  ),
  caption: [*Zephyr on each camera.* Zephyr trained on the face, side or front camera, and
    scored on the recording spans all three cameras share, against the same thermistor
    events. Scores are on held-out animals (#v("cameras.clips_a") clips) and held-out
    sessions (#v("cameras.clips_b") clips) of #v("cameras.folds") of the cross-validation
    folds, as the mean over animal × rig units with its 95% bootstrap interval. Rate MAE is
    the error in respiration rate over 10-s windows, in cycles per minute.],
) <tab:cameras>

#pagebreak()
= Supplementary figures

#figure(
  fit(fig2),
  caption: [*Summary of the benchmarking corpus.* *a,* Selection of the recorded
    sessions. Each step removes the sessions that failed a rule, labelled by the first rule
    they failed (rules and thresholds in @tab:qc). *b,* Distribution of respiration rates
    over thermistor cycles at 2–15 Hz, for the whole corpus (black; n = #v("rates.breaths")
    cycles) and for each animal (grey; n = #v("rates.animals")). *c,* Number of sessions per
    animal, coloured by the number of rigs the animal was recorded on
    (n = #v("corpus.animals") animals, #v("corpus.sessions") sessions).],
) <fig:corpus>

#figure(
  fit(figs1),
  caption: [*Scores differ less between rigs than between animals.* Zephyr on the held-out
    animals of the five folds. A unit is an animal on one rig, with its score averaged
    over its clips. Histograms show the absolute difference in inhalation F1 (left) and
    waveform correlation (right) between pairs of units that share the animal but not the rig
    (#tbd[159] pairs of #tbd[22] animals), and between pairs that share the rig but not the
    animal (#tbd[520] pairs of #tbd[48] animals). Points above give each kind's mean with
    its 95% bootstrap interval over animals.],
) <fig:rigs>

#figure(
  fit(figs2),
  caption: [*Zephyr works on 30-Hz video, with slightly lower accuracy.* Zephyr trained
    on every other frame (30 Hz) of each of the five folds, still predicting on the 60-Hz
    grid. *a,* The 30-Hz network minus the 60-Hz network on each test set, in waveform
    correlation, inhalation F1 and exhalation F1. Grey points are single folds; the mean is
    shown with its 95% bootstrap interval over folds. *b,* Inhalation F1 on held-out
    animals by respiration rate, mean over folds with its 95% interval. The bottom panel
    shows the share of cycles in each rate bin, mean over animal × rig units with its 95%
    bootstrap interval.],
) <fig:input-rate>

#figure(
  fit(figs3),
  caption: [*Zephyr recovers respiration from the side and front cameras.* Zephyr trained on
    each camera's video in #v("cameras.folds") of the cross-validation folds, and scored on
    the recording spans all three cameras share. The held-out animals have
    #v("cameras.clips_a") clips from #v("cameras.units_a") units of #v("cameras.animals_a")
    animals, and the held-out sessions #v("cameras.clips_b") clips from
    #v("cameras.units_b") units. Every score is in @tab:cameras. *a,* Inhalation F1 of each
    held-out-animal unit, ranked by the face camera, as in #fig(<fig:general>, "b"). *b,*
    Inhalation F1 of each held-out-animal clip from the side and front cameras, plotted
    against the face camera. The dotted line marks equality, and the legend gives the share
    of clips that scored higher than with the face camera.],
) <fig:cameras>

#figure(
  fit(figs4),
  caption: [*Representative frames from corpus* First frames of the face, side and front cameras in
    #v("views.sessions") representative sessions, each labelled by its session name, with
    the network's crop box outlined (#link(<sec:methods>)[Methods]).],
) <fig:views>

#figure(
  fit(figs5),
  caption: [*A fixed tolerance is strictest for slow breathing.* Zephyr on the held-out
    animals of the #v("benchmark.folds") cross-validation folds, by the thermistor's
    respiration rate (bins labelled by their lower edge; 12 is 12–15 Hz). Lines show the
    mean over folds and bands its 95% interval. *a,* Share of thermistor inhalations found
    within 25 ms (the tolerance used throughout), within 50 ms, and within 10% of the
    cycle. *b,* Spread of Zephyr's inhalation times around the thermistor's (interquartile
    range of the lag). *c,* The same spread as a percentage of the cycle. *d,* Share of
    thermistor inhalations found 25–100 ms off, which count as both a miss and a false
    inhalation, and of those with no inhalation within 100 ms. *e,* Inhalations added by
    the thermistor's detector, as a percentage of the labelled cycles in each bin, when its
    prominence fraction is lowered from 0.2 to 0.1 or 0.05.],
) <fig:slow>

#figure(
  fit(figs6),
  caption: [*Spatial and temporal receptive fields.* The face network of each
    cross-validation fold on its held-out animals (#v("saliency.units") animal × rig units),
    and the side and front networks of the first fold on the same recording spans
    (#v("saliency.units_side") units; #link(<sec:methods>)[Methods]). *a,* One held-out
    animal seen by the three cameras, with the occlusion map inside each crop box: how much
    hiding each part of the crop lowered the inhalation onset probability at the thermistor's
    inhalations. *b,* Maps averaged over each camera's units: occlusion, the gradient of
    the inhalation onset output with respect to the input pixels, and the same gradient for the
    network with random weights. Each map is scaled to its own range. *c,* Share of an
    inhalation's gradient carried by each frame, plotted against the frame's time from the
    inhalation. Lines show the mean over folds; for the face network, the band shows the
    95% interval over the #v("benchmark.folds") folds.],
) <fig:saliency>

#figure(
  fit(figs7),
  caption: [*Refining on a different task costs little on the corpus.* The network of
    #fig(<fig:general>, "f") after refining on the dynamic-foraging experiment
    (#fig(<fig:general>, "g–i")), minus the same network before refining, on the held-out
    animals and held-out sessions of the first fold, over all three cameras. Points show the
    mean over animal × rig units with its 95% paired bootstrap interval.],
) <fig:refine>

#figure(
  fit(figs9),
  caption: [*Every quality-control measure.* Each measure of @tab:qc-measures over the 30-s
    windows of the sets of #fig(<fig:trust>, "e"), drawn as there. *a,* Measures of the
    respiratory rhythm. *b,* Measures of the inhalation onset output, beside the true inhalation F1.
    Expected precision is not computed for the noise sets.],
) <fig:qc-all>

#figure(
  fit(figs8),
  caption: [*Quality control of the sessions without a thermistor on each
    camera.* The #v("application.sessions") sessions of @fig:application, predicted from the
    face, side and front videos by the network trained on all three cameras. Each camera's
    quality-control scores over the 30-s windows of every session are shown beside
    predictions that work and known failures, drawn as in #fig(<fig:trust>, "e").],
) <fig:application-cameras>
