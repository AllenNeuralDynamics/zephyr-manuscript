// Supplementary Fig. 11: the behavioral rig. Build:
// typst compile figures/figs11.typ (from manuscript/).
// Panel a: rendered in Blender (3d-vr-foraging, tools/frontleft.py) and labelled by hand.

#import "lib.typ": assemble

// Panels that exist as figs11/<name>.svg; the rest are drawn as placeholders.
#let files = ("a",)

#let body = assemble("figs11", files: files)

#set page(width: 183mm + 6mm, height: auto, margin: (left: 4mm, rest: 2mm))
#body
