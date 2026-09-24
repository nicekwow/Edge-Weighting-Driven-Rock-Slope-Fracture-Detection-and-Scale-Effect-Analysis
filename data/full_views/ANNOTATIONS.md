# Full-view images and reference annotations

Images a, b, and c correspond to `a11`, `a13`, and `a14`, respectively. Each view includes the source PNG, a LabelMe JSON annotation, and a binary `*_mask.png` reference mask. In the masks, 0 denotes background and 255 denotes fracture. In the JSON files, `fracture` polygons mark fractures, while `background` polygons represent holes within them.

The reference annotations were reconstructed from an earlier display panel for Figure 13 and mapped to the original 2560 × 1440 image grid. Small isolated fragments were excluded. Because the display panel had been downsampled and rasterized, these masks should not be treated as the unavailable original full-resolution annotations. Metrics based on these masks must be recomputed from the published files.
