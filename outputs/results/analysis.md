# Quantitative Performance and Failure Analysis

### Final Benchmark Metrics
- **Dev (n=88)**: Mean IoU **0.667** (median 0.746), Dice **0.757**, Precision **0.776**, Recall **0.788**, Boundary F1 **0.557**.
- **Eval (n=22)**: Mean IoU **0.287** (median 0.001), Dice **0.339**, Precision **0.322**, Recall **0.409**, Boundary F1 **0.307**.

### Weakest Scenes
1. **Bag** (eval, n=7): Mean IoU **0.042** — near-total failure on white eggs against white crumpled plastic bags.
2. **Cloth** (dev, n=6): Mean IoU **0.408** — patterned fabric folds confuse Otsu and GrabCut initialization.
3. **Ash** (dev, n=5): Mean IoU **0.558** — low chromatic contrast between dusty grey ash and pale eggshell.

### 6 Worst Eval Images
IDs **104, 112, 123, 109** (white plastic bag), **155**, and **32** (all six have IoU = **0.0000**).

### Primary Failure Modes
1. **White-on-White Contrast**: Under identical luminance in the bag cluster, Lab distance from border collapses, causing GrabCut to segment bag folds.
2. **Wrong Region Selected**: Shiny non-egg table spots (`32, 155`) with high compactness score higher than diffuse eggs.
3. **Multi-Egg Low Recall**: Clustered eggs (`131, 142, 144`) suffer when secondary eggs fail the score/non-overlap threshold.

### Improvement Ideas
1. **Local Gradient Prior**: Integrate edge-aware bilateral filtering and Canny edge verification to trace boundaries when color distance fails.
2. **Watershed Pre-Partitioning**: Apply distance-transform watershed on candidate regions to robustly isolate clustered, touching eggs.
3. **Illumination-Invariant Chromaticity**: Use log-chromaticity or Retinex decomposition to remove specular glare and shadow boundaries.
