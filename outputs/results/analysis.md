# Quantitative Performance and Failure Analysis

### Final Benchmark Metrics
- **Dev (n=88)**: Mean IoU **0.707** (median 0.861), Dice **0.781**, Precision **0.773**, Recall **0.845**, Boundary F1 **0.614**.
- **Eval (n=22)**: Mean IoU **0.326** (median 0.001), Dice **0.368**, Precision **0.348**, Recall **0.430**, Boundary F1 **0.358**.

### Weakest Scenes
1. **Bag** (eval, n=7): Mean IoU **0.044** — near-total failure on white eggs against white crumpled plastic bags.
2. **Cloth** (dev, n=6): Mean IoU **0.357** — patterned fabric folds confuse Otsu and GrabCut initialization.
3. **Ash** (dev, n=5): Mean IoU **0.539** — low chromatic contrast between dusty grey ash and pale eggshell.

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
