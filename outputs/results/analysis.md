# Quantitative Performance and Failure Analysis

### Final Benchmark Metrics (Method T21)
- **Dev (n=88)**: Mean IoU **0.766** (median 0.896), Dice **0.831**, Precision **0.847**, Recall **0.853**, Boundary F1 **0.668**.
- **Eval (n=22)**: Mean IoU **0.764** (median 0.945), Dice **0.820**, Precision **0.813**, Recall **0.846**, Boundary F1 **0.701**.

*Note: T21 tightens the multi-egg union threshold to 0.85 (up from 0.75 used in T10–T19) after tracing zero-IoU dev images (e.g. ID 108, which had a correct top candidate with IoU 0.93 swamped by a false second egg). This prevents secondary false-positive additions from the expanded Felzenszwalb candidate pool. The gain (+0.04 on dev) is statistically robust: bootstrap 95% CI is $[+0.005, +0.08]$ (excluding 0).*

### Weakest Scenes
1. **Cloth** (dev, n=6): Mean IoU **0.317** — patterned fabric folds confuse Otsu/K-means and GrabCut boundaries.
2. **Ash** (dev, n=5): Mean IoU **0.456** — low chromatic contrast between dusty grey ash and pale eggshell.
3. **Bag** (eval, n=7): Mean IoU **0.682** — dramatically improved from 0.044 through combined Felzenszwalb candidates and disciplined 0.85 union gating.

### 6 Worst Eval Images
IDs **123** (0.0000), **20** (0.0001), **119** (0.4309), **109** (0.5568), **106** (0.6042), and **154** (0.6088).

### Primary Failure Modes
1. **Specular Glare Disruption**: Extreme crumpled surface reflections (`123`, IoU 0.0000) where specular plastic contours outscore diffuse egg boundaries.
2. **Extreme Out-of-Center Positioning**: Small off-center eggs (`20`, IoU 0.0001) heavily penalized by spatial centrality prior.
3. **Fabric Texture Folds**: High-frequency cloth weaves competing with egg curvature.

### Improvement Ideas
1. **Adaptive Center Prior**: Attenuate the centrality penalty when candidate edge contrast is exceptionally sharp.
2. **Specular Highlight Normalization**: Apply log-chromaticity or Retinex preprocessing to cancel specular highlights on glossy surfaces and plastic.
3. **Active Contour Boundary Smoothing**: Regularize post-GrabCut boundaries with parametric curvature penalties.
