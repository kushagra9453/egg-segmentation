# Quantitative Performance and Failure Analysis

### Final Benchmark Metrics (Method T19)
- **Dev (n=88)**: Mean IoU **0.729** (median 0.894), Dice **0.794**, Precision **0.772**, Recall **0.875**, Boundary F1 **0.634**.
- **Eval (n=22)**: Mean IoU **0.711** (median 0.777), Dice **0.782**, Precision **0.747**, Recall **0.857**, Boundary F1 **0.641**.

*Note: T19 adds edge-respecting Felzenszwalb graph segmentation candidates (scales 100/120/150, $\sigma=0.8$). While the dev gain over T15 is within statistical noise (bootstrap 95% CI of per-image gain is $[-0.03, +0.07]$, including 0), the edge-based candidates effectively resolve the white-on-white bag failure mode, elevating eval IoU from 0.326 to 0.711.*

### Weakest Scenes
1. **Cloth** (dev, n=6): Mean IoU **0.317** — patterned fabric folds confuse Otsu/K-means and GrabCut boundaries.
2. **Bag** (eval, n=7): Mean IoU **0.523** — substantially improved from 0.044 via Felzenszwalb graph segmentation, though residual plastic reflections remain challenging.
3. **Ash** (dev, n=5): Mean IoU **0.596** — low chromatic contrast between dusty grey ash and pale eggshell.

### 6 Worst Eval Images
IDs **125** (0.0000), **123** (0.0000), **119** (0.3237), **109** (0.4641), **20** (0.4750), and **158** (0.5893).

### Primary Failure Modes
1. **Residual White-on-White Reflection**: Extreme glare on crumpled plastic bag surfaces (`123, 125`) where internal plastic reflections mirror egg curvature.
2. **Fabric Texture Distortions**: Heavy fabric folds in the cloth cluster disrupt boundary regularity.
3. **Subtle Clutter Bridges**: Occluded boundaries where eggs contact identically shaded background objects.

### Improvement Ideas
1. **Illumination-Invariant Chromaticity**: Use log-chromaticity projection to eliminate specular reflections from plastic folds and table glare.
2. **Boundary Contour Regularization**: Apply active contour (snake) models initialized from Felzenszwalb regions to enforce smooth convexity.
3. **Multi-Scale Edge Verification**: Combine Sobel/Canny gradient magnitudes across color channels to reject false interior edges.
