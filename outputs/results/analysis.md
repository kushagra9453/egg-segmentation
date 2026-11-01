# Quantitative Performance and Failure Analysis

### Benchmark Metrics

#### Final Adopted Method (Trial T29) — Full Benchmark
- **Dev (n=88)**: Mean IoU **0.773** (median **0.896**), Dice **0.840**, Precision **0.851**, Recall **0.866**, Boundary F1 **0.670**.
- **Eval (n=22)**: Mean IoU **0.782** (median **0.944**), Dice **0.834**, Precision **0.830**, Recall **0.856**, Boundary F1 **0.717**.
- *Improvement over T21*: $+0.0073$ mean IoU on Dev, and $+0.018$ mean IoU on unseen Eval (0.764 $\to$ 0.782), with Boundary F1 rising to 0.717.
- *Method summary*: T21 candidate pool (K-Means Lab + Felzenszwalb graph segmentation) with strict multi-egg union gating ($\ge 0.85 \times \text{best}$), marker-controlled watershed refinement on Lab gradient, equivalent-radius adaptive GrabCut bands, augmented with:
  1. Internal texture penalty ($w_{\text{tex}}=0.05 \times \frac{|\nabla^2 I|_{\text{mean}}}{50}$) to deprioritize high-frequency fabric weaves.
  2. Biological egg area fraction prior ($w_{\text{size}}=0.10$, penalizing candidates outside $0.03 \le \text{area} \le 0.35$).

#### Milestone Baseline Method (Trial T21)
- **Dev (n=88)**: Mean IoU **0.766** (median **0.896**), Dice **0.831** (median **0.945**), Precision **0.847**, Recall **0.853**, Boundary F1 **0.668**.
- **Eval (n=22)**: Mean IoU **0.764** (median **0.945**), Dice **0.820**, Precision **0.813**, Recall **0.846**, Boundary F1 **0.701**.

---

### Eval Disclosure
> **Eval disclosure:** All model development, parameter sweeps, and algorithm decisions (including T29's texture weight and area prior) were tuned strictly and exclusively on the development split (88 images). Held-out evaluation (22 images) was executed only for the intermediate milestone adoptions (T10, T13, T15, T19, T21) and the final evaluation run of T29 upon user request. Zero exploratory tuning or post-hoc threshold selection was performed on the evaluation set.

---

### Failure Analysis on Development Split (T21 Baseline)

A comprehensive audit of all 88 development images under method T21 identified **14 failure cases** ($\text{IoU} < 0.50$):

| Image | IoU | Pred Count | GT Count | Scene Cluster | Failure Cause | Detail / Mechanism |
|---|:---:|:---:|:---:|:---:|---|---|
| `_00004` | 0.0000 | 1 | 1 | cloth | Wrong object chosen | High-contrast fabric fold outscores off-center egg |
| `_00005` | 0.0000 | 1 | 1 | cloth | Wrong object chosen | Woven textile texture forms false elliptical contour |
| `_00052` | 0.0000 | 1 | 1 | cloth | Wrong object chosen | Printed cloth fold pattern selected over true egg |
| `_00027` | 0.0000 | 1 | 1 | ash | White egg on white bg | Pale shell exhibits negligible chromatic $\Delta E$ against dusty ash |
| `_00036` | 0.0000 | 1 | 1 | ash | White egg on white bg | Ash powder blends seamlessly into diffuse egg surface |
| `_00117` | 0.0000 | 1 | 1 | other | Glare | Extreme specular reflection breaks contour continuity |
| `_00150` | 0.0000 | 1 | 1 | other | Wrong object chosen | Bright, compact background ceramic object selected |
| `_00039` | 0.1654 | 1 | 1 | other | Hand / occlusion | Heavy occluding shadow truncates egg mask |
| `_00135` | 0.2078 | 1 | 1 | pan | Glare | Specular rim reflection on pan surface corrupts GrabCut |
| `_00144` | 0.3512 | 1 | 2 | hand | Wrong egg count | Secondary clustered egg rejected by union threshold |
| `_00042` | 0.4021 | 1 | 1 | other | Other (low contrast) | Diffuse penumbra shadow blurs edge boundary |
| `_00121` | 0.4430 | 1 | 1 | hand | Hand / occlusion | Interleaved fingers fragment egg boundary into background |
| `_00126` | 0.4682 | 1 | 2 | other | Wrong egg count | Secondary partially occluded egg falls below union ratio |
| `_00134` | 0.4715 | 1 | 2 | hand | Wrong egg count | Touching egg pair merged or secondary egg excluded |

#### Summary of Failure Causes
1. **Wrong object chosen**: 4 images (`_00004`, `_00005`, `_00052`, `_00150`) — predominantly in textured cloth scenes.
2. **Wrong egg count**: 3 images (`_00144`, `_00126`, `_00134`) — clustered/touching eggs where secondary candidates fall below union ratio.
3. **White egg on white background**: 2 images (`_00027`, `_00036`) — ash scene low contrast.
4. **Glare / Specular reflections**: 2 images (`_00117`, `_00135`) — specular highlights disrupt edge continuity.
5. **Hand / finger occlusion**: 2 images (`_00039`, `_00121`) — fingers breaking continuous boundary.
6. **Other (low contrast)**: 1 image (`_00042`).

#### Diagnostic Overlay Visualizations
Six 4-panel diagnostic overlays (Original Image, Ground Truth Mask, Classical Prediction, False Positive [Red] / False Negative [Blue] Error Map) were generated and saved in [`outputs/results/failures_dev/`](file:///Users/krishna/Downloads/egg-segmentation/outputs/results/failures_dev):
- [`failure_004_cloth.jpg`](file:///Users/krishna/Downloads/egg-segmentation/outputs/results/failures_dev/failure_004_cloth.jpg): Fabric fold selected over off-center egg.
- [`failure_027_ash.jpg`](file:///Users/krishna/Downloads/egg-segmentation/outputs/results/failures_dev/failure_027_ash.jpg): Complete camouflage in dusty ash.
- [`failure_117_other.jpg`](file:///Users/krishna/Downloads/egg-segmentation/outputs/results/failures_dev/failure_117_other.jpg): Specular reflection blinding edge detector.
- [`failure_150_other.jpg`](file:///Users/krishna/Downloads/egg-segmentation/outputs/results/failures_dev/failure_150_other.jpg): Background artifact outscoring egg.
- [`failure_144_hand.jpg`](file:///Users/krishna/Downloads/egg-segmentation/outputs/results/failures_dev/failure_144_hand.jpg): Secondary egg dropped in 2-egg cluster.
- [`failure_121_hand.jpg`](file:///Users/krishna/Downloads/egg-segmentation/outputs/results/failures_dev/failure_121_hand.jpg): Fingers occluding perimeter.

---

### Step 2: Primary Failure Mode Fixes (Trials T22 & T23)

Attempting to resolve the top failure modes (wrong object chosen and wrong egg count):

1. **Trial T22 (Adaptive Union Threshold)**:
   - *Hypothesis*: Relax the secondary egg union threshold from 0.85 to 0.72 specifically for high-quality candidate shapes ($\text{fit} \ge 0.75, \text{solidity} \ge 0.85$) to capture missed secondary eggs.
   - *Outcome*: Dev IoU dropped from **0.766** to **0.710** (median 0.844). False background candidates in single-egg images also possess high elliptical fit, resulting in severe over-segmentation.
   - *Decision*: **DISCARDED**.
2. **Trial T23 (Quadratic Centrality Penalty)**:
   - *Hypothesis*: Replace the linear center distance penalty ($-0.6 \cdot cd$) with a quadratic penalty ($-0.8 \cdot cd^2$) so off-center eggs (e.g., ID 4) are not disproportionately penalized.
   - *Outcome*: Dev IoU dropped from **0.766** to **0.749** (median 0.894). Softening the penalty near borders allowed peripheral clutter in complex backgrounds to outscore central eggs.
   - *Decision*: **DISCARDED**.

---

### Step 3: Boundary Refinement Trials (Single Trial-ID Mapping: 3a–3d $\to$ T24–T28)

To improve Boundary F1 without degrading region overlap, four boundary post-processing techniques were tested on the chosen candidate mask:

| Step | Trial ID | Technique | Dev IoU Mean | Dev IoU Med | Dev Dice Mean | Boundary F1 (2px) | Decision | Reason |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|---|
| Baseline | **T21** | Strict Multi-Egg Union + Radius GrabCut | 0.7659 | 0.8965 | 0.8312 | 0.6677 | **KEPT** | Baseline milestone |
| 3a | **T24** | Post-GrabCut refinement (erode 5 FG, dilate 11 BG, 2 iters) | 0.7677 | 0.8972 | 0.8326 | 0.6713 | DISCARDED | Boundary F1 rose $+0.0036$, but IoU gain $+0.0018 < +0.005$ |
| 3b | **T25** | Morphological smoothing (open 3, close 5, hole fill) | 0.7660 | 0.8970 | 0.8312 | 0.6670 | DISCARDED | No gain over baseline |
| 3c | **T26** | Ellipse-fit shape prior ($\text{IoU} \ge \text{threshold}$) | 0.7581 | 0.8774 | 0.8268 | 0.6215 | DISCARDED | Collapsed Boundary F1 from 0.668 to 0.621; biological asymmetry |
| 3d | **T27** | Gaussian contour smoothing ($k=7, \sigma=1.5$, thr 0.5) | 0.7662 | 0.8972 | 0.8314 | 0.6686 | DISCARDED | Boundary F1 $+0.0009$, IoU $+0.0003 < +0.005$ |
| 3a+3d | **T28** | Combined Post-GrabCut + Gaussian smoothing | 0.7675 | 0.8962 | 0.8323 | 0.6718 | DISCARDED | Boundary F1 $+0.0041$, IoU gain $+0.0016 < +0.005$ |

#### Reconciliation of Ellipse-Fit Numbers (Trial T26 / Step 3c)
- In the initial exploration, two fit thresholds were evaluated:
  - $\text{threshold} = 0.85$: Mean IoU **0.7563**, Median IoU **0.8820**, Mean Dice **0.8252**, Boundary F1 **0.6004**.
  - $\text{threshold} = 0.90$: Mean IoU **0.7581**, Median IoU **0.8774**, Mean Dice **0.8268**, Boundary F1 **0.6215**.
- *Resolution*: The standardized entry in `experiment_log.csv` and benchmark tables is **0.758 / 0.877 / 0.827 / 0.621** (corresponding to the superior 0.90 threshold).
- *Scientific Cause*: True eggs are asymmetric ovals (one tapered pole, one blunt pole). Snapping to mathematical 2D ellipses cuts off natural poles and over-extends along equators, severely penalizing boundary accuracy ($\Delta \text{BoundF1} = -0.046$).

---

### Step 4: Trial T29 — Candidate Texture Penalty and Size Prior

Aiming to resolve the dominant cloth and ash failure modes without boundary distortion:
- **Internal Texture Penalty**: Evaluated mean Laplacian magnitude $\frac{|\nabla^2 I|}{50}$ within candidate masks. High-frequency fabric weaves incur significant penalties, while smooth eggshells incur near-zero penalty.
- **Egg-Size Prior**: Applied piece-wise linear penalty for candidates covering $<3\%$ or $>35\%$ of total frame area.
- **Hyperparameter Sweep on Dev**:
  - $w_{\text{tex}}=0.05, w_{\text{size}}=0.10 \implies$ Dev IoU **0.7732**, Median **0.8970**, Dice **0.8398**, Boundary F1 **0.6728**.
  - Mean IoU improved by $+0.0073$ (surpassing the $+0.005$ keep threshold), and Boundary F1 increased from 0.6677 to 0.6728.
- **Decision**: **KEPT** as the final classical method on Dev.

---

### Scene Performance Breakdown (Dev and Eval)

| Split | Scene | Images | T21 Mean IoU | T29 Mean IoU | Notes |
|---|---|:---:|:---:|:---:|---|
| **Dev** | **Keyboard** | 4 | 0.991 | **0.988** | Flawless contrast |
| **Dev** | **Leaves** | 4 | 0.884 | **0.879** | Resilient against natural clutter |
| **Dev** | **Tub** | 5 | 0.885 | **0.879** | Clear specular separation |
| **Dev** | **Hand** | 16 | 0.819 | **0.824** | Robust to finger occlusion |
| **Dev** | **Other** | 40 | 0.803 | **0.805** | High general precision |
| **Dev** | **Pan** | 8 | 0.741 | **0.757** | Metallic reflection suppression |
| **Dev** | **Ash** | 5 | 0.456 | **0.456** | Chromatic camouflage persists |
| **Dev** | **Cloth** | 6 | 0.317 | **0.411** | Texture penalty suppresses fabric weaves (+0.094 gain) |
| **Eval** | **Bag** | 7 | 0.682 | **0.681** | Rescued by Felzenszwalb candidates |
| **Eval** | **Other** | 15 | 0.803 | **0.829** | Substantial generalization gain (+0.026) |

---

### 6 Worst Eval Images (Method T29)
1. `_00123` (IoU: **0.0000**): Extreme specular crumples on reflective white plastic bag.
2. `_00020` (IoU: **0.0000**): Extreme peripheral position on border margin.
3. `_00109` (IoU: **0.5568**): Heavy plastic sheen and crease lines.
4. `_00106` (IoU: **0.6042**): Bag boundary merge.
5. `_00154` (IoU: **0.6088**): Low contrast illumination.
6. `_00045` (IoU: **0.6709**): Partial background shadow.

*(Note: Image `_00119`, previously among the bottom 6 at IoU 0.4309 under T21, improved to **0.8449** under T29 due to texture/size regularization).*
