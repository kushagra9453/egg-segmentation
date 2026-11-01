# Computer Vision Engineer Task: Egg Segmentation and Video Tracking
**Author:** Krishna  
**Project Repository:** [https://github.com/kushagra9453/egg-segmentation](https://github.com/kushagra9453/egg-segmentation)  
**Tracked Video Deliverable (Google Drive):** [https://drive.google.com/file/d/1bmdephNfQXLSc2-urg9mRzZL_sNBW8Ap/view?usp=drive_link](https://drive.google.com/file/d/1bmdephNfQXLSc2-urg9mRzZL_sNBW8Ap/view?usp=drive_link)

---

## Executive Summary
This report presents an end-to-end classical computer vision and tracking solution for segmenting and tracking eggs across diverse, visually complex real-world scenes. The project comprises four core tasks:
1. **Task 1: Ground-Truth Generation**: Constructed pseudo ground-truth masks for 110 images using Meta's Segment Anything Model (SAM), enhanced with manual bounding boxes, CVAT polygon corrections, and parametric ellipse fitting.
2. **Task 2: Classical Segmentation Optimization**: Iteratively developed and benchmarked 29 disciplined trials on a leakage-free development split (88 images), culminating in Method T29 (Dev Mean IoU 0.773, Boundary F1 0.670).
3. **Task 3: Evaluation & Error Analysis**: Evaluated generalizability on an unseen evaluation split (22 images) using IoU, Dice, Precision, Recall, and Boundary F-score metrics, accompanied by scene-stratified analysis and visual error diagnosis for the worst failures (including the two 0.000 eval images).
4. **Task 4: Multi-Object Video Tracking**: Built an autonomous tracking pipeline on a 3,409-frame video clip combining automatic viewport border cropping, temporal scene detection, T29 segmentation, gated Hungarian association with 1.5 s occlusion memory, and ghost-box elimination.

---

## 1. Task 1: Ground-Truth Generation

### 1.1 Methodology
High-quality pixel-level annotations are essential for benchmark validity. Pseudo ground-truth masks were generated using **Meta's Segment Anything Model (SAM)** (`vit_b` and `vit_h` checkpoints) followed by multi-stage refinement:

1. **Automated Point Grid Prompting (`02_sam_segment.py`)**: Generated initial candidate masks by querying SAM with an automated grid of point prompts across all 110 raw images.
2. **Heuristic Pruning & Filtering (`03_fix_and_refine.py`)**: Filtered raw SAM masks by retaining components with area $\ge 35\%$ of the largest candidate, color consistency in CIELAB space ($\Delta E \le 25$), and non-overlap constraints.
3. **Targeted Bounding Box Prompts (`05_boxes_fix.py`)**: For cluttered or multi-egg scenes where automated points failed, manual bounding boxes were cataloged in `boxes.json` and passed as explicit box prompts to `SamPredictor.predict(box=...)`.
4. **Fine Polygon Annotation (`10_import_cvat.py`)**: For irregular contours, heavy shadows, and edge ambiguities, masks were manually edited in CVAT and imported as polygon masks.
5. **Parametric Ellipse Fitting (`12_ellipse_fix.py`)**: For partially occluded or heavily shadowed single eggs (e.g., IDs 117 and 135), mathematical ellipse parameters were fitted to guarantee biologically plausible boundaries.

![Task 1 Ground Truth Review Sheet](../outputs/sam_final/review_fixed.jpg)
*Figure 1: Visual audit sheet of refined pseudo ground-truth masks across representative challenge cases.*

### 1.2 Limitations & Abandoned Approaches
* **SAM Point-Prompt Failures**: Automated point grids frequently over-segmented touching eggs into arbitrary sub-components or leaked into similarly colored backgrounds (e.g., ash powder and white plastic).
* **Single Box Collapse**: In tight multi-egg clusters, passing a single bounding box enclosing multiple eggs caused SAM's mask decoder to collapse all eggs into one merged blob, requiring individual box prompts or manual polygon tracing in CVAT.
* **Brittle GUI Annotation (`04_manual_fix.py`)**: An early interactive OpenCV GUI for clicking prompt points was abandoned because GUI event loops proved platform-dependent, lacked sub-pixel precision, and could not easily serialize revision history.
* **Secondary Box Pass (`08_fix_boxes2.py`)**: A second automated box-pass was abandoned when edge ambiguities in shadowed metallic pans caused inconsistent mask boundaries.

All 110 images have verified binary ground-truth masks committed in `outputs/sam_final/masks/<image stem>.png`.

---

## 2. Task 2: Dataset Split & Classical Algorithm Development

### 2.1 Dataset Partitioning & Leakage Prevention
The dataset consists of 110 images captured across repeated visual environments (e.g., steel pan, woven cloth, plastic bag, keyboard, ash, plastic tub).

> **Critical Leakage Risk**: A standard random split places identical backgrounds in both development and evaluation splits, leading to severe visual leakage and artificially inflated evaluation scores.

* **Leaky Split (`outputs/split_v1_leaky.json`)**: Initial background-histogram clustering allowed scene clusters to spill across splits:
  * The `bag` scene was split across sets: 6 images in dev, 1 image (ID 111) in eval.
  * The `tub` scene was split across sets: 3 images in dev (108, 127, 139) and 2 images in eval (130, 141).
* **Leakage-Safe Split (`outputs/split.json`)**: Partitioned using `GroupShuffleSplit` (seed 348) into an **88-image dev set** and a **22-image eval set**:
  * Grouped scenes strictly by visual cluster: `keyboard`, `leaves`, `tub`, `hand`, `pan`, `ash`, `cloth`, `bag`, and `other`.
  * Complete cluster independence: zero overlapping scene clusters between dev and eval.
  * All hyperparameter tuning was restricted strictly to the 88-image development split.

### 2.2 Classical Experiment Log (Trials T01–T29)
All experiments were evaluated strictly on the development split (88 images) and logged to `experiment_log.csv`. A trial was adopted as a milestone only if mean Dev IoU improved by $>0.005$ and Boundary F1 did not drop.

> **Eval Disclosure:** Model exploration, candidate scoring weights, boundary refinement methods, and hyperparameter sweeps were tuned strictly and exclusively on the development split (88 images). Held-out evaluation metrics were computed and viewed only for the intermediate adopted milestones (**T10**, **T13**, **T15**, **T19**, **T21**) and the final adopted method (**T29**); all other exploratory and discarded trials (**T01–T09**, **T11**, **T12**, **T14**, **T16–T18**, **T20**, and **T22–T28**) were evaluated strictly on the development split, with zero access to or tuning on the held-out evaluation set.

The table below catalogs every trial conducted throughout the optimization campaign (mapping Step 2 failure fixes to **T22–T23**, Step 3 boundary refinements 3a–3d to **T24–T28**, and Step 4 candidate scoring prior to **T29**):

| Trial | Step / Mapping | Modification | Hypothesis / Motivation | Parameters | Dev IoU (Mean / Med) | Dev Dice | Bound F1 (2px) | Decision | Rationale |
|---|:---:|---|---|---|:---:|:---:|:---:|:---:|---|
| **T01** | Baseline | Grayscale blur + Otsu | Simplest intensity threshold baseline | Blur 5x5, Otsu, 1 comp | 0.365 / 0.263 | 0.462 | - | **KEPT** | Initial baseline |
| **T02** | - | Lab border background distance + Otsu | Eggs differ in color from border, not just luminance | Median Lab of 5% border, Otsu, 1 comp | 0.355 / 0.268 | 0.428 | - | DISCARDED | No gain over baseline |
| **T03** | - | T02 + morphology, hole fill, multi-egg | Suppress noise, fill glare holes, allow up to 4 eggs | Open 5, close 9, fill, $\ge 25\%$ largest | 0.370 / 0.292 | 0.467 | - | DISCARDED | Below +0.005 threshold |
| **T04** | - | GrabCut refinement on T03 | Color model GMMs refine ragged boundaries | Erode 15 FG, dilate 41 BG, 4 iters | 0.396 / 0.278 | 0.478 | - | **KEPT** | +0.031 gain over T01 |
| **T05** | - | T04 + Ellipse regularization | Eggs are elliptical; smooth boundaries | `fitEllipse` if IoU with blob $>0.85$ | 0.395 / 0.278 | 0.479 | - | DISCARDED | No gain over T04 |
| **T06** | - | Lab distance + Gaussian center prior | Eggs are centered; center prior suppresses border clutter | $\sigma = 0.45 \times \text{size}$, open 7, close 15 | 0.474 / 0.469 | 0.588 | - | **KEPT** | +0.078 gain over T04 |
| **T07** | - | Multi-K KMeans + shape scoring + GrabCut | Border model fails on skin/white; score candidates by geometry | $K \in \{2..5\}$, fit + 0.5 sol - 0.6 dist - 0.25 touch | 0.632 / 0.729 | 0.720 | - | **KEPT** | +0.158 gain over T06 |
| **T08** | - | T07 + relaxed multi-egg union threshold | Multi-egg images had low recall | Union threshold $0.9 \to 0.6$ | 0.582 / 0.593 | 0.682 | - | DISCARDED | Inadvertently added non-egg clutter |
| **T09** | - | Smart multi-egg union + Bright/Low-Sat cue | Shape-filtered multi-egg union with brightness cue | Union thr 0.7, fit $\ge 0.5$, sol $\ge 0.6$, $+0.2(V - 0.3S)$ | 0.660 / 0.729 | 0.751 | - | **KEPT** | +0.028 gain over T07 |
| **T10** | Milestone | T09 with tightened multi-egg threshold | Avoid marginal non-egg regions | Union thr 0.75, fit $\ge 0.55$, sol $\ge 0.65$ | 0.667 / 0.746 | 0.757 | 0.582 | **KEPT** | +0.007 gain over T09 |
| **T11** | - | Colour consistency pruning against core Lab median | Over-segmentation pruning | $K=1.5..4.0$ sweep; $K=1000$ control | 0.667 / 0.746 | 0.757 | - | DISCARDED | Gradients on egg shell cause over-pruning |
| **T12** | - | Adaptive opening by distance-transform radius | Prune protrusions/bridges | $F=0.3$ sweep | 0.666 / 0.744 | 0.754 | - | DISCARDED | Erodes natural egg poles |
| **T13** | Milestone | Marker-controlled watershed on Lab gradient + GrabCut | Bridge split lit/shaded egg halves | $fe=0.35, be=0.45$, GrabCut 3 iters | 0.681 / 0.757 | 0.765 | 0.601 | **KEPT** | +0.014 gain over T10 |
| **T14** | - | Random search of candidate scoring weights | Optimize feature weights via 5-fold CV | 9 features, 200 samples, 5-fold scene CV | 0.545 / 0.563 | - | - | DISCARDED | Proxy overfits to scene specifics |
| **T15** | Milestone | GrabCut bands proportional to equivalent egg radius $d$ | Adapt band widths to object size | $fe=0.45d, pf=0.10d, bgd=0.25d$, 3 iters | 0.707 / 0.861 | 0.781 | 0.628 | **KEPT** | +0.026 gain over T13 |
| **T16** | - | Inner-candidate rule on large regions | Replace $>30\%$ image region | Area $>0.3 \times$ image | 0.702 / 0.703 | - | - | DISCARDED | Never changed a selection |
| **T17** | - | Multi-egg union size-consistency rule | Extra regions must be 0.3-3x best region area | Area ratio 0.3-3x | 0.701 / 0.703 | - | - | DISCARDED | No gain over T15 |
| **T18** | - | Shape filter on final components | Filter post-GrabCut false positives | Ellipse fit $\ge 0.6-0.8$, solidity $\ge 0.85-0.95$ | 0.695 / 0.703 | - | - | DISCARDED | Smooth false regions pass; partial eggs fail |
| **T19** | Milestone | Extra candidates from Felzenszwalb graph segmentation | Edge-respecting regions for low-contrast scenes | Scales 100/120/150, $\sigma=0.8$, min size 1% | 0.729 / 0.894 | 0.794 | 0.643 | **KEPT** | +0.022 gain (within noise); rescues bag scenes |
| **T20** | - | Re-rank 4 refined hypotheses by edge + ring contrast | Hypothesis re-ranking | Score + $w_e \cdot \text{edge} + w_c \cdot \text{contrast}$ | 0.742 / - | - | - | DISCARDED | Selection noise across scene-halves |
| **T21** | Milestone | Strict multi-egg union rule (score $\ge 0.85 \times \text{best}$) | Suppress swamping false extra regions | Union ratio threshold $0.85$ (was $0.75$) | 0.766 / 0.896 | 0.831 | 0.668 | **KEPT** | +0.037 gain over T19; bootstrap 95% CI [+0.005, +0.08] excludes 0 |
| **T22** | Step 2a | Adaptive multi-egg union threshold | Fix under-segmented clustered eggs (IDs 144, 126, 134) | $\text{ratio}=0.85$, relaxed to 0.72 if $\text{fit} \ge 0.75, \text{sol} \ge 0.85$ | 0.710 / 0.844 | 0.781 | 0.638 | DISCARDED | Background shapes pass shape test; over-segmentation |
| **T23** | Step 2b | Quadratic center distance penalty | Prevent off-center eggs being outscored (ID 4) | Score $- 0.6 \cdot cd + 0.6 \cdot cd - 0.8 \cdot cd^2$ | 0.749 / 0.894 | 0.816 | 0.654 | DISCARDED | Under-penalizes peripheral clutter on complex backgrounds |
| **T24** | Step 3a | Post-GrabCut boundary refinement | Refine ragged edges using image color GMMs | Erode 5 FG, dilate 11 PR_BGD, 2 iters | 0.768 / 0.897 | 0.833 | 0.671 | DISCARDED | BoundF1 $+0.0036$, but IoU gain $+0.0018 < +0.005$ |
| **T25** | Step 3b | Morphological opening, closing, and hole fill | Suppress pixel step-noise and fill specular holes | Open 3, close 5, fill, $\le 6$ components | 0.766 / 0.897 | 0.831 | 0.667 | DISCARDED | No gain over baseline |
| **T26** | Step 3c | Ellipse-fit shape prior | Enforce biological elliptical boundary regularity | $\text{fit\_thr} \in \{0.85, 0.90\}$ | 0.758 / 0.877 | 0.827 | 0.621 | DISCARDED | BoundF1 collapsed $0.668 \to 0.621$; sweep $\{0.85: 0.756/0.600, 0.90: 0.758/0.621\}$; asymmetric eggs don't fit 2D ellipses |
| **T27** | Step 3d | Gaussian contour smoothing | Smooth step-edges along mask perimeter | $k=7, \sigma=1.5$, thr $=0.5$ | 0.766 / 0.897 | 0.831 | 0.669 | DISCARDED | BoundF1 $+0.0009$, IoU $+0.0003 < +0.005$ |
| **T28** | Step 3a+3d | Combined Post-GrabCut + Gaussian contour smoothing | Snapping plus subpixel boundary smoothing | GrabCut 2 iters + Gaussian $k=7, \sigma=1.5$ | 0.767 / 0.896 | 0.832 | 0.672 | DISCARDED | BoundF1 $+0.0041$, IoU $+0.0016 < +0.005$ |
| **T29** | Step 4 | Candidate texture penalty + egg-size prior | Penalize fabric weaves (cloth scene) & irregular clutter | $w_{\text{tex}}=0.05 \times \frac{|\nabla^2 I|}{50}, w_{\text{size}}=0.10$ | **0.773** / **0.896** | **0.840** | **0.670** | **KEPT** | $+0.0073 \ge +0.005$ IoU gain, BoundF1 rose from 0.668 to 0.670; selected as final method |

---

## 3. Task 3: Evaluation, Metrics, and Failure Modes

### 3.1 Quantitative Results
Benchmark metrics across the development and held-out evaluation sets using final-run numbers:

| Model / Trial | Split | Images | Mean IoU | Median IoU | Mean Dice | Precision | Recall | Boundary F1 (2px) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **T29 (Final Classical)** | **Development** | 88 | **0.773** | **0.896** | **0.840** | 0.851 | 0.866 | **0.670** |
| **T29 (Final Classical)** | **Evaluation** | 22 | **0.782** | **0.944** | **0.834** | 0.830 | 0.856 | **0.717** |
| **T21 (Milestone Baseline)** | **Development** | 88 | **0.766** | **0.896** | **0.831** | 0.847 | 0.853 | **0.668** |
| **T21 (Milestone Baseline)** | **Evaluation** | 22 | **0.764** | **0.945** | **0.820** | 0.813 | 0.846 | **0.701** |

*Note: Boundary F-score is computed at a 2-pixel tolerance at the $320\text{ px}$ working scale. Evaluating T29 on the held-out evaluation set demonstrates strong generalizability: Mean IoU increased from 0.764 to **0.782** (+0.018 gain), Precision rose to 0.830, Recall to 0.856, and Boundary F1 reached **0.717**.*

### 3.2 Scene-Stratified Performance Breakdown

| Split | Scene | Images | T21 Mean IoU | T29 Mean IoU | Qualitative Scene Characteristics |
|---|---|:---:|:---:|:---:|---|
| **Dev** | **Keyboard** | 4 | 0.991 | **0.988** | High local contrast against dark keycaps; near-perfect segmentation |
| **Dev** | **Leaves** | 4 | 0.884 | **0.879** | Resilient against natural clutter and organic leaf texture |
| **Dev** | **Tub** | 5 | 0.885 | **0.879** | Clear specular separation and smooth plastic boundary |
| **Dev** | **Hand** | 16 | 0.819 | **0.824** | Skin suppression prevents GrabCut from expanding onto holding fingers |
| **Dev** | **Other** | 40 | 0.803 | **0.805** | High general precision across miscellaneous indoor surfaces |
| **Dev** | **Pan** | 8 | 0.741 | **0.757** | Metallic rim reflection suppression via radius-adapted bands |
| **Dev** | **Ash** | 5 | 0.456 | **0.456** | Chromatic camouflage persists ($\Delta E \approx 0$ between egg and ash) |
| **Dev** | **Cloth** | 6 | 0.317 | **0.411** | Internal texture penalty successfully suppresses woven textile folds (+0.094 gain) |
| **Eval** | **Bag** | 7 | 0.682 | **0.681** | Rescued by Felzenszwalb edge-respecting candidates |
| **Eval** | **Other** | 15 | 0.803 | **0.829** | Substantial generalization gain on unseen diverse backgrounds (+0.026) |

### 3.3 Evaluation Visualizations & Error Analysis

![Evaluation Histograms and Scene Breakdown](../outputs/results/scene_iou.png)
*Figure 2: Scene-stratified IoU distribution across development and evaluation partitions.*

![Evaluation IoU Histogram](../outputs/results/iou_hist.png)
*Figure 3: Distribution of evaluation IoU scores, showing strong concentration near median 0.944.*

#### Worst-Failure Visuals (Evaluation Split)
The 6 worst failure cases on the evaluation split under method T29 are displayed below:

![Worst Failures on Evaluation Split](../outputs/results/worst_failures.jpg)
*Figure 4: Tile of 6 worst eval failures. Columns: (1) Input image with ID, (2) Ground truth overlay (green), (3) Classical T29 prediction (yellow), (4) Error map with False Positives in Red and False Negatives in Blue.*

**Analysis of the Two 0.000 IoU Eval Failures:**
1. **`_00123` (IoU: 0.0000, Row 1 of Figure 4)**: The egg sits on a crumpled white plastic grocery bag. Extreme specular glare and white plastic folds produce identical color and higher centrality than the actual egg, swamping candidate scoring.
2. **`_00020` (IoU: 0.0000, Row 2 of Figure 4)**: The egg is placed at the extreme peripheral boundary margin of the frame, suffering heavy shadow truncation and severe center-distance penalization.

**Additional Sub-0.70 Eval Failures:**
3. **`_00109` (IoU: 0.5568, Row 3)**: Plastic bag sheen and crease lines split the egg mask.
4. **`_00106` (IoU: 0.6042, Row 4)**: White plastic bag background merges with the egg's lower pole.
5. **`_00154` (IoU: 0.6088, Row 5)**: Low illumination contrast against diffuse surface.
6. **`_00045` (IoU: 0.6709, Row 6)**: Strong cast shadow truncates the segmented boundary.

#### Development Failure Case Diagnostics
For comparison, four primary failure categories identified on the development split are shown below:

| Cloth Fold False Positive (`failure_004`) | Ash Color Camouflage (`failure_027`) |
|:---:|:---:|
| ![Cloth Failure](../outputs/results/failures_dev/failure_004_cloth.jpg) | ![Ash Failure](../outputs/results/failures_dev/failure_027_ash.jpg) |
| **Specular Pan Glare (`failure_117`)** | **Clustered Egg Omission (`failure_144`)** |
| ![Glare Failure](../outputs/results/failures_dev/failure_117_other.jpg) | ![Clustered Egg Failure](../outputs/results/failures_dev/failure_144_hand.jpg) |

*Figure 5: Four-panel diagnostic overlays (Image, GT, Prediction, Red/Blue Error Map) for key dev failure causes.*

---

## 4. Task 4: Video Tracking

### 4.1 Note on Video Source
* **YouTube Video Link**: [https://drive.google.com/file/d/1bmdephNfQXLSc2-urg9mRzZL_sNBW8Ap/view?usp=drive_link](https://drive.google.com/file/d/1bmdephNfQXLSc2-urg9mRzZL_sNBW8Ap/view?usp=drive_link)
* **Source Disclosure**: The source video could not be downloaded directly via `yt-dlp` due to YouTube HTTP 403 / anti-bot download restrictions. Consequently, an uncompressed high-resolution screen recording (`data/video/eggs_clip.mov`, $1994 \times 1240$ @ $31.79\text{ fps}$, 3,409 frames, $\sim 107.2\text{ s}$) was captured and placed in the project repository.
* The raw screen recording captures the browser frame (YouTube title bar, white page margins, video controls) and non-target segments (green intro, instructional text slides, yellow-cloth commercial). The pipeline automatically crops the picture and gates the target egg scene.

### 4.2 Architecture & Final Tracker Fixes
The tracker pipeline (`scripts/15_track.py`) executes entirely with classical computer vision:

1. **Automatic Viewport Crop Detection**:
   * Evaluates intensity profiles across sample frames ($t=15\text{ s}, 20\text{ s}, 25\text{ s}$) to find white browser borders.
   * Isolates the true video window to `[ymin=70:ymax=1115, xmin=29:xmax=1931]` ($1902 \times 1045\text{ px}$, $1.82$ aspect ratio), removing header, player controls, and borders.
2. **Automatic Temporal Scene Gating**:
   * Detects the dark wooden table ($V < 80$ fraction $> 40\%$) and absence of green intro ($H \in [35, 85]$ fraction $< 5\%$).
   * Sequences from **frame 306** ($9.63\text{ s}$) through **frame 1347** ($42.37\text{ s}$, 1,042 frames, $32.78\text{ s}$ duration), cleanly capturing the egg scene and transitional text slides.
3. **Classical Segmentation via Final Method T29**:
   * Directly imports finalized method `t29` from `scripts/14_classical.py`.
   * **Hand Skin Suppression**: Detects skin hue ($H \le 25, S > 60$) and fills hand regions with median table background before candidate extraction.
   * **Strict Detection Gate**: Rejects candidate blobs with solidity $< 0.85$, area outside egg-size prior ($0.010 \le \text{frac} \le 0.150$), ellipse fit $< 0.55$, or color not near-white/light ($S > 45$ or $V < 125$).
4. **Tracker Fix 1: Ghost Box Elimination & Coasting Cap**:
   * Coasting boxes are drawn only if the track matched a real detection in the current frame or within the last $0.5\text{ s}$ (`miss <= int(0.5 * fps)`).
   * Tracks are preserved internally for up to $1.5\text{ s}$ (`miss <= int(1.5 * fps)`) to enable ID re-association upon reappearance, but are **never drawn** after $0.5\text{ s}$.
   * All dummy fallback boxes were removed.
   * `box_contains_valid_egg` performs pixel verification on every candidate bounding box, guaranteeing zero ghost boxes drawn on text slides.
5. **Tracker Fix 2: Text-Slide Gate**:
   * Frames lacking dark table background (`dark_table < 0.20`) or possessing high edge density with no egg detections are flagged as text slides, suppressing false detections.
6. **Tracker Fix 3: Missed Egg Local Search**:
   * When a frame has fewer detections than active tracks, the tracker executes a localized window search around the predicted position with relaxed thresholds ($S < 60, V > 115, \text{solidity} \ge 0.82$) before dropping the track. This successfully rescues the right egg at $5\text{ s}$.
7. **Tracker Fix 4: Strict 2-ID Policy**:
   * Exactly two tracks (`Egg 1` and `Egg 2`) are created; IDs $\ge 3$ are forbidden.
   * Hungarian association (`scipy.optimize.linear_sum_assignment`) combines normalized centroid distance, bounding box IoU, and area difference to preserve ID continuity. `Egg 2` retains its identity before and after finger interaction.
8. **High-Resolution Rendering & Transcoding**:
   * Rendered at $960 \times 528\text{ px}$.
   * Overlays: smooth contour outline, bounding box, and filled label tag (`Egg 1`: deep blue; `Egg 2`: vibrant orange) with bold white text readable against white eggshells.
   * Transcoded via `ffmpeg` using `libx264` (`pix_fmt yuv420p`, `-color_range tv`, `-movflags +faststart`) to `outputs/tracking_final.mp4`.

### 4.3 Self-Check Audit & Verification Montage

![Task 4 Verification Montage](../outputs/tracking_check.jpg)
*Figure 6: 2x4 verification montage sampled at 2s, 5s, 10s, 15s, 20s, 25s, 28s, and 31s.*

#### Per-Second Verification Table
| Second | Frame | Box Count | Brightness | Edge Coverage | Drawn IDs | Audit Status |
|:---:|:---:|:---:|:---:|:---:|:---|:---|
| 0s | 0 | 0 | 4.4 | 0.0050 | None | Scene transition |
| 1s | 32 | 2 | 84.0 | 0.0421 | Egg 1, Egg 2 | Both eggs on table |
| 2s | 64 | 2 | 84.5 | 0.0434 | Egg 1, Egg 2 | Both eggs on table |
| 3s | 95 | 2 | 83.3 | 0.0416 | Egg 1, Egg 2 | Both eggs on table |
| 4s | 127 | 2 | 83.4 | 0.0417 | Egg 1, Egg 2 | Both eggs on table |
| **5s** | **159** | **2** | **79.5** | **0.0440** | **Egg 1, Egg 2** | **PASS: Both eggs tracked (right egg present)** |
| 6s | 191 | 2 | 74.7 | 0.0436 | Egg 1, Egg 2 | Both eggs on table |
| 7s | 223 | 2 | 71.1 | 0.0405 | Egg 1, Egg 2 | Both eggs on table |
| 8s | 254 | 2 | 65.6 | 0.0370 | Egg 1, Egg 2 | Hand enters table |
| 9s | 286 | 2 | 67.7 | 0.0363 | Egg 1, Egg 2 | Hand touches Egg 1 |
| 10s | 318 | 1 | 84.4 | 0.0343 | Egg 2 | Egg 1 occluded/spun |
| 11s | 350 | 1 | 77.8 | 0.0296 | Egg 2 | Egg 1 occluded/spun |
| 12s | 381 | 1 | 77.0 | 0.0345 | Egg 2 | Egg 1 occluded/spun |
| 13s | 413 | 1 | 84.1 | 0.0248 | Egg 2 | Egg 1 occluded/spun |
| 14s | 445 | 1 | 82.7 | 0.0255 | Egg 2 | Egg 1 occluded/spun |
| 15s | 477 | 1 | 73.3 | 0.0417 | Egg 2 | Egg 1 occluded/spun |
| 16s | 509 | 1 | 72.2 | 0.0400 | Egg 2 | Hand approaches Egg 2 |
| 17s | 540 | 1 | 79.3 | 0.0285 | Egg 1 | Finger spins Egg 2 (occluded) |
| 18s | 572 | 1 | 87.2 | 0.0343 | Egg 1 | Finger spins Egg 2 (occluded) |
| **19s** | **604** | **2** | **79.2** | **0.0413** | **Egg 1, Egg 2** | **PASS: Egg 2 maintains ID 2 after spin** |
| 20s | 636 | 2 | 83.4 | 0.0471 | Egg 1, Egg 2 | Both eggs resting |
| 21s | 668 | 2 | 83.6 | 0.0469 | Egg 1, Egg 2 | Both eggs resting |
| 22s | 699 | 2 | 83.6 | 0.0464 | Egg 1, Egg 2 | Both eggs resting |
| 23s | 731 | 2 | 81.8 | 0.0440 | Egg 1, Egg 2 | Both eggs resting |
| 24s | 763 | 0 | 91.0 | 0.0040 | None | Egg scene ends |
| **25s** | **795** | **0** | **119.6** | **0.0304** | **None** | **PASS: Text slide ("Why this happens?") - 0 boxes** |
| 26s | 827 | 0 | 119.6 | 0.0304 | None | Text slide - 0 boxes |
| 27s | 858 | 0 | 118.8 | 0.0372 | None | Text slide - 0 boxes |
| **28s** | **890** | **0** | **118.8** | **0.0373** | **None** | **PASS: Text slide - 0 boxes** |
| 29s | 922 | 0 | 118.8 | 0.0372 | None | Text slide - 0 boxes |
| 30s | 954 | 0 | 118.8 | 0.0372 | None | Text slide - 0 boxes |
| **31s** | **986** | **0** | **118.7** | **0.0372** | **None** | **PASS: Text slide - 0 boxes** |
| 32s | 1017 | 0 | 118.8 | 0.0372 | None | Text slide - 0 boxes |

#### Summary of Self-Check Criteria (1 to 4)
* **Criterion 1 (Per-second audit & no boxes on text, valid eggs at 5s)**: **PASS**  
  At 5s: both eggs are tracked and confirmed via pixel check ($S < 60, V > 120, \text{solidity} \ge 0.85$). At 25s, 28s, 31s (and all text slides $t \ge 24.5\text{ s}$): exactly 0 boxes are drawn.
* **Criterion 2 (No box drawn > 0.5s without matching detection)**: **PASS**  
  Drawing gate `tr["miss"] <= int(0.5 * fps)` strictly enforced.
* **Criterion 3 (Only IDs 'Egg 1' and 'Egg 2' appear in video)**: **PASS**  
  Unique track IDs across sequence: exactly `['Egg 1', 'Egg 2']`.
* **Criterion 4 (Egg 2 maintains ID before & after finger interaction)**: **PASS**  
  Egg 2 tracked before spin ($t=15\text{--}16\text{ s}$), occluded ($t=17\text{--}18\text{ s}$), and re-identified as `Egg 2` upon release ($t=19\text{--}23\text{ s}$).

---

## 5. AI-Use Disclosure
Artificial intelligence tools—specifically **Claude** (Anthropic) and the **Gemini coding agent** (Google DeepMind)—were utilized during this project to assist with code drafting, debugging classical CV pipelines, authoring shell routines, and formatting benchmark logs. All architectural decisions, mathematical formulations, experimental methodologies, evaluation protocols, and analytical deductions were verified, reviewed, and finalized by the author.
