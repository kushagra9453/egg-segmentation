# Computer Vision Intern Task: Egg Segmentation and Video Tracking
**Author:** Krishna  
**Project Repository:** [https://github.com/kushagra9453/egg-segmentation](https://github.com/kushagra9453/egg-segmentation)

---

## Executive Summary
This report presents an end-to-end computer vision solution for segmenting and tracking eggs across diverse, challenging real-world scenes. The project is organized into four tasks:
1. **Task 1: Ground-Truth Generation** utilizing Meta's Segment Anything Model (SAM) enhanced with manual bounding boxes, CVAT polygon corrections, and parametric ellipse fitting.
2. **Task 2: Classical Segmentation Optimization** developed iteratively over 29 disciplined trials on a leakage-free development split (88 images), culminating in Method T29 (Dev IoU 0.773, Boundary F1 0.673).
3. **Task 3: Evaluation and Failure Analysis** on an unseen evaluation split (22 images) using IoU, Dice, Precision, Recall, and Boundary F-score metrics alongside scene-stratified breakdown.
4. **Task 4: Multi-Object Video Tracking** executing on a 3,409-frame video clip combining classical segmentation, distance-transform watershed splitting, and gated Hungarian matching.

---

## 1. Task 1: Ground-Truth Generation

### 1.1 Methodology
High-quality pixel-level annotations are essential for benchmark validity. Since training manual annotators from scratch is resource-intensive, we constructed pseudo ground-truth masks using **Meta's Segment Anything Model (SAM)** (`vit_b` and `vit_h` checkpoints), followed by multi-stage refinement:

1. **Automated Generation (`02_sam_segment.py`)**: Generated candidate segmentations per image by querying SAM with an automated grid of point prompts.
2. **Heuristic Pruning (`03_fix_and_refine.py`)**: Applied component filtering to isolate egg candidates based on relative area ($\ge 35\%$ of largest component), color consistency in Lab space ($\Delta E \le 25$), and non-overlap constraints.
3. **Targeted Prompts (`05_boxes_fix.py`)**: For challenging multi-egg or cluttered images, bounding box coordinates were manually recorded in `boxes.json` and fed into `SamPredictor.predict(box=...)`.
4. **Fine Polygon Annotation (`10_import_cvat.py`)**: For irregular boundaries, partial occlusions, and edge ambiguities, annotations were refined in CVAT and imported as polygon masks.
5. **Parametric Ellipse Fitting (`12_ellipse_fix.py`)**: For occluded or heavily shadowed single eggs (e.g., IDs 117 and 135), precise mathematical ellipses were parameterized to guarantee smooth, biologically plausible contours.

### 1.2 Abandoned Approaches
* **`04_manual_fix.py`**: An early interactive OpenCV GUI window designed for manual point clicking was discarded because GUI loops proved brittle across platforms, lacked sub-pixel precision, and could not easily serialize revision history.
* **`08_fix_boxes2.py`**: A secondary box-only prompt pass was abandoned when touching eggs in tight clusters caused SAM's mask decoder to collapse multiple eggs into a single merged blob, necessitating explicit CVAT polygon tracing instead.

### 1.3 Dataset Deliverable
All 110 images have verified binary ground-truth masks saved in `outputs/sam_final/masks/<image stem>.png` (where pixel values $>0$ indicate egg foreground).

---

## 2. Task 2: Dataset Split & Classical Algorithm Development

### 2.1 Dataset Partitioning & Leakage Prevention
The dataset consists of 110 images captured across repeated visual environments (e.g., steel pan, woven cloth, plastic bag, keyboard, ash, plastic tub). 

> **Critical Leakage Risk**: A standard random split places identical backgrounds in both development and evaluation splits, leading to severe visual leakage and artificially inflated evaluation scores.

* **Leaky Split (`outputs/split_v1_leaky.json`)**: Initial background-histogram clustering allowed scene clusters to spill across splits. In particular:
  * The `bag` scene was split across sets: 6 images in dev, 1 image (ID 111) in eval.
  * The `tub` scene was split across sets: 3 images in dev (108, 127, 139) and 2 images in eval (130, 141).
* **Leakage-Safe Split (`outputs/split.json`)**: Grouped scenes manually and with KMeans background descriptors into strict scene clusters, then partitioned using `GroupShuffleSplit` (seed 348) into an **88-image dev set** and a **22-image eval set**.
  * Complete cluster independence: zero overlapping scenes between dev and eval.
  * All hyperparameter tuning was restricted strictly to the development split.

### 2.2 Classical Experiment Log (Trials T01–T29)
All experiments were evaluated strictly on the development split (88 images) and logged to `experiment_log.csv`. A trial was adopted as a milestone only if mean Dev IoU improved by $>0.005$ and Boundary F1 did not drop. 

> **Eval Disclosure:** Model exploration, candidate scoring weights, boundary refinement methods, and hyperparameter sweeps were tuned strictly and exclusively on the development split (88 images). Held-out evaluation metrics were computed and viewed only for the five intermediate adopted milestones (**T10**, **T13**, **T15**, **T19**, **T21**) and the final adopted method (**T29**); all other exploratory and discarded trials (**T01–T09**, **T11**, **T12**, **T14**, **T16–T18**, **T20**, and **T22–T28**) were evaluated strictly on the development split, with zero access to or tuning on the held-out evaluation set.

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
| **T08** | - | T07 + relaxed multi-egg union threshold | Multi-egg images (44, 144, 131) had low recall | Union threshold $0.9 \to 0.6$ | 0.582 / 0.593 | 0.682 | - | DISCARDED | Inadvertently added non-egg clutter |
| **T09** | - | Smart multi-egg union + Bright/Low-Sat cue | Shape-filtered multi-egg union with brightness cue | Union thr 0.7, fit $\ge 0.5$, sol $\ge 0.6$, $+0.2(V - 0.3S)$ | 0.660 / 0.729 | 0.751 | - | **KEPT** | +0.028 gain over T07 |
| **T10** | Milestone | T09 with tightened multi-egg threshold | Avoid marginal non-egg regions | Union thr 0.75, fit $\ge 0.55$, sol $\ge 0.65$ | 0.667 / 0.746 | 0.757 | 0.582 | **KEPT** | +0.007 gain over T09 |
| **T11** | - | Colour consistency pruning against core Lab median | Over-segmentation pruning | $K=1.5..4.0$ (sweep: 0.44-0.50); $K=1000$ control (0.667) | 0.667 / 0.746 | 0.757 | - | DISCARDED | Gradients on egg shell cause over-pruning |
| **T12** | - | Adaptive opening by distance-transform radius | Prune protrusions/bridges | $F=0.3$ (sweep: 0.615-0.666) | 0.666 / 0.744 | 0.754 | - | DISCARDED | Erodes natural egg poles |
| **T13** | Milestone | Marker-controlled watershed on Lab gradient + GrabCut | Bridge split lit/shaded egg halves | $fe=0.35, be=0.45$, GrabCut 3 iters | 0.681 / 0.757 | 0.765 | 0.601 | **KEPT** | +0.014 gain over T10 |
| **T14** | - | Random search of candidate scoring weights | Optimize feature weights via 5-fold CV | 9 features, 200 samples, 5-fold scene-grouped CV | 0.545 / 0.563 | - | - | DISCARDED | Proxy overfits to scene specifics |
| **T15** | Milestone | GrabCut bands proportional to equivalent egg radius $d$ | Adapt band widths to object size | $fe=0.45d, pf=0.10d, bgd=0.25d$, 3 iters | 0.707 / 0.861 | 0.781 | 0.628 | **KEPT** | +0.026 gain over T13 |
| **T16** | - | Inner-candidate rule on large regions | Replace $>30\%$ image region | Area $>0.3 \times$ image | 0.702 / 0.703 | - | - | DISCARDED | Never changed a selection |
| **T17** | - | Multi-egg union size-consistency rule | Extra regions must be 0.3-3x best region area | Area ratio 0.3-3x | 0.701 / 0.703 | - | - | DISCARDED | No gain over T15 (0.7031) |
| **T18** | - | Shape filter on final components | Filter post-GrabCut false positives | Ellipse fit $\ge 0.6-0.8$, solidity $\ge 0.85-0.95$ | 0.695 / 0.703 | - | - | DISCARDED | Smooth GrabCut false regions pass; partial eggs fail |
| **T19** | Milestone | Extra candidates from Felzenszwalb graph segmentation | Edge-respecting regions for low-contrast scenes | Scales 100/120/150, $\sigma=0.8$, min size 1% | 0.729 / 0.894 | 0.794 | 0.643 | **KEPT** | +0.022 gain (within noise); rescues bag scenes |
| **T20** | - | Re-rank 4 refined hypotheses by edge + ring contrast | Hypothesis re-ranking | Score + $w_e \cdot \text{edge} + w_c \cdot \text{contrast}$ | 0.742 / - | - | - | DISCARDED | Selection noise across scene-halves |
| **T21** | Milestone | Strict multi-egg union rule (score $\ge 0.85 \times \text{best}$) | Suppress swamping false extra regions | Union ratio threshold $0.85$ (was $0.75$) | 0.766 / 0.897 | 0.831 | 0.668 | **KEPT** | +0.037 gain over T19; bootstrap 95% CI [+0.005, +0.08] excludes 0 |
| **T22** | Step 2a | Adaptive multi-egg union threshold | Fix under-segmented clustered eggs (IDs 144, 126, 134) | $\text{ratio}=0.85$, relaxed to 0.72 if $\text{fit} \ge 0.75, \text{sol} \ge 0.85$ | 0.710 / 0.844 | 0.781 | 0.638 | DISCARDED | Background shapes pass shape test; over-segmentation |
| **T23** | Step 2b | Quadratic center distance penalty | Prevent off-center eggs being outscored (ID 4) | Score $- 0.6 \cdot cd + 0.6 \cdot cd - 0.8 \cdot cd^2$ | 0.749 / 0.894 | 0.816 | 0.654 | DISCARDED | Under-penalizes peripheral clutter on complex backgrounds |
| **T24** | Step 3a | Post-GrabCut boundary refinement | Refine ragged edges using image color GMMs | Erode 5 FG, dilate 11 PR_BGD, 2 iters | 0.768 / 0.897 | 0.833 | 0.671 | DISCARDED | BoundF1 $+0.0036$, but IoU gain $+0.0018 < +0.005$ |
| **T25** | Step 3b | Morphological opening, closing, and hole fill | Suppress pixel step-noise and fill specular holes | Open 3, close 5, fill, $\le 6$ components | 0.766 / 0.897 | 0.831 | 0.667 | DISCARDED | No gain over baseline |
| **T26** | Step 3c | Ellipse-fit shape prior | Enforce biological elliptical boundary regularity | $\text{fit\_thr} \in \{0.85, 0.90\}$ | 0.758 / 0.877 | 0.827 | 0.621 | DISCARDED | BoundF1 collapsed $0.668 \to 0.621$; sweep $\{0.85: 0.756/0.600, 0.90: 0.758/0.621\}$; asymmetric eggs don't fit 2D ellipses |
| **T27** | Step 3d | Gaussian contour smoothing | Smooth step-edges along mask perimeter | $k=7, \sigma=1.5$, thr $=0.5$ | 0.766 / 0.897 | 0.831 | 0.669 | DISCARDED | BoundF1 $+0.0009$, IoU $+0.0003 < +0.005$ |
| **T28** | Step 3a+3d | Combined Post-GrabCut + Gaussian contour smoothing | Snapping plus subpixel boundary smoothing | GrabCut 2 iters + Gaussian $k=7, \sigma=1.5$ | 0.767 / 0.896 | 0.832 | 0.672 | DISCARDED | BoundF1 $+0.0041$, IoU $+0.0016 < +0.005$ |
| **T29** | Step 4 | Candidate texture penalty + egg-size prior | Penalize fabric weaves (cloth scene) & irregular clutter | $w_{\text{tex}}=0.05 \times \frac{|\nabla^2 I|}{50}, w_{\text{size}}=0.10$ | **0.773** / **0.897** | **0.840** | **0.673** | **KEPT** | $+0.0073 \ge +0.005$ IoU gain, BoundF1 rose from 0.668 to 0.673; selected as final method |

---

## 3. Task 3: Evaluation, Metrics, and Failure Modes

### 3.1 Quantitative Results
The benchmark metrics across development and held-out evaluation sets are summarized below:

| Model / Trial | Split | Images | Mean IoU | Median IoU | Mean Dice | Precision | Recall | Boundary F1 (2px) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **T29 (Final Classical)** | **Development** | 88 | **0.773** | **0.896** | **0.840** | 0.851 | 0.866 | **0.670** |
| **T29 (Final Classical)** | **Evaluation** | 22 | **0.782** | **0.944** | **0.834** | 0.830 | 0.856 | **0.717** |
| **T21 (Milestone Baseline)** | **Development** | 88 | **0.766** | **0.896** | **0.831** | 0.847 | 0.853 | **0.668** |
| **T21 (Milestone Baseline)** | **Evaluation** | 22 | **0.764** | **0.945** | **0.820** | 0.813 | 0.846 | **0.701** |

*Note: Boundary F-score is computed at a 2-pixel tolerance at the $320\text{ px}$ working scale. In accordance with strict leakage-prevention protocols, T29 was tuned strictly on the development split (88 images). Evaluating T29 on the held-out evaluation set demonstrates strong generalizability: Mean IoU increased from 0.764 to **0.782** (+0.018 gain), Precision rose to 0.830, Recall to 0.856, and Boundary F1 reached **0.717**.*

### 3.2 Evaluation Visualizations
* **IoU Distribution Histogram (`outputs/results/iou_hist.png`)**: Confirms an overwhelming concentration of evaluation images at the high end, with median IoU reaching **0.945**.
* **Scene-Grouped IoU Comparison (`outputs/results/scene_iou.png`)**:
  * High-performing scenes: `keyboard` (0.99), `tub` (0.89), `leaves` (0.88), `hand` (0.82), `other` (0.80 dev / 0.80 eval), `pan` (0.74), `bag` (0.68 eval).
  * Challenging scenes: `ash` (0.46), `cloth` (0.32).
* **Worst Failures Tile (`outputs/results/worst_failures.jpg`)**: Compares original image, ground truth overlay, classical prediction, and false-positive/false-negative error maps for the worst eval failures.
* **Diagnostic Failure Overlays (`outputs/results/failures_dev/`)**: High-resolution 4-panel diagnostic overlays documenting the primary failure categories on the development split.

### 3.3 Failure Mode Analysis
Detailed auditing across development and evaluation partitions reveals four distinct failure categories:
1. **Wrong Object Selection on Textured Cloth**: In cloth scenes (`004`, `005`, `052`), high-contrast woven textile folds form closed loops that compete with egg geometry. Method T29 resolved this by adding an internal Laplacian texture penalty, lifting cloth scene IoU from 0.317 to 0.395.
2. **Camouflage in Dusty Ash**: In ash scenes (`027`, `036`), pale eggshell and grey ash exhibit nearly identical Lab color coordinates ($\Delta E \approx 0$), making boundary localization purely dependent on edge contrast.
3. **Clustered Touching Eggs**: In multi-egg clusters (`144`, `126`, `134`), secondary eggs sometimes score below the strict $0.85$ union ratio and are omitted. Attempts to relax this threshold adaptively (T22) introduced worse over-segmentation on single-egg images.
4. **Extreme Specular Reflections and Glare (`117`, `123`, `135`)**: On glossy eggs, metallic pans, or crumpled plastic bags, specular glare saturates pixel channels and breaks edge continuity.

### 3.4 Future Improvements
1. **Edge-Aware Gradient Verification**: Integrate Canny edge loops with bilateral filtering or active contours (Chan-Vese) to trace egg contours when color differences vanish.
2. **Pre-GrabCut Watershed Partitioning**: Apply distance-transform watershed directly to candidate masks before GrabCut initialization to decouple touching eggs into distinct Gaussian mixture models.
3. **Log-Chromaticity Specular Glare Removal**: Apply illumination-invariant log-chromaticity projection to neutralize specular highlights before K-means clustering.

---

## 4. Task 4: Video Tracking

### 4.1 Architecture & Pipeline
`scripts/15_track.py` implements an egg tracking pipeline over `data/video/eggs_clip.mov`:
* **Downsampling**: Frames are resized to $320\text{ px}$ on the longest side ($320 \times 198$) for fast execution.
* **Frame Skipping & Mask Caching**: Classical segmentation (T10) executes on every 2nd frame ($\sim 1,705$ frames); masks and tracks are cached and rendered on skipped frames to maintain full source temporal fidelity.
* **Watershed Splitting**: Connected components are analyzed with Euclidean distance transforms (`scipy.ndimage.maximum_filter`). Touching eggs are decoupled via marker-controlled watershed before association.
* **Hungarian Data Association**: Solved via `scipy.optimize.linear_sum_assignment` using a combined cost:
  $$\text{Cost} = (1 - \text{IoU}) + 0.3 \times \frac{\text{Centroid Distance}}{\text{Frame Diagonal}}$$
* **Gating & Track Confirmation**:
  * Gating threshold: matches with $\text{Cost} > 0.8$ are rejected.
  * Track confirmation: tentative tracks require **3 consecutive detections** before receiving a persistent ID.
  * Occlusion tolerance: tracks persist through **30 missed evaluations** before termination.
  * Minimum area filter: blobs $< 0.5\%$ of frame area ($< 316.8\text{ px}$) are ignored.

### 4.2 Tracking Results
* **Total Frames Processed**: 3,409 frames @ 31.79 fps.
* **Confirmed Unique Tracks**: 69 unique tracks across the duration.
* **Output Video**: Exported as an H.264 MP4 at `outputs/tracking_output.mp4` (3.23 MB) with smooth playback and persistent ID labels.
* **Video Link**: `Video link: <PASTE GOOGLE DRIVE LINK>`

### 4.3 Limitations
* Rapid egg rotations that disrupt aspect ratio can elevate Hungarian matching cost beyond 0.8.
* Complete visual occlusion exceeding 30 frames causes re-entering eggs to be assigned a new track ID.

---

## 5. AI-Use Disclosure
Artificial intelligence tools (Claude and the Google DeepMind Antigravity coding agent) assisted in authoring boilerplate code, formatting scripts, and debugging shell environments. All underlying algorithms, computer vision pipelines, evaluation metrics, and analytical conclusions were verified, understood, and explained by the author.

