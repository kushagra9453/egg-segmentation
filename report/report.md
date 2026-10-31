# Computer Vision Intern Task: Egg Segmentation and Video Tracking
**Author:** Krishna  
**Project Repository:** [https://github.com/kushagra9453/egg-segmentation](https://github.com/kushagra9453/egg-segmentation)

---

## Executive Summary
This report presents an end-to-end computer vision solution for segmenting and tracking eggs across diverse, challenging real-world scenes. The project is organized into four tasks:
1. **Task 1: Ground-Truth Generation** utilizing Meta's Segment Anything Model (SAM) enhanced with manual bounding boxes, CVAT polygon corrections, and parametric ellipse fitting.
2. **Task 2: Classical Segmentation Optimization** developed iteratively over 10 disciplined trials on a leakage-free development split (88 images).
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

### 2.2 Classical Experiment Log (Trials T01–T10)
All experiments were logged to `experiment_log.csv`. A trial was adopted only if mean Dev IoU improved by $>0.005$:

| Trial | Modification | Hypothesis / Motivation | Parameters | Dev IoU | Dev Dice | Decision | Rationale |
|---|---|---|---|:---:|:---:|:---:|---|
| **T01** | Grayscale blur + Otsu | Simplest baseline | Blur 5x5, Otsu, largest component | 0.365 | 0.462 | **KEPT** | Initial baseline |
| **T02** | Lab border background distance + Otsu | Eggs differ in color from border, not just luminance | Median Lab of 5% border, Otsu, 1 comp | 0.355 | 0.428 | DISCARDED | No gain over baseline |
| **T03** | T02 + morphology, hole fill, multi-egg | Suppress noise, fill glare holes, allow up to 4 eggs | Open 5, close 9, fill, $\ge 25\%$ largest | 0.370 | 0.467 | DISCARDED | Below +0.005 threshold |
| **T04** | GrabCut refinement on T03 | Color model GMMs refine ragged boundaries | Erode 15 FG, dilate 41 BG, 4 iters | 0.396 | 0.478 | **KEPT** | +0.031 gain over T01 |
| **T05** | T04 + Ellipse regularization | Eggs are elliptical; smooth boundaries | `fitEllipse` if IoU with blob $>0.85$ | 0.392 | 0.477 | DISCARDED | Degraded irregular silhouettes |
| **T06** | Lab distance + Gaussian center prior | Eggs are centered; center prior suppresses border clutter | $\sigma = 0.45 \times \text{size}$, open 7, close 15 | 0.474 | 0.588 | **KEPT** | +0.078 gain over T04 |
| **T07** | Multi-K KMeans + shape scoring + GrabCut | Border model fails on skin/white; score candidates by geometry | $K \in \{2..5\}$, fit + 0.5 sol - 0.6 dist - 0.25 touch | 0.632 | 0.720 | **KEPT** | +0.158 gain over T06 |
| **T08** | T07 + relaxed multi-egg union threshold | Multi-egg images (44, 144, 131) had low recall | Union threshold $0.9 \to 0.6$ | 0.582 | 0.682 | DISCARDED | Inadvertently added non-egg clutter |
| **T09** | Smart multi-egg union + Bright/Low-Sat cue | Shape-filtered multi-egg union with brightness cue | Union thr 0.7, fit $\ge 0.5$, sol $\ge 0.6$, $+0.2(V - 0.3S)$ | 0.660 | 0.751 | **KEPT** | +0.028 gain over T07 |
| **T10** | T09 with tightened multi-egg threshold | Avoid marginal non-egg regions | Union thr 0.75, fit $\ge 0.55$, sol $\ge 0.65$ | **0.667** | **0.757** | **KEPT** | +0.007 gain; selected as final method |

---

## 3. Task 3: Evaluation, Metrics, and Failure Modes

### 3.1 Quantitative Results
Evaluating the selected **T10** method across the dataset yields:

| Split | Images | Mean IoU | Median IoU | Mean Dice | Precision | Recall | Boundary F1 (2px) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Development** | 88 | **0.667** | **0.746** | **0.757** | 0.776 | 0.788 | **0.557** |
| **Evaluation** | 22 | **0.287** | **0.001** | **0.339** | 0.322 | 0.409 | **0.307** |

*Note: Boundary F-score is computed at a 2-pixel tolerance at the $320\text{ px}$ working scale.*

### 3.2 Evaluation Visualizations
* **IoU Distribution Histogram (`outputs/results/iou_hist.png`)**: Illustrates a bimodal performance distribution on eval — high accuracy on contrasting scenes alongside a zero-IoU cluster corresponding to the white bag failure mode.
* **Scene-Grouped IoU Comparison (`outputs/results/scene_iou.png`)**:
  * High-performing scenes: `keyboard` (0.94), `leaves` (0.89), `hand` (0.76), `tub` (0.72), `pan` (0.72).
  * Low-performing scenes: `ash` (0.56), `cloth` (0.41), `bag` (0.04).
* **Worst Failures Tile (`outputs/results/worst_failures.jpg`)**: Compares original image, ground truth overlay, classical prediction, and false-positive/false-negative error maps for the 6 worst eval failures.

### 3.3 Failure Mode Analysis
1. **Low-Contrast White-on-White (`bag` scene, Mean IoU: 0.042)**: In eval images **104, 112, 123, 109**, pale eggs lie on crumpled white plastic. Because both egg and background exhibit near-identical high luminance and low saturation, color distance drops to zero. GrabCut segments the high-contrast plastic bag folds instead of the egg boundary.
2. **False Specular Curvatures (`32, 155`, IoU: 0.000)**: Glossy non-egg regions and bright specular table spots exhibit high circular compactness and center proximity, leading the candidate ranker to prefer them over diffuse, off-center eggs.
3. **Clustered Multi-Egg Under-Segmentation (`cloth` scene / Dev multi-egg)**: When eggs touch, connected components treat them as a single mass whose eccentricity and solidity fail ellipse thresholds, suppressing secondary eggs.

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

