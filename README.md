# Egg Segmentation & Video Tracking (NOSH CV Task)

This repository contains an end-to-end computer vision pipeline for egg segmentation and video tracking:
1. **Task 1: Pseudo Ground-Truth Generation** using Meta's Segment Anything Model (SAM) with manual curation, bounding-box prompts, and CVAT polygon fixes.
2. **Task 2: Classical Image Processing Segmentation** evaluated iteratively across 29 disciplined trials on a leakage-free development split (culminating in Method T29, Dev Mean IoU 0.773, Boundary F1 0.670).
3. **Task 3: Evaluation & Benchmarking** against a held-out evaluation set with boundary F-score, error visualizations, and scene-stratified analysis.
4. **Task 4: Multi-Egg Video Tracking** using classical segmentation, distance-transform watershed splitting, and Hungarian matching with track confirmation, ghost-box prevention, and gating.

---

## 1. Setup & Environment

Clone the repository and initialize the Python virtual environment:

```bash
git clone https://github.com/kushagra9453/egg-segmentation.git
cd egg-segmentation

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data and Checkpoint Setup
* **Dataset**: Place the unzipped dataset in `data/egg_v1i_folder/`. Images follow the Roboflow structure containing 110 total images (`train/`, `valid/`, `test/`).
* **SAM Checkpoint (Task 1 only)**: Not tracked in Git due to file size. Download the ViT-B checkpoint to project root:
  ```bash
  curl -L -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  ```
  *(Or ViT-H: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`)*
* **Video (Task 4 only)**: The YouTube screen-recording clip is placed at `data/video/eggs_clip.mov`.
* **Tracked Video Deliverable**: Available on Google Drive at [https://drive.google.com/file/d/1bmdephNfQXLSc2-urg9mRzZL_sNBW8Ap/view?usp=drive_link](https://drive.google.com/file/d/1bmdephNfQXLSc2-urg9mRzZL_sNBW8Ap/view?usp=drive_link).

---

## 2. Reproduction Commands

### Task 1: Pseudo Ground Truth Masks
All 110 final binary ground truth masks are committed under `outputs/sam_final/masks/`. To inspect or re-run the pipeline:
```bash
# 1. Initial SAM segmentation
python scripts/02_sam_segment.py data/egg_v1i_folder sam_vit_b_01ec64.pth

# 2. Heuristic filtering & refinement
python scripts/03_fix_and_refine.py data/egg_v1i_folder sam_vit_b_01ec64.pth

# 3. Manual bounding box overrides
python scripts/05_boxes_fix.py data/egg_v1i_folder sam_vit_b_01ec64.pth

# 4. Import CVAT polygons & parametric ellipse fixes
python scripts/10_import_cvat.py
python scripts/12_ellipse_fix.py
```

### Dataset Split (Leakage-Safe Clustering)
To recreate the 4:1 dev/eval leakage-free split (`outputs/split.json`):
```bash
python scripts/13_split_v2.py
```

### Tasks 2 & 3: Classical Segmentation & Evaluation
Executes trials T01–T29 on the development split (88 images), picks the best method (**T29**), evaluates on the held-out evaluation split (22 images), and generates evaluation metrics, boundary F-scores, worst-failure tiles, and scene-stratified IoU plots:
```bash
python scripts/14_classical.py
```
**Outputs generated**:
* `outputs/results/dev_scores.csv`
* `outputs/results/eval_scores.csv`
* `outputs/results/worst_failures.jpg`
* `outputs/results/iou_hist.png`
* `outputs/results/scene_iou.png`
* `outputs/results/analysis.md`
* `experiment_log.csv`

### Task 4: Video Tracking
Executes the refined classical tracking pipeline with automatic viewport cropping, temporal scene gating, T29 segmentation, Hungarian data association, local search on missed detections, and ghost box suppression:
```bash
python scripts/15_track.py
```
**Outputs generated**:
* `outputs/tracking_final.mp4` (H.264 encoded, 960x528, 31.79 fps, yuv420p, QuickTime/browser compatible)
* `outputs/tracking_check.jpg` (2x4 verification montage at 2s, 5s, 10s, 15s, 20s, 25s, 28s, 31s)

### Report Generation
To regenerate the PDF report (`report/report.pdf`) with all figures embedded:
```bash
python scripts/build_report_pdf.py
```

---

## 3. Results Summary

### Classical Segmentation Benchmark
| Model / Trial | Split | Images | Mean IoU | Median IoU | Mean Dice | Precision | Recall | Boundary F1 (2px) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **T29 (Final Classical)** | **Dev** | 88 | **0.773** | **0.896** | **0.840** | 0.851 | 0.866 | **0.670** |
| **T29 (Final Classical)** | **Eval** | 22 | **0.782** | **0.944** | **0.834** | 0.830 | 0.856 | **0.717** |
| **T21 (Milestone Baseline)** | **Dev** | 88 | **0.766** | **0.896** | **0.831** | 0.847 | 0.853 | **0.668** |
| **T21 (Milestone Baseline)** | **Eval** | 22 | **0.764** | **0.945** | **0.820** | 0.813 | 0.846 | **0.701** |

*Eval Disclosure: Model exploration, candidate scoring weights, boundary refinement methods, and hyperparameter sweeps were tuned strictly on the development split (88 images). Held-out evaluation metrics were computed only for the adopted milestones (T10, T13, T15, T19, T21) and the final evaluated model (T29); all other exploratory trials were restricted strictly to the development split.*

*Note: T29 incorporates candidate internal texture penalty (Laplacian mean / 50) and egg-size area priors ($0.03 \le \text{area} \le 0.35$), providing a $+0.0073$ mean IoU gain over T21 on Dev while increasing Eval Mean IoU to 0.782 (+0.018 gain) and Boundary F1 to 0.717.*

### Video Tracking Performance
* **Total Frames Processed**: 1,042 frames @ 31.79 fps ($32.78\text{ s}$ duration).
* **Viewport Isolation**: Automatically cropped to `[70:1115, 29:1931]` ($1902 \times 1045\text{ px}$).
* **Tracking Strategy**: Gated Hungarian cost $(0.60 \cdot \text{dist} + 0.30 \cdot (1 - \text{IoU}) + 0.10 \cdot \Delta \text{area})$ with constant-velocity prediction, 1.5 s internal memory, max 0.5 s drawing coasting cap, and strict 2-ID policy.
* **Self-Check Results**:
  1. No boxes on text slides ($t \ge 24.5\text{ s}$, 0 boxes); both eggs tracked at 5s: **PASS**.
  2. No box drawn $>0.5\text{ s}$ without matching detection: **PASS**.
  3. Only IDs "Egg 1" and "Egg 2" appear across video: **PASS**.
  4. Egg 2 preserves ID 2 before and after finger occlusion: **PASS**.
