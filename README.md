# Egg Segmentation & Video Tracking (NOSH CV Task)

This repository contains an end-to-end computer vision pipeline for egg segmentation and video tracking:
1. **Task 1: Pseudo Ground-Truth Generation** using Meta's Segment Anything Model (SAM) with manual curation, bounding-box prompts, and CVAT polygon fixes.
2. **Task 2: Classical Image Processing Segmentation** evaluated iteratively across 10 trials on a leakage-free development split.
3. **Task 3: Evaluation & Benchmarking** against a held-out evaluation set with boundary F-score, error visualizations, and scene-stratified analysis.
4. **Task 4: Multi-Egg Video Tracking** using classical segmentation, distance-transform watershed splitting, and Hungarian matching with track confirmation and gating.

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
Executes trials T01–T10 on the development split (88 images), picks the best method (**T10**), evaluates on the held-out evaluation split (22 images), and generates evaluation metrics, boundary F-scores, worst-failure tiles, and scene-stratified IoU plots:
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
Segments every 2nd frame at 320 px using method T10, splits touching eggs via distance-transform watershed, tracks eggs with Hungarian association (`scipy.optimize.linear_sum_assignment`), and exports H.264 video:
```bash
python scripts/15_track.py
```
**Output generated**:
* `outputs/tracking_output.mp4` (H.264 encoded, 320x198, source fps, persistent IDs)

---

## 3. Results Summary

### Classical Segmentation Benchmark (T10)
| Split | Images | Mean IoU | Median IoU | Mean Dice | Precision | Recall | Boundary F1 (2px) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dev** | 88 | **0.667** | **0.746** | **0.757** | 0.776 | 0.788 | **0.557** |
| **Eval** | 22 | **0.287** | **0.001** | **0.339** | 0.322 | 0.409 | **0.307** |

### Video Tracking Performance
* **Total Frames Processed**: 3,409 frames @ 31.79 fps.
* **Tracking Strategy**: Gated Hungarian cost $(1 - \text{IoU}) + 0.3 \times \frac{\text{dist}}{\text{diag}}$ with 0.8 cost cutoff, 3-frame confirmation threshold, and 30-frame missed track retention.
* **Output Format**: H.264 MP4 (`outputs/tracking_output.mp4`, 3.23 MB).
