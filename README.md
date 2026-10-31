# Egg segmentation: SAM pseudo-GT vs classical image processing

## Setup
```bash
git clone <your-repo-url> && cd egg-segmentation
pip install -r requirements.txt
# put the dataset at data/egg_v1i_folder/  (unzip egg_v1i_folder.zip into data/)
```

## Run order
| Step | Command | Where |
|---|---|---|
| 1. Leakage-aware 4:1 split | `python scripts/01_make_split.py` | local |
| 2. Task 1 – SAM masks (pseudo GT) | `python scripts/02_sam_segment.py data/egg_v1i_folder sam_vit_h_4b8939.pth` | Colab GPU |
| 3. Task 2 – classical algorithm + experiment log | *(to be added)* | local |
| 4. Task 3 – evaluation | *(to be added)* | local |
| 5. Task 4 – video tracking | *(to be added)* | local |

SAM install (Colab): `pip install git+https://github.com/facebookresearch/segment-anything.git`
Checkpoint: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

## Split / leakage handling
Images are clustered by background colour statistics (egg-centre masked out); whole clusters go to
either dev or eval (`GroupShuffleSplit`), so no scene appears in both. Result: `outputs/split.json`.

## AI tools disclosure
*(fill in honestly)*
