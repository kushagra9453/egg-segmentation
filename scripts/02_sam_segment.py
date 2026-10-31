"""Step 2 / Task 1: pseudo ground truth with SAM.  RUN ON COLAB (GPU) or a local GPU.

Pipeline
  1. SamAutomaticMaskGenerator proposes many masks per image.
  2. We keep masks that look like eggs (size, convexity, ellipse fit) -> handles 1-3 eggs.
  3. For images where this fails, add click prompts to outputs/sam_overrides.json
     ({"img.jpg": [[x,y],[x,y]]}  one click per egg) and re-run; SamPredictor is used for those.
Outputs (outputs/sam/): masks/<name>.png (0=bg, 1..n = egg instances), overlays/<name>.jpg,
  review_sheet_*.jpg (contact sheets to eyeball all 110), report.csv (flags suspicious images).
"""
import os, sys, glob, json, csv
import numpy as np, cv2, torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/egg_v1i_folder"
CKPT = sys.argv[2] if len(sys.argv) > 2 else "sam_vit_h_4b8939.pth"
MODEL = "vit_h" if "vit_h" in CKPT else ("vit_l" if "vit_l" in CKPT else "vit_b")
OUT = "outputs/sam"
for d in ("masks", "overlays"): os.makedirs(f"{OUT}/{d}", exist_ok=True)
overrides = json.load(open("outputs/sam_overrides.json")) if os.path.exists("outputs/sam_overrides.json") else {}

dev = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL](checkpoint=CKPT).to(dev)
auto = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=0.86,
                                 stability_score_thresh=0.90, min_mask_region_area=500)
pred = SamPredictor(sam)

def egg_score(m):
    """How egg-like is a binary mask? returns (score, area_frac). Eggs ~ convex ellipses."""
    area = m.sum() / m.size
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0, area
    c = max(cnts, key=cv2.contourArea)
    m = np.ascontiguousarray(m)
    if len(c) < 5: return 0, area
    hull = cv2.convexHull(c)
    solidity = cv2.contourArea(c) / (cv2.contourArea(hull) + 1e-6)
    el = cv2.fitEllipse(c); ell = np.zeros(m.shape, np.uint8)
    cv2.ellipse(ell, el, 1, -1)
    inter = (ell.astype(bool) & m).sum(); union = (ell.astype(bool) | m).sum()
    ell_iou = inter / (union + 1e-6)
    (w, h) = el[1]; aspect = min(w, h) / (max(w, h) + 1e-6)
    s = 0.5 * ell_iou + 0.3 * solidity + 0.2 * (1.0 if 0.55 < aspect < 1.0 else 0.0)
    return s, area

def border_touch(m, t=3):
    return m[:t].any() or m[-t:].any() or m[:, :t].any() or m[:, -t:].any()

def auto_eggs(img):
    cands = []
    for r in auto.generate(img):
        m = r["segmentation"]; s, a = egg_score(m)
        if 0.005 < a < 0.65 and s > 0.85 and not (border_touch(m) and a > 0.3):
            cands.append((s, m))
    cands.sort(key=lambda x: -x[0]); keep = []
    for s, m in cands:                                       # drop duplicates / nested masks
        if all((m & k).sum() / min(m.sum(), k.sum()) < 0.5 for k in keep): keep.append(m)
    return keep[:4]

def clicked_eggs(img, pts):
    pred.set_image(img); keep = []
    for x, y in pts:
        ms, sc, _ = pred.predict(point_coords=np.array([[x, y]]), point_labels=np.array([1]), multimask_output=True)
        best = max(range(3), key=lambda i: egg_score(ms[i])[0] + 0.1 * sc[i])
        keep.append(ms[best])
    return keep

files = sorted(glob.glob(os.path.join(DATA, "**", "*.jpg"), recursive=True))
rows, overlays = [], []
for f in files:
    name = os.path.basename(f); img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    masks = clicked_eggs(img, overrides[name]) if name in overrides else auto_eggs(img)
    lab = np.zeros(img.shape[:2], np.uint8)
    for i, m in enumerate(masks, 1): lab[m] = i
    cv2.imwrite(f"{OUT}/masks/{os.path.splitext(name)[0]}.png", lab)
    ov = img.copy(); ov[lab > 0] = (0.4 * ov[lab > 0] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    cv2.putText(ov, f"{len(masks)} egg(s)", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.imwrite(f"{OUT}/overlays/{name}", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)); overlays.append((name, ov))
    flag = "NO_EGG" if not masks else ("MANY" if len(masks) > 2 else "")
    rows.append([name, len(masks), flag, "override" if name in overrides else "auto"])
    print(name, len(masks), flag)

with open(f"{OUT}/report.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["image", "n_eggs", "flag", "source"]); w.writerows(rows)
for k in range(0, len(overlays), 20):                       # 20 per sheet (5x4) for quick visual review
    sheet = np.zeros((4 * 256, 5 * 256, 3), np.uint8)
    for j, (n, ov) in enumerate(overlays[k:k + 20]):
        t = cv2.resize(ov, (256, 256)); cv2.putText(t, n[:14], (4, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        sheet[(j // 5) * 256:(j // 5 + 1) * 256, (j % 5) * 256:(j % 5 + 1) * 256] = t
    cv2.imwrite(f"{OUT}/review_sheet_{k // 20}.jpg", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
print("done. Review outputs/sam/review_sheet_*.jpg, fix bad ones via outputs/sam_overrides.json and re-run.")
