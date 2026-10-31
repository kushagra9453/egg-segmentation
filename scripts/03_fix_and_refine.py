"""Task 1 clean-up: turn the raw SAM masks into the final pseudo ground truth.

Raw SAM output (outputs/sam/masks) often has extra small blobs (fingers, glare, patterns).
Rule: keep the LARGEST egg-like instance; keep another one only if it is big enough
(>= 35% of the largest), looks similar (colour) and does not overlap it -> real 2-egg photos survive.
Images with no egg found get manual click prompts (outputs/sam_overrides.json) run through SAM.
Output: outputs/sam_final/{masks,overlays,review_sheet_*.jpg,report.csv}
"""
import os, sys, glob, json, csv
import numpy as np, cv2

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/egg_v1i_folder"
CKPT = sys.argv[2] if len(sys.argv) > 2 else "sam_vit_b_01ec64.pth"
SKIP_SAM = "--skip-sam" in sys.argv
RAW, OUT = "outputs/sam/masks", "outputs/sam_final"
MIN_REL_AREA, MAX_COLOR_DIST, MAX_OVERLAP, MIN_FRAC = 0.35, 25.0, 0.2, 0.004
for d in ("masks", "overlays"): os.makedirs(f"{OUT}/{d}", exist_ok=True)
overrides = json.load(open("outputs/sam_overrides.json")) if os.path.exists("outputs/sam_overrides.json") else {}

def solid_ellipse_score(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5: return 0
    ell = np.zeros(m.shape, np.uint8); cv2.ellipse(ell, cv2.fitEllipse(c), 1, -1)
    e = ell.astype(bool); return (e & m).sum() / ((e | m).sum() + 1e-6)

def sam_clicks(files_by_name):
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    model = "vit_h" if "vit_h" in CKPT else ("vit_l" if "vit_l" in CKPT else "vit_b")
    sam = sam_model_registry[model](checkpoint=CKPT).to("cuda" if torch.cuda.is_available() else "cpu")
    pred = SamPredictor(sam); res = {}
    for name, pts in overrides.items():
        img = cv2.cvtColor(cv2.imread(files_by_name[name]), cv2.COLOR_BGR2RGB); pred.set_image(img)
        lab = np.zeros(img.shape[:2], np.uint8)
        for i, (x, y) in enumerate(pts, 1):
            ms, sc, _ = pred.predict(point_coords=np.array([[x, y]]), point_labels=np.array([1]), multimask_output=True)
            ok = [k for k in range(3) if ms[k].mean() < 0.7]           # reject "whole background" masks
            best = max(ok or [0], key=lambda k: solid_ellipse_score(ms[k]) + 0.1 * sc[k])
            lab[ms[best]] = i
        res[name] = lab
    return res

files = {os.path.basename(f): f for f in glob.glob(os.path.join(DATA, "**", "*.jpg"), recursive=True)}
fixed = sam_clicks(files) if (overrides and not SKIP_SAM) else {}
rows, overlays = [], []
for name, path in sorted(files.items()):
    stem = os.path.splitext(name)[0]
    img = cv2.imread(path); h, w = img.shape[:2]
    note = ""
    if name in fixed:
        lab, note = fixed[name], "manual_click"
    else:
        raw = cv2.imread(f"{RAW}/{stem}.png", cv2.IMREAD_UNCHANGED)
        ids = [i for i in np.unique(raw) if i > 0]
        if not ids:
            lab, note = np.zeros((h, w), np.uint8), "NO_EGG"
        else:
            lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            inst = [(i, raw == i) for i in ids]
            inst = [(i, m) for i, m in inst if m.mean() >= MIN_FRAC] or inst
            inst.sort(key=lambda t: -t[1].sum())
            big = inst[0][1]; keep = [big]; big_col = lab_img[big].mean(0)
            for _, m in inst[1:]:
                rel = m.sum() / big.sum()
                dist = np.linalg.norm(lab_img[m].mean(0) - big_col)
                overlap = (m & big).sum() / m.sum()
                if rel >= MIN_REL_AREA and dist <= MAX_COLOR_DIST and overlap <= MAX_OVERLAP and all((m & k).sum() / m.sum() <= MAX_OVERLAP for k in keep):
                    keep.append(m)
            lab = np.zeros((h, w), np.uint8)
            for k, m in enumerate(keep, 1): lab[m] = k
            if len(ids) != len(keep): note = f"filtered {len(ids)}->{len(keep)}"
            if big.mean() > 0.5: note += " BIG_MASK_CHECK"
    cv2.imwrite(f"{OUT}/masks/{stem}.png", lab)
    ov = img.copy(); sel = lab > 0
    ov[sel] = (0.4 * ov[sel] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.putText(ov, f"{int(lab.max())} egg(s)", (8, 28), 0, 0.9, (0, 255, 255), 2)
    cv2.imwrite(f"{OUT}/overlays/{name}", ov); overlays.append((name, ov))
    rows.append([name, int(lab.max()), note])
with open(f"{OUT}/report.csv", "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["image", "n_eggs", "note"]); wr.writerows(rows)
for k in range(0, len(overlays), 20):
    sheet = np.zeros((4 * 256, 5 * 256, 3), np.uint8)
    for j, (n, ov) in enumerate(overlays[k:k + 20]):
        t = cv2.resize(ov, (256, 256)); cv2.putText(t, n[:16], (4, 250), 0, 0.45, (255, 255, 255), 1)
        sheet[(j // 5) * 256:(j // 5 + 1) * 256, (j % 5) * 256:(j % 5 + 1) * 256] = t
    cv2.imwrite(f"{OUT}/review_sheet_{k // 20}.jpg", sheet)
from collections import Counter
print("egg-count histogram:", dict(sorted(Counter(r[1] for r in rows).items())))
print("images with notes:", sum(1 for r in rows if r[2]))
print("done -> outputs/sam_final/")
