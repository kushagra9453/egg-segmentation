import sys, os, glob, json, shutil, csv, cv2, numpy as np, torch
from segment_anything import sam_model_registry, SamPredictor

DATA, CKPT = sys.argv[1], sys.argv[2]
OUT = "outputs/sam_final"
BOXES = json.load(open("boxes.json"))   # {"10": [[x1,y1,x2,y2], ...]}, values 0-1 fractions; [] = no egg

if not os.path.exists(f"{OUT}/masks_before_manual"):
    shutil.copytree(f"{OUT}/masks", f"{OUT}/masks_before_manual")

dev = "mps" if torch.backends.mps.is_available() else "cpu"
pred = SamPredictor(sam_model_registry["vit_b"](checkpoint=CKPT).to(dev))
imgs = {os.path.basename(p): p for p in glob.glob(f"{DATA}/**/*.jpg", recursive=True)}

def mask_path(name):
    stem = os.path.splitext(name)[0]
    f = sorted(glob.glob(f"{OUT}/masks/{stem}*"))
    return f[0] if f else f"{OUT}/masks/{stem}.png"

log = []
for i, boxes in BOXES.items():
    name = next((n for n in imgs if n.startswith(f"not_damaged_{i}_jpg.")), None)
    if name is None:
        print("not found:", i); continue
    bgr = cv2.imread(imgs[name]); H, W = bgr.shape[:2]
    mp = mask_path(name); old = cv2.imread(mp, 0) if os.path.exists(mp) else None
    val255 = old is None or old.max() == 255
    lab = np.zeros((H, W), np.uint8)
    if boxes:
        pred.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        for k, (x1, y1, x2, y2) in enumerate(boxes):
            b = np.array([x1 * W, y1 * H, x2 * W, y2 * H])
            m, _, _ = pred.predict(box=b, multimask_output=False)
            lab[m[0]] = k + 1
    cv2.imwrite(mp, ((lab > 0) * 255).astype(np.uint8) if val255 else lab)
    ov = bgr.copy(); s = lab > 0
    ov[s] = (0.5 * ov[s] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.imwrite(f"{OUT}/overlays/{name}", ov)
    log.append((name, f"box_fix x{len(boxes)}")); print("fixed", i, len(boxes))
csv.writer(open(f"{OUT}/manual_fixes.csv", "w", newline="")).writerows([("image", "note")] + log)
print("done")