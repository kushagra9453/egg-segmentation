import cv2, glob, os, shutil, sys
import numpy as np
import xml.etree.ElementTree as ET

cands = [p for p in glob.glob("cvat_export*") + glob.glob("cvat_export/**/*", recursive=True) if os.path.isfile(p)]
cands = [p for p in cands if p.endswith(".xml") or os.path.basename(p) == "cvat_export"]
print("xml candidates:", cands)
xml = cands[0]
root = ET.parse(xml).getroot()
os.makedirs("outputs/sam_final/masks_cvat_backup", exist_ok=True)
done = 0
for img in root.iter("image"):
    name = os.path.basename(img.get("name"))
    stem = os.path.splitext(name)[0]
    target = glob.glob(f"outputs/sam_final/masks/{stem}*")
    if not target:
        print("NO MATCH", name); continue
    t = target[0]
    old = cv2.imread(t, 0)
    H, W = old.shape
    sx, sy = W / float(img.get("width")), H / float(img.get("height"))
    mask = np.zeros((H, W), np.uint8)
    n = 0
    for el in img:
        if el.tag == "polygon":
            pts = np.array([[float(a) * sx, float(b) * sy] for a, b in (q.split(",") for q in el.get("points").split(";"))], np.int32)
            cv2.fillPoly(mask, [pts], 255); n += 1
        elif el.tag == "box":
            x1, y1, x2, y2 = [float(el.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
            cv2.rectangle(mask, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), 255, -1); n += 1
        elif el.tag == "ellipse":
            cv2.ellipse(mask, (int(float(el.get("cx")) * sx), int(float(el.get("cy")) * sy)), (int(float(el.get("rx")) * sx), int(float(el.get("ry")) * sy)), float(el.get("rotation", 0)), 0, 360, 255, -1); n += 1
        else:
            print("  unsupported tag", el.tag, "in", name[:25])
    if n == 0:
        print("no shapes for", name[:25], "- skipped"); continue
    shutil.copy(t, os.path.join("outputs/sam_final/masks_cvat_backup", os.path.basename(t)))
    cv2.imwrite(t, mask)
    done += 1
    print("replaced", name[:28], "eggs", n, "pixels", int((mask > 0).sum()))
print("total replaced:", done)