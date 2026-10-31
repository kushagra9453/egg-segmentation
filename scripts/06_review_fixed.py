import cv2, json, glob, os
import numpy as np

ids = [10,11,12,13,31,42,44,55,5,113,114,116,117,118,121,124,125,128,129,132,133,134,135,140,143,144,145,147,152,163]
imgs = {}
for p in glob.glob("data/egg_v1i_folder/**/*.jpg", recursive=True):
    n = os.path.basename(p).split("_")[2]
    imgs[int(n)] = p
tiles = []
for i in ids:
    p = imgs.get(i)
    if p is None:
        continue
    im = cv2.imread(p)
    stem = os.path.splitext(os.path.basename(p))[0]
    mp = glob.glob(f"outputs/sam_final/masks/{stem}*")
    if not mp:
        continue
    m = cv2.imread(mp[0], 0)
    m = (m > 0).astype(np.uint8)
    im = cv2.resize(im, (256, 256))
    m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
    ov = im.copy()
    ov[m > 0] = (0.5 * ov[m > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.putText(ov, str(i), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    tiles.append(ov)
while len(tiles) % 6:
    tiles.append(np.zeros((256, 256, 3), np.uint8))
rows = [np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles), 6)]
cv2.imwrite("outputs/sam_final/review_fixed.jpg", np.vstack(rows))
print("tiles:", len(tiles), "saved review_fixed.jpg")