import cv2, glob, os
import numpy as np

skip = {10,134,133,140,144,118,125}
imgs = {}
for p in glob.glob("data/egg_v1i_folder/**/*.jpg", recursive=True):
    imgs[int(os.path.basename(p).split("_")[2])] = p
ids = sorted(i for i in imgs if i not in skip)
tiles = []
for i in ids:
    p = imgs[i]
    im = cv2.resize(cv2.imread(p), (300, 300))
    stem = os.path.splitext(os.path.basename(p))[0]
    mp = glob.glob(f"outputs/sam_final/masks/{stem}*")
    if mp:
        m = cv2.imread(mp[0], 0)
        m = cv2.resize((m > 0).astype(np.uint8), (300, 300), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnts, -1, (0, 255, 0), 3)
    cv2.putText(im, str(i), (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    tiles.append(im)
per = 20
for s in range(0, len(tiles), per):
    chunk = tiles[s:s+per]
    while len(chunk) % 5:
        chunk.append(np.zeros((300, 300, 3), np.uint8))
    rows = [np.hstack(chunk[k:k+5]) for k in range(0, len(chunk), 5)]
    cv2.imwrite(f"outputs/sam_final/check_{s//per}.jpg", np.vstack(rows))
print("sheets:", (len(tiles) + per - 1) // per)
