import cv2, json, glob, os
import numpy as np

ids = [117,135]
boxes = json.load(open("boxes.json"))
imgs = {}
for p in glob.glob("data/egg_v1i_folder/**/*.jpg", recursive=True):
    imgs[int(os.path.basename(p).split("_")[2])] = p
tiles = []
for i in ids:
    p = imgs[i]
    im = cv2.resize(cv2.imread(p), (400, 400))
    stem = os.path.splitext(os.path.basename(p))[0]
    m = cv2.imread(glob.glob(f"outputs/sam_final/masks/{stem}*")[0], 0)
    m = cv2.resize((m > 0).astype(np.uint8), (400, 400), interpolation=cv2.INTER_NEAREST)
    im[m > 0] = (0.6 * im[m > 0] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    for b in boxes.get(str(i), []):
        cv2.rectangle(im, (int(b[0]*400), int(b[1]*400)), (int(b[2]*400), int(b[3]*400)), (255, 0, 0), 2)
    cv2.putText(im, str(i), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    tiles.append(im)
while len(tiles) % 4:
    tiles.append(np.zeros((400, 400, 3), np.uint8))
rows = [np.hstack(tiles[k:k+4]) for k in range(0, len(tiles), 4)]
sheet = np.vstack(rows)
cv2.imwrite("outputs/sam_final/big_a.jpg", sheet[:800])
cv2.imwrite("outputs/sam_final/big_b.jpg", sheet[800:])
print("ok")