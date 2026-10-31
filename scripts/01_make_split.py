"""Step 1: leakage-aware 4:1 dev/eval split.

Images share backgrounds (keyboard, pink bowl, steel pot, rug, ...). A random split
would put near-identical scenes in both dev and eval, inflating eval scores.
So we cluster images by BACKGROUND appearance and split whole clusters.
"""
import json, glob, os, sys
import numpy as np, cv2
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/egg_v1i_folder"
K = 12
files = sorted(glob.glob(os.path.join(DATA, "**", "*.jpg"), recursive=True))
print("images:", len(files))

def bg_feature(path):
    im = cv2.resize(cv2.imread(path), (128, 128))
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = np.ones((128, 128), np.uint8) * 255
    m[32:96, 32:96] = 0                      # drop the centre (egg usually there)
    h = cv2.calcHist([hsv], [0, 1, 2], m, [12, 4, 4], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, None, norm_type=cv2.NORM_L1).flatten()

X = np.stack([bg_feature(f) for f in files])
groups = KMeans(K, n_init=10, random_state=0).fit_predict(np.sqrt(X))  # sqrt ~ Hellinger

target = round(len(files) * 0.2)
best = None
for seed in range(500):                      # pick the best seed
    tr, ev = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, groups=groups))
    # prefer eval size near 20% AND an eval set spanning many different backgrounds
    score = abs(len(ev) - target) * 10 - len(set(groups[ev]))
    if best is None or score < best[0]:
        best = (score, seed, tr, ev)
_, seed, tr, ev = best
out = {
    "seed": seed, "k_clusters": K,
    "dev":  [os.path.relpath(files[i], DATA) for i in tr],
    "eval": [os.path.relpath(files[i], DATA) for i in ev],
    "cluster_of": {os.path.relpath(f, DATA): int(g) for f, g in zip(files, groups)},
}
os.makedirs("outputs", exist_ok=True)
json.dump(out, open("outputs/split.json", "w"), indent=1)
print(f"dev={len(tr)} eval={len(ev)} (ratio {len(tr)/len(ev):.2f}:1)")
print("eval clusters:", sorted(set(groups[ev])), "| dev clusters:", sorted(set(groups[tr])))
assert not set(groups[tr]) & set(groups[ev]), "leakage: cluster in both splits"
