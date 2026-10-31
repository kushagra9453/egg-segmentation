import json, os, shutil
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

shutil.copy("outputs/split.json", "outputs/split_v1_leaky.json")
s = json.load(open("outputs/split_v1_leaky.json"))
files = sorted(s["dev"] + s["eval"])
scenes = {
 "ash": [27,36,41,50,78], "pan": [113,114,128,129,147,135,124,145],
 "keyboard": [17,18,19,43], "cloth": [4,5,6,62,52,55],
 "bag": [104,106,109,111,112,123,125], "tub": [108,127,130,139,141],
 "leaves": [10,11,12,13],
}
idmap = {i: 100 + n for n, v in enumerate(scenes.values()) for i in v}
gid = lambda f: idmap.get(int(os.path.basename(f).split("_")[2]), s["cluster_of"][f])
groups = np.array([gid(f) for f in files])
X = np.zeros(len(files))
best = None
for seed in range(3000):
    tr, ev = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, groups=groups))
    sc = abs(len(ev) - 22) * 10 - len(set(groups[ev]))
    if best is None or sc < best[0]:
        best = (sc, seed, tr, ev)
_, seed, tr, ev = best
assert not set(groups[tr]) & set(groups[ev])
out = {"seed": seed, "method": "kmeans background clusters + manual scene groups, GroupShuffleSplit",
       "dev": [files[i] for i in tr], "eval": [files[i] for i in ev],
       "cluster_of": {f: int(g) for f, g in zip(files, groups)}}
json.dump(out, open("outputs/split.json", "w"), indent=1)
print("dev", len(tr), "eval", len(ev), "seed", seed)
evi = set(int(os.path.basename(files[i]).split("_")[2]) for i in ev)
for k, v in scenes.items():
    print(k, "eval:", [i for i in v if i in evi], "dev:", [i for i in v if i not in evi])