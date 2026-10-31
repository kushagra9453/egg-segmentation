import json, csv, os, cv2, numpy as np

ROOT, MASKS, LOG, S = "data/egg_v1i_folder", "outputs/sam_final/masks", "experiment_log.csv", 320
OUT = "outputs/results"; os.makedirs(OUT, exist_ok=True)

def load(rel):
    img = cv2.imread(os.path.join(ROOT, rel))
    gt = cv2.imread(f"{MASKS}/{os.path.splitext(os.path.basename(rel))[0]}.png", 0)
    if img is None or gt is None:
        return None
    f = S / max(img.shape[:2]); sz = (int(img.shape[1] * f), int(img.shape[0] * f))
    return cv2.resize(img, sz, interpolation=cv2.INTER_AREA), cv2.resize(gt, sz, interpolation=cv2.INTER_NEAREST) > 0

def keep_eggs(m, max_n=4):
    m = m.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(m.shape, bool)
    areas = st[1:, cv2.CC_STAT_AREA]; out = np.zeros(m.shape, bool)
    for k in np.argsort(-areas)[:max_n]:
        if areas[k] >= 0.25 * areas.max() and areas[k] >= 0.01 * m.size:
            out |= lab == k + 1
    return out

def fill(m):
    m = m.astype(np.uint8)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    o = np.zeros_like(m); cv2.drawContours(o, cnt, -1, 1, -1)
    return o > 0

def bgdist(img):
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(4, int(0.05 * S))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    d = np.linalg.norm(lab - np.median(border, 0), axis=2)
    return cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def t01(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return keep_eggs(th > 0, 1)

def t02(img):
    _, th = cv2.threshold(bgdist(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return keep_eggs(th > 0, 1)

def t03(img):
    _, th = cv2.threshold(bgdist(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    el = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx((th > 0).astype(np.uint8), cv2.MORPH_OPEN, el(5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, el(9))
    return keep_eggs(fill(m), 4)

def t04(img):
    m = t03(img)
    if not m.any():
        return m
    mu = m.astype(np.uint8); el = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    gc = np.full(m.shape, cv2.GC_PR_BGD, np.uint8)
    gc[cv2.dilate(mu, el(15)) > 0] = cv2.GC_PR_FGD
    gc[cv2.dilate(mu, el(41)) == 0] = cv2.GC_BGD
    gc[cv2.erode(mu, el(15)) > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(img, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m
    return keep_eggs(fill((gc == 1) | (gc == 3)), 4)

def t05(img):
    m = t04(img); out = np.zeros(m.shape, np.uint8)
    n, lab = cv2.connectedComponents(m.astype(np.uint8))
    for k in range(1, n):
        c = (lab == k).astype(np.uint8)
        cnt, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnt, key=cv2.contourArea)
        if len(cnt) >= 5:
            e = np.zeros_like(c); cv2.ellipse(e, cv2.fitEllipse(cnt), 1, -1)
            out |= e if (e & c).sum() / max(1, (e | c).sum()) > 0.85 else c
        else:
            out |= c
    return out > 0

def t06(img):
    d = bgdist(img).astype(np.float32)
    yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    c = np.exp(-(((xx - d.shape[1] / 2) / (0.45 * d.shape[1])) ** 2 + ((yy - d.shape[0] / 2) / (0.45 * d.shape[0])) ** 2))
    s = cv2.normalize(d * c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    el = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx((th > 0).astype(np.uint8), cv2.MORPH_OPEN, el(7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, el(15))
    return keep_eggs(fill(m), 3)

def t06(img):
    d = bgdist(img).astype(np.float32)
    yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    c = np.exp(-(((xx - d.shape[1] / 2) / (0.45 * d.shape[1])) ** 2 + ((yy - d.shape[0] / 2) / (0.45 * d.shape[0])) ** 2))
    s = cv2.normalize(d * c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    el = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx((th > 0).astype(np.uint8), cv2.MORPH_OPEN, el(7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, el(15))
    return keep_eggs(fill(m), 3)

def t07(img):
    h, w = img.shape[:2]
    el = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB)
    cands = []
    _, th = cv2.threshold(bgdist(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cands.append(th > 0)
    data = lab.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(0)
    for K in (2, 3, 4, 5):
        _, lb, _ = cv2.kmeans(data, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 1, cv2.KMEANS_PP_CENTERS)
        lb = lb.reshape(h, w)
        cands += [lb == k for k in range(K)]
    comps = []
    for m in cands:
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, el(5))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, el(9))
        n, cl, st, _ = cv2.connectedComponentsWithStats(m)
        for k in range(1, n):
            x, y, bw, bh, a = st[k]
            if a / (h * w) < 0.02 or a / (h * w) > 0.7:
                continue
            c = fill(cl == k).astype(np.uint8)
            cnt, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cnt, key=cv2.contourArea)
            if len(cnt) < 5:
                continue
            e = np.zeros_like(c); cv2.ellipse(e, cv2.fitEllipse(cnt), 1, -1)
            fit = (e & c).sum() / max(1, (e | c).sum())
            sol = cv2.contourArea(cnt) / max(cv2.contourArea(cv2.convexHull(cnt)), 1)
            ys, xs = np.nonzero(c)
            cd = np.hypot(xs.mean() / w - 0.5, ys.mean() / h - 0.5)
            touch = int(x <= 1) + int(y <= 1) + int(x + bw >= w - 1) + int(y + bh >= h - 1)
            comps.append((fit + 0.5 * sol - 0.6 * cd - 0.25 * touch, c))
    if not comps:
        return np.zeros((h, w), bool)
    comps.sort(key=lambda t: -t[0])
    union = comps[0][1].copy()
    for sc, c in comps[1:]:
        if sc >= 0.9 * comps[0][0] and (c & union).sum() / max(1, c.sum()) < 0.1 and (c & (union > 0)).sum() == 0:
            union |= c
    mu = union.astype(np.uint8)
    gc = np.full(mu.shape, cv2.GC_PR_BGD, np.uint8)
    gc[cv2.dilate(mu, el(15)) > 0] = cv2.GC_PR_FGD
    gc[cv2.dilate(mu, el(41)) == 0] = cv2.GC_BGD
    gc[cv2.erode(mu, el(15)) > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(img, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
        r = keep_eggs(fill((gc == 1) | (gc == 3)), 4)
        return r if r.sum() > 0.5 * mu.sum() else mu > 0
    except cv2.error:
        return mu > 0

def t08(img):
    h, w = img.shape[:2]
    el = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB)
    cands = []
    _, th = cv2.threshold(bgdist(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cands.append(th > 0)
    data = lab.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(0)
    for K in (2, 3, 4, 5):
        _, lb, _ = cv2.kmeans(data, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 1, cv2.KMEANS_PP_CENTERS)
        lb = lb.reshape(h, w)
        cands += [lb == k for k in range(K)]
    comps = []
    for m in cands:
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, el(5))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, el(9))
        n, cl, st, _ = cv2.connectedComponentsWithStats(m)
        for k in range(1, n):
            x, y, bw, bh, a = st[k]
            if a / (h * w) < 0.02 or a / (h * w) > 0.7:
                continue
            c = fill(cl == k).astype(np.uint8)
            cnt, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cnt, key=cv2.contourArea)
            if len(cnt) < 5:
                continue
            e = np.zeros_like(c); cv2.ellipse(e, cv2.fitEllipse(cnt), 1, -1)
            fit = (e & c).sum() / max(1, (e | c).sum())
            sol = cv2.contourArea(cnt) / max(cv2.contourArea(cv2.convexHull(cnt)), 1)
            ys, xs = np.nonzero(c)
            cd = np.hypot(xs.mean() / w - 0.5, ys.mean() / h - 0.5)
            touch = int(x <= 1) + int(y <= 1) + int(x + bw >= w - 1) + int(y + bh >= h - 1)
            comps.append((fit + 0.5 * sol - 0.6 * cd - 0.25 * touch, c))
    if not comps:
        return np.zeros((h, w), bool)
    comps.sort(key=lambda t: -t[0])
    union = comps[0][1].copy()
    for sc, c in comps[1:]:
        if sc >= 0.6 * comps[0][0] and (c & union).sum() == 0:
            union |= c
    mu = union.astype(np.uint8)
    gc = np.full(mu.shape, cv2.GC_PR_BGD, np.uint8)
    gc[cv2.dilate(mu, el(15)) > 0] = cv2.GC_PR_FGD
    gc[cv2.dilate(mu, el(41)) == 0] = cv2.GC_BGD
    gc[cv2.erode(mu, el(15)) > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(img, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
        r = keep_eggs(fill((gc == 1) | (gc == 3)), 4)
        return r if r.sum() > 0.5 * mu.sum() else mu > 0
    except cv2.error:
        return mu > 0

def _candidate_regions(img):
    """Shared: build scored candidate regions from K-means + bg-distance."""
    h, w = img.shape[:2]
    el = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cands = []
    _, th = cv2.threshold(bgdist(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cands.append(th > 0)
    data = lab.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(0)
    for K in (2, 3, 4, 5):
        _, lb, _ = cv2.kmeans(data, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 1, cv2.KMEANS_PP_CENTERS)
        lb = lb.reshape(h, w)
        cands += [lb == k for k in range(K)]
    comps = []
    for m in cands:
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, el(5))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, el(9))
        n, cl, st, _ = cv2.connectedComponentsWithStats(m)
        for k in range(1, n):
            x, y, bw, bh, a = st[k]
            if a / (h * w) < 0.02 or a / (h * w) > 0.7:
                continue
            c = fill(cl == k).astype(np.uint8)
            cnt, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cnt, key=cv2.contourArea)
            if len(cnt) < 5:
                continue
            e = np.zeros_like(c); cv2.ellipse(e, cv2.fitEllipse(cnt), 1, -1)
            fit = (e & c).sum() / max(1, (e | c).sum())
            sol = cv2.contourArea(cnt) / max(cv2.contourArea(cv2.convexHull(cnt)), 1)
            ys, xs = np.nonzero(c)
            cd = np.hypot(xs.mean() / w - 0.5, ys.mean() / h - 0.5)
            touch = int(x <= 1) + int(y <= 1) + int(x + bw >= w - 1) + int(y + bh >= h - 1)
            # brightness & saturation cue: eggs are bright and low-sat relative to border
            region_v = hsv[:, :, 2][c > 0].mean() / 255.0
            region_s = hsv[:, :, 1][c > 0].mean() / 255.0
            bright_cue = region_v - 0.3 * region_s  # higher = more egg-like
            comps.append((fit + 0.5 * sol - 0.6 * cd - 0.25 * touch + 0.2 * bright_cue, fit, sol, c))
    return comps

def _grabcut_refine(img, mu, max_eggs=4):
    el = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    gc = np.full(mu.shape, cv2.GC_PR_BGD, np.uint8)
    gc[cv2.dilate(mu, el(15)) > 0] = cv2.GC_PR_FGD
    gc[cv2.dilate(mu, el(41)) == 0] = cv2.GC_BGD
    gc[cv2.erode(mu, el(15)) > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(img, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
        r = keep_eggs(fill((gc == 1) | (gc == 3)), max_eggs)
        return r if r.sum() > 0.5 * mu.sum() else mu > 0
    except cv2.error:
        return mu > 0

def t09(img):
    """T07 + smarter multi-egg: accept regions with score >= 0.7*best AND fit >= 0.5 AND solidity >= 0.6."""
    comps = _candidate_regions(img)
    if not comps:
        return np.zeros(img.shape[:2], bool)
    comps.sort(key=lambda t: -t[0])
    best_sc = comps[0][0]
    union = comps[0][3].copy()
    for sc, fit, sol, c in comps[1:]:
        if sc >= 0.7 * best_sc and fit >= 0.5 and sol >= 0.6 and (c & union).sum() == 0:
            union |= c
    return _grabcut_refine(img, union.astype(np.uint8), 6)

def t10(img):
    """T09 + enhanced scoring: brightness/saturation cue already in _candidate_regions, plus
    add centre prior from T06 to break ties when all scores are close."""
    comps = _candidate_regions(img)
    if not comps:
        return np.zeros(img.shape[:2], bool)
    comps.sort(key=lambda t: -t[0])
    best_sc = comps[0][0]
    union = comps[0][3].copy()
    for sc, fit, sol, c in comps[1:]:
        if sc >= 0.75 * best_sc and fit >= 0.55 and sol >= 0.65 and (c & union).sum() == 0:
            union |= c
    return _grabcut_refine(img, union.astype(np.uint8), 6)

TRIALS = [
 ("T01", t01, "Gray blur + Otsu, largest component", "Simplest baseline", "blur 5x5, Otsu, 1 component"),
 ("T02", t02, "Lab colour distance from border background + Otsu", "Egg differs in colour from background, not necessarily in brightness", "median Lab of 5% border, Otsu, 1 component"),
 ("T03", t03, "T02 + morphology, hole filling, multi-egg components", "Remove noise, fill highlights, allow up to 4 eggs", "open 5, close 9, fill, keep up to 4 comps >=25% of largest"),
 ("T04", t04, "T03 mask used to initialise GrabCut", "Refine boundaries using colour models", "sure-FG erode15, sure-BG outside dilate41, 4 iters"),
 ("T05", t05, "T04 + ellipse regularisation", "Eggs are elliptical; smooth ragged edges", "fitEllipse used if IoU with blob > 0.85"),
 ("T06", t06, "Lab distance weighted by centre Gaussian prior", "Eggs are centred and compact; centre prior suppresses border clutter", "sigma 0.45 of size, open 7, close 15, up to 3 comps"),
 ("T07", t07, "Candidate regions (K-means Lab K=2..5 + bg-distance) scored by ellipse fit, solidity, centrality, border contact; best + GrabCut", "Border-colour background model fails on white/skin backgrounds; select by egg shape instead", "ellipse-fit + 0.5*solidity - 0.6*centre dist - 0.25*border touches; GrabCut 3 iters"),
 ("T08", t08, "T07 with relaxed multi-egg union (score >= 0.6*best, non-overlapping)", "Dev failures 44,144,131,142,121 had high precision but low recall: only one egg found", "union threshold 0.9 -> 0.6"),
 ("T09", t09, "T07 + smart multi-egg union (score>=0.7*best, fit>=0.5, sol>=0.6) + brightness/saturation cue in scoring", "T08 added wrong regions; filter by egg-shape thresholds and add bright/low-sat cue to scoring", "union thr 0.7, fit>=0.5, sol>=0.6, +0.2*(V-0.3*S)"),
 ("T10", t10, "T09 with stricter multi-egg filter (score>=0.75*best, fit>=0.55, sol>=0.65)", "T09 may still add marginal regions; tighten thresholds", "union thr 0.75, fit>=0.55, sol>=0.65, +0.2*(V-0.3*S)"),
]

def metrics(p, g):
    i, u, s = (p & g).sum(), (p | g).sum(), p.sum() + g.sum()
    return (1.0 if u == 0 else i / u, 1.0 if s == 0 else 2 * i / s, 1.0 if p.sum() == 0 else i / p.sum(), 1.0 if g.sum() == 0 else i / g.sum())

def run(fn, files):
    res = []
    for rel in files:
        d = load(rel)
        if d is None:
            print("missing", rel); continue
        img, gt = d
        res.append((rel, *metrics(fn(img), gt)))
    return res

split = json.load(open("outputs/split.json"))
w = csv.writer(open(LOG, "w", newline=""))
w.writerow(["trial", "change", "why", "params", "dev_n", "iou_mean", "iou_median", "dice_mean", "dice_median", "decision", "reason"])
best, best_iou = None, -1
for tid, fn, change, why, params in TRIALS:
    r = run(fn, split["dev"]); a = np.array([x[1:] for x in r])
    m = a[:, 0].mean()
    keep = m > best_iou + 0.005
    reason = f"improved dev IoU over best so far ({best_iou:.3f})" if keep else f"no gain over best so far ({best_iou:.3f})"
    w.writerow([tid, change, why, params, len(r), f"{m:.4f}", f"{np.median(a[:,0]):.4f}", f"{a[:,1].mean():.4f}", f"{np.median(a[:,1]):.4f}", "KEPT" if keep else "DISCARDED", reason])
    print(f"{tid} dev IoU mean={m:.3f} median={np.median(a[:,0]):.3f} Dice mean={a[:,1].mean():.3f} -> {'KEPT' if keep else 'DISCARDED'}")
    if keep:
        best, best_iou = (tid, fn), m

tid, fn = best
print("FINAL METHOD:", tid)
for name, files in (("dev", split["dev"]), ("eval", split["eval"])):
    r = run(fn, files); a = np.array([x[1:] for x in r])
    csv.writer(open(f"{OUT}/{name}_scores.csv", "w", newline="")).writerows([("image", "iou", "dice", "precision", "recall")] + r)
    print(f"{name.upper()} n={len(r)} IoU mean={a[:,0].mean():.3f} median={np.median(a[:,0]):.3f} Dice={a[:,1].mean():.3f} P={a[:,2].mean():.3f} R={a[:,3].mean():.3f}")
    if name == "eval":
        r_eval = r

worst = sorted(r_eval, key=lambda x: x[1])[:6]; rows = []
for rel, *_ in worst:
    img, gt = load(rel); p = fn(img)
    err = img.copy(); err[p & ~gt] = (0, 0, 255); err[gt & ~p] = (255, 0, 0)
    gtv = img.copy(); gtv[gt] = (0.5 * gtv[gt] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    pv = img.copy(); pv[p] = (0.5 * pv[p] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
    tile = [cv2.resize(x, (S, S)) for x in (img, gtv, pv, err)]
    cv2.putText(tile[0], os.path.basename(rel)[:18], (4, 16), 0, 0.5, (255, 255, 255), 1)
    rows.append(np.hstack(tile))
cv2.imwrite(f"{OUT}/worst_failures.jpg", np.vstack(rows))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    v = [x[1] for x in r_eval]; plt.hist(v, bins=10, range=(0, 1)); plt.xlabel("IoU (eval)"); plt.ylabel("images")
    plt.savefig(f"{OUT}/iou_hist.png", dpi=120)
except ImportError:
    print("matplotlib missing: pip install matplotlib")
print("saved to", OUT)