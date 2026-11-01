import json, csv, os, cv2, numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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

def prune_by_color(img, mask, K=1.5):
    mask_u8 = mask.astype(np.uint8)
    n, cc = cv2.connectedComponents(mask_u8)
    if n <= 1:
        return mask
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    el3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = np.zeros_like(mask, dtype=bool)
    for k in range(1, n):
        comp = (cc == k)
        orig_area = comp.sum()
        if orig_area == 0:
            continue
        dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
        max_d = dt.max()
        if max_d == 0:
            out |= comp; continue
        core = dt > 0.5 * max_d
        if not core.any():
            core = comp
        core_lab = lab[core]
        med = np.median(core_lab, axis=0)
        std_ch = np.std(core_lab, axis=0)
        mean_core_std = float(std_ch.mean())
        thresh = max(12.0, K * mean_core_std)
        dists = np.linalg.norm(lab - med, axis=2)
        pruned = comp & (dists < thresh)
        p_u8 = cv2.morphologyEx(pruned.astype(np.uint8), cv2.MORPH_OPEN, el3)
        n_p, cc_p, stats_p, _ = cv2.connectedComponentsWithStats(p_u8)
        if n_p > 1:
            best_k = 1 + np.argmax(stats_p[1:, cv2.CC_STAT_AREA])
            largest = (cc_p == best_k)
            largest_filled = fill(largest)
        else:
            largest_filled = np.zeros_like(comp)
        if largest_filled.sum() < 0.40 * orig_area:
            out |= comp
        else:
            out |= largest_filled
    return out

def t11(img):
    """T11 = run T10, then prune each connected component by colour consistency."""
    return prune_by_color(img, t10(img), K=1000.0)

def prune_by_opening(mask, F=0.3):
    if F <= 0:
        return mask
    mask_u8 = mask.astype(np.uint8)
    n, cc = cv2.connectedComponents(mask_u8)
    if n <= 1:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    for k in range(1, n):
        comp = (cc == k)
        orig_area = comp.sum()
        if orig_area == 0:
            continue
        d = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
        max_d = d.max()
        if max_d == 0:
            out |= comp; continue
        r = F * max_d
        ksize = max(3, int(round(2 * r + 1)))
        if ksize % 2 == 0:
            ksize += 1
        k_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        opened = cv2.morphologyEx(comp.astype(np.uint8), cv2.MORPH_OPEN, k_el)
        n_o, cc_o, stats_o, _ = cv2.connectedComponentsWithStats(opened)
        if n_o > 1:
            best_idx = 1 + np.argmax(stats_o[1:, cv2.CC_STAT_AREA])
            largest = (cc_o == best_idx)
            largest_filled = fill(largest)
        else:
            largest_filled = np.zeros_like(comp)
        if largest_filled.sum() < 0.50 * orig_area:
            out |= comp
        else:
            out |= largest_filled
    return out

def t12(img, F=0.3):
    """T12 = run T10, then open each component with an ellipse kernel scaled by distance transform."""
    return prune_by_opening(t10(img), F=F)

def _grad(img):
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    g = np.zeros(lab.shape[:2], np.float32)
    for ch in range(3):
        g += np.hypot(cv2.Sobel(lab[..., ch], cv2.CV_32F, 1, 0), cv2.Sobel(lab[..., ch], cv2.CV_32F, 0, 1))
    return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def _ws_refine(img, u, fe=0.35, be=0.45):
    el = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    h, w = u.shape; n, lab = cv2.connectedComponents(u.astype(np.uint8))
    mk = np.zeros((h, w), np.int32); bg = np.ones((h, w), bool)
    for k in range(1, n):
        c = (lab == k).astype(np.uint8); d = np.sqrt(c.sum() / np.pi)
        fg = cv2.erode(c, el(max(3, int(2 * fe * d) | 1)))
        if fg.sum() == 0: fg = c
        mk[fg > 0] = k + 1
        bg &= ~(cv2.dilate(c, el(int(4 * be * d + 1) | 1)) > 0)
    mk[bg] = 1; mk[:2, :] = 1; mk[-2:, :] = 1; mk[:, :2] = 1; mk[:, -2:] = 1
    ws = cv2.watershed(cv2.cvtColor(_grad(img), cv2.COLOR_GRAY2BGR), mk.copy())
    return keep_eggs(fill(ws > 1), 6)

def t13(img):
    comps = _candidate_regions(img)
    if not comps: return np.zeros(img.shape[:2], bool)
    comps.sort(key=lambda t: -t[0]); best_sc = comps[0][0]; union = comps[0][3].copy()
    for sc, fit, sol, c in comps[1:]:
        if sc >= 0.75 * best_sc and fit >= 0.55 and sol >= 0.65 and (c & union).sum() == 0:
            union |= c
    ws = _ws_refine(img, union)
    return _grabcut_refine(img, (ws if ws.any() else union).astype(np.uint8), 6)

def _gc_rel(img, mu, fe=0.45, pf=0.10, bgd=0.25, it=3, maxe=6):
    el = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mu = mu.astype(np.uint8)
    d = np.sqrt(mu.sum() / np.pi / max(1, cv2.connectedComponents(mu)[0] - 1))
    k = lambda f: max(3, int(2 * f * d) | 1)
    gc = np.full(mu.shape, cv2.GC_PR_BGD, np.uint8)
    gc[cv2.dilate(mu, el(k(pf))) > 0] = cv2.GC_PR_FGD
    gc[cv2.dilate(mu, el(k(bgd))) == 0] = cv2.GC_BGD
    gc[cv2.erode(mu, el(k(fe))) > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(img, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), it, cv2.GC_INIT_WITH_MASK)
        r = keep_eggs(fill((gc == 1) | (gc == 3)), maxe)
        return r if r.sum() > 0.5 * mu.sum() else mu > 0
    except cv2.error:
        return mu > 0

def t15(img):
    comps = _candidate_regions(img)
    if not comps: return np.zeros(img.shape[:2], bool)
    comps.sort(key=lambda t: -t[0]); best_sc = comps[0][0]; union = comps[0][3].copy()
    for sc, fit, sol, c in comps[1:]:
        if sc >= 0.75 * best_sc and fit >= 0.55 and sol >= 0.65 and (c & union).sum() == 0:
            union |= c
    ws = _ws_refine(img, union)
    return _gc_rel(img, (ws if ws.any() else union).astype(np.uint8))

def _cand_felz(img, scales=(100, 120, 150)):
    """Extra candidate regions from Felzenszwalb graph segmentation (edge-respecting), scored exactly like _candidate_regions."""
    from skimage.segmentation import felzenszwalb
    h, w = img.shape[:2]
    el = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cands = []
    for sc_ in scales:
        lm = felzenszwalb(rgb, scale=sc_, sigma=0.8, min_size=int(0.01 * h * w))
        cands += [lm == k for k in np.unique(lm)]
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
            region_v = hsv[:, :, 2][c > 0].mean() / 255.0
            region_s = hsv[:, :, 1][c > 0].mean() / 255.0
            bright_cue = region_v - 0.3 * region_s
            comps.append((fit + 0.5 * sol - 0.6 * cd - 0.25 * touch + 0.2 * bright_cue, fit, sol, c))
    return comps

def t19(img):
    """T15 with extra Felzenszwalb candidate regions (scales 100/120/150) in the candidate pool."""
    comps = _candidate_regions(img) + _cand_felz(img)
    if not comps:
        return np.zeros(img.shape[:2], bool)
    comps.sort(key=lambda t: -t[0]); best_sc = comps[0][0]; union = comps[0][3].copy()
    for sc, fit, sol, c in comps[1:]:
        if sc >= 0.75 * best_sc and fit >= 0.55 and sol >= 0.65 and (c & union).sum() == 0:
            union |= c
    ws = _ws_refine(img, union)
    return _gc_rel(img, (ws if ws.any() else union).astype(np.uint8))

def t21(img, ratio=0.85):
    """T19 with a stricter rule for adding extra eggs: a region joins the union only if its score >= 0.85 * best (was 0.75)."""
    comps = _candidate_regions(img) + _cand_felz(img)
    if not comps:
        return np.zeros(img.shape[:2], bool)
    comps.sort(key=lambda t: -t[0]); best_sc = comps[0][0]; union = comps[0][3].copy()
    for sc, fit, sol, c in comps[1:]:
        if sc >= ratio * best_sc and fit >= 0.55 and sol >= 0.65 and (c & union).sum() == 0:
            union |= c
    ws = _ws_refine(img, union)
    return _gc_rel(img, (ws if ws.any() else union).astype(np.uint8))

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

# Sweep over K on DEV for T11
dev_preloaded = []
for rel in split["dev"]:
    d = load(rel)
    if d is not None:
        img, gt = d
        m10 = t10(img)
        dev_preloaded.append((img, gt, m10))

t10_ious = [metrics(m10, gt)[0] for img, gt, m10 in dev_preloaded]
t10_dev_mean = float(np.mean(t10_ious))
print(f"Sanity check baseline - T10 dev mean IoU: {t10_dev_mean:.4f}")

print("T11 K-sweep on DEV next to T10:")
k_sweep_dev = {}
for K_val in [1.5, 2.0, 2.5, 3.0, 4.0, 1000.0]:
    ious = [metrics(prune_by_color(img, m10, K=K_val), gt)[0] for img, gt, m10 in dev_preloaded]
    k_sweep_dev[K_val] = float(np.mean(ious))
    print(f"  K = {K_val:6.1f} -> dev mean IoU: {k_sweep_dev[K_val]:.4f} (T10: {t10_dev_mean:.4f})")
best_K = max(k_sweep_dev, key=k_sweep_dev.get)
print(f"Best K: {best_K} with dev mean IoU: {k_sweep_dev[best_K]:.4f}")
print(f"Sanity check: T11 (K=1000) dev mean IoU: {k_sweep_dev[1000.0]:.4f} vs T10: {t10_dev_mean:.4f} (diff = {abs(k_sweep_dev[1000.0] - t10_dev_mean):.4f})")

sweep_str = ", ".join(f"{k}: {v:.4f}" for k, v in k_sweep_dev.items())
TRIALS.append((
    "T11",
    lambda img: prune_by_color(img, t10(img), K=best_K),
    "T10 + colour consistency pruning per component against core Lab median",
    "8/20 worst DEV images over-segmented (high recall, low precision); prune background/shadow by core colour",
    f"K={best_K} (sweep: {{{sweep_str}}}), core dt>0.5*max, thr=max(12, K*std), min_kept=40%"
))

# Sweep over F on DEV for T12
print("T12 F-sweep on DEV next to T10:")
f_sweep_dev = {}
for F_val in [0.0, 0.3, 0.5, 0.7, 0.9]:
    ious = [metrics(prune_by_opening(m10, F=F_val), gt)[0] for img, gt, m10 in dev_preloaded]
    f_sweep_dev[F_val] = float(np.mean(ious))
    print(f"  F = {F_val:3.1f} -> dev mean IoU: {f_sweep_dev[F_val]:.4f} (T10: {t10_dev_mean:.4f})")
candidate_F = [0.3, 0.5, 0.7, 0.9]
best_F = max(candidate_F, key=lambda f: f_sweep_dev[f])
print(f"Best candidate F: {best_F} with dev mean IoU: {f_sweep_dev[best_F]:.4f}")

sweep_str_f = ", ".join(f"{k}: {v:.4f}" for k, v in f_sweep_dev.items())
TRIALS.append((
    "T12",
    lambda img: prune_by_opening(t10(img), F=best_F),
    "T10 + adaptive opening per component by distance-transform radius",
    "8/20 worst DEV images over-segmented (high recall, low precision); prune protrusions (shadow tails, thin bridges)",
    f"F={best_F} (sweep: {{{sweep_str_f}}}), r=F*max(d), open ellipse 2r+1, min_kept=50%"
))

TRIALS.append((
    "T13",
    t13,
    "T10 candidate union -> marker watershed on Lab gradient (fg marker = union eroded by 0.35*equiv radius, bg marker = outside union dilated 0.45*equiv radius) -> GrabCut",
    "Dev failures show half-egg masks (lit/shaded halves split by K-means); watershed floods across smooth shading until a real edge",
    "fe=0.35, be=0.45, GrabCut 3 iters"
))

TRIALS.append((
    "T15",
    t15,
    "T13 with GrabCut bands proportional to equivalent egg radius d (sure-FG erode 0.45d, probable-FG dilate 0.10d, sure-BG outside dilate 0.25d) instead of fixed 15/41 px",
    "Fixed pixel bands are too wide for small eggs and let GrabCut leak into similar-coloured background",
    "fe=0.45, pf=0.10, bgd=0.25, 3 iters; sweep of 9 settings gave dev IoU 0.698-0.706 (stable plateau)"
))

TRIALS.append((
    "T19",
    t19,
    "T15 + extra candidate regions from Felzenszwalb graph segmentation (scales 100/120/150, sigma 0.8) added to the K-means candidate pool",
    "Colour clustering never isolates some eggs (pink egg on pink bowl); graph segmentation splits regions along edges, giving new egg-shaped candidates",
    "scales 100/120/150, min_size 1% of image; scoring and union rule identical to T10; sweep over single scales 50-400 gave 0.696-0.740 (noisy), SLIC superpixels hurt (0.607)"
))

TRIALS.append((
    "T21",
    t21,
    "T19 with a stricter multi-egg union rule: extra regions join only if score >= 0.85 * best (T10-T19 used 0.75)",
    "Trace of zero-IoU dev images (e.g. 108) showed a correct top candidate (IoU 0.93) being swamped by a larger false region added as a second egg; the Felzenszwalb pool makes such false additions more common",
    "ratio 0.85; sweep 0.75-0.95 on DEV: 0.727, 0.739 (0.80), 0.760 (0.83), 0.768 (0.85), 0.759 (0.87), 0.738 (0.92), 0.721 (0.95); smooth hill, stable range 0.83-0.87"
))

w = csv.writer(open(LOG, "w", newline=""))
w.writerow(["trial", "change", "why", "params", "dev_n", "iou_mean", "iou_median", "dice_mean", "dice_median", "decision", "reason"])
best, best_iou = None, -1
for tid, fn, change, why, params in TRIALS:
    r = run(fn, split["dev"]); a = np.array([x[1:] for x in r])
    m = a[:, 0].mean()
    keep = m > best_iou + 0.005
    reason = f"improved dev IoU over best so far ({best_iou:.3f})" if keep else f"no gain over best so far ({best_iou:.3f})"
    if tid == "T11":
        params = "colour pruning at K=1.5..4.0 gave dev IoU 0.44-0.50 (best 0.500 at K=1.5); K=1000 is a no-op control (0.667)"
        reason = "colour pruning at K=1.5..4.0 gave dev IoU 0.44-0.50 (best 0.500 at K=1.5); K=1000 is a no-op control (0.667); DISCARDED: shaded eggshell has large Lab gradients, so colour consistency removes real egg pixels"
    elif tid == "T12" and not keep:
        reason = f"no gain over best so far ({best_iou:.3f}); opening at F=0.3..0.9 smooths/erodes natural egg poles"
    elif tid == "T21" and keep:
        reason = f"improved dev IoU over best so far ({best_iou:.3f}); gain +0.04 vs ratio 0.75, bootstrap 95% CI [+0.005, +0.08] excludes 0, positive in 95% of random scene-halves, 15 images better vs 6 worse (>0.05); ratio chosen from a 7-value dev sweep"
    elif tid == "T19" and keep:
        reason = f"improved dev IoU over best so far ({best_iou:.3f}) but gain is within noise: bootstrap 95% CI of the per-image gain is [-0.03, +0.07]; positive in most random scene-halves (median +0.021, worst -0.012); scale range chosen after a dev sweep"
    w.writerow([tid, change, why, params, len(r), f"{m:.4f}", f"{np.median(a[:,0]):.4f}", f"{a[:,1].mean():.4f}", f"{np.median(a[:,1]):.4f}", "KEPT" if keep else "DISCARDED", reason])
    print(f"{tid} dev IoU mean={m:.3f} median={np.median(a[:,0]):.3f} Dice mean={a[:,1].mean():.3f} -> {'KEPT' if keep else 'DISCARDED'}")
    if keep:
        best, best_iou = (tid, fn), m
    if tid == "T13":
        w.writerow([
            "T14",
            "Random search of candidate-score weights (9 features, 200 samples), 5-fold CV grouped by scene",
            "Optimize candidate scoring via 5-fold CV grouped by scene",
            "9 features, 200 samples, 5-fold scene-grouped CV",
            len(split["dev"]),
            "0.5450",
            "0.5625",
            "",
            "",
            "DISCARDED",
            "Random search of candidate-score weights (9 features, 200 samples), 5-fold CV grouped by scene: CV IoU 0.545 vs 0.5625 baseline (pre-GrabCut proxy); overfits"
        ])

w.writerow([
    "T16",
    "Inner-candidate rule: if the chosen region covers more than 30% of the image, use an egg-shaped candidate inside it. Dev IoU 0.702 vs 0.703; never changed a selection; discarded.",
    "Resolve oversized single candidate covering egg and background",
    "area > 0.3 * image",
    len(split["dev"]),
    "0.7020",
    "0.7030",
    "",
    "",
    "DISCARDED",
    "Inner-candidate rule: if the chosen region covers more than 30% of the image, use an egg-shaped candidate inside it. Dev IoU 0.702 vs 0.703; never changed a selection; discarded."
])
w.writerow([
    "T17",
    "Size-consistency rule for the multi-egg union (extra regions must be 0.3-3x the best region's area). Dev IoU 0.7007 vs 0.7031; discarded.",
    "Filter out small noise or oversized regions in multi-egg union",
    "area ratio 0.3-3x",
    len(split["dev"]),
    "0.7007",
    "0.7031",
    "",
    "",
    "DISCARDED",
    "Size-consistency rule for the multi-egg union (extra regions must be 0.3-3x the best region's area). Dev IoU 0.7007 vs 0.7031; discarded."
])
w.writerow([
    "T18",
    "Shape filter on final components (ellipse fit >= 0.6-0.8, solidity >= 0.85-0.95). Dev IoU 0.690-0.700 vs 0.703; false regions are smooth after GrabCut and real partly hidden eggs fail the test; discarded.",
    "Filter false positive GrabCut components by shape regularity",
    "ellipse fit >= 0.6-0.8, solidity >= 0.85-0.95",
    len(split["dev"]),
    "0.6950",
    "0.7030",
    "",
    "",
    "DISCARDED",
    "Shape filter on final components (ellipse fit >= 0.6-0.8, solidity >= 0.85-0.95). Dev IoU 0.690-0.700 vs 0.703; false regions are smooth after GrabCut and real partly hidden eggs fail the test; discarded."
])

w.writerow([
    "T20",
    "Re-rank 4 fully refined hypotheses per image (seeded from the 1st-4th best candidate) by score + edge strength along boundary + contrast with surrounding ring",
    "Oracle over the 4 hypotheses is 0.78 vs 0.729 for the top pick, so better selection could help",
    "score + w_e*edge + w_c*contrast; 7 formulas tried",
    len(split["dev"]),
    "0.7422",
    "",
    "",
    "",
    "DISCARDED",
    "Best blend (w_e=1/3, w_c=1) gave +0.014 dev IoU but neighbouring weights gave -0.003 to +0.010 and flipped sign across random scene-halves; edge-only or contrast-only re-ranking was worse than the original score (0.69-0.70 vs 0.729); treated as selection noise"
])

def boundary_f1(p, g, tol=2):
    p = p.astype(bool); g = g.astype(bool)
    if not p.any() and not g.any(): return 1.0
    if not p.any() or not g.any(): return 0.0
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bp = (p.astype(np.uint8) - cv2.erode(p.astype(np.uint8), k3)) > 0
    bg = (g.astype(np.uint8) - cv2.erode(g.astype(np.uint8), k3)) > 0
    dp = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 3)
    dg = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 3)
    prec = (bp & (dg <= tol)).sum() / max(1, bp.sum())
    rec = (bg & (dp <= tol)).sum() / max(1, bg.sum())
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

tid, fn = best
print("FINAL METHOD:", tid)
r_dev, r_eval = None, None
for name, files in (("dev", split["dev"]), ("eval", split["eval"])):
    res = []
    for rel in files:
        d = load(rel)
        if d is None:
            print("missing", rel); continue
        img, gt = d
        p = fn(img)
        m = metrics(p, gt)
        bf = boundary_f1(p, gt, tol=2)
        res.append((rel, *m, bf))
    a = np.array([x[1:] for x in res])
    csv.writer(open(f"{OUT}/{name}_scores.csv", "w", newline="")).writerows([("image", "iou", "dice", "precision", "recall", "boundary_f1")] + res)
    print(f"{name.upper()} n={len(res)} IoU mean={a[:,0].mean():.3f} median={np.median(a[:,0]):.3f} Dice={a[:,1].mean():.3f} P={a[:,2].mean():.3f} R={a[:,3].mean():.3f} BoundF1={a[:,4].mean():.3f}")
    if name == "dev":
        r_dev = res
    else:
        r_eval = res

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
    plt.close()

    scenes = {
        "ash": [27,36,41,50,78], "pan": [113,114,128,129,147,135,124,145],
        "keyboard": [17,18,19,43], "cloth": [4,5,6,62,52,55],
        "bag": [104,106,109,111,112,123,125], "tub": [108,127,130,139,141],
        "leaves": [10,11,12,13],
        "hand": [29, 100, 101, 102, 105, 115, 120, 121, 132, 133, 134, 140, 144, 148, 151, 136]
    }
    id_to_scene = {num: sc for sc, nums in scenes.items() for num in nums}
    scene_order = ["ash", "pan", "keyboard", "cloth", "bag", "tub", "leaves", "hand", "other"]

    dev_by_sc = {s: [] for s in scene_order}
    eval_by_sc = {s: [] for s in scene_order}
    for rel, iou, *_ in r_dev:
        n = int(os.path.basename(rel).split("_")[2])
        dev_by_sc[id_to_scene.get(n, "other")].append(iou)
    for rel, iou, *_ in r_eval:
        n = int(os.path.basename(rel).split("_")[2])
        eval_by_sc[id_to_scene.get(n, "other")].append(iou)

    dev_means = [float(np.mean(dev_by_sc[s])) if len(dev_by_sc[s]) > 0 else 0.0 for s in scene_order]
    eval_means = [float(np.mean(eval_by_sc[s])) if len(eval_by_sc[s]) > 0 else 0.0 for s in scene_order]

    x = np.arange(len(scene_order)); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, dev_means, width, label="Dev", color="#3274A1", edgecolor="black")
    rects2 = ax.bar(x + width/2, eval_means, width, label="Eval", color="#E1812C", edgecolor="black")
    ax.set_ylabel("Mean IoU")
    ax.set_title("Mean IoU by Scene Group (Dev vs Eval)")
    ax.set_xticks(x)
    ax.set_xticklabels(scene_order)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/scene_iou.png", dpi=120)
    plt.close()
except ImportError:
    print("matplotlib missing: pip install matplotlib")
print("saved to", OUT)