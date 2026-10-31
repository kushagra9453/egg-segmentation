"""Task 4 – Egg tracking in video using classical segmentation.
Processes every frame but uses a fast segmentation path (bgdist + centre prior + morphology)
for real-time-ish speed, with watershed splitting for touching eggs.
Tracks eggs across frames using the Hungarian algorithm.
"""
import sys, os, cv2, numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt

# ── parameters ──
VIDEO   = "data/video/eggs_clip.mov"
OUT_VID = "outputs/tracking_output.mp4"
LONG_SIDE = 240          # smaller = faster
MAX_MISS  = 8            # frames a track can be missing before deletion
DIST_NORM = 80.0         # normalise centroid distance
IOU_W     = 1.0
DIST_W    = 0.3
MIN_AREA  = 80           # min blob area at this scale

# ── colours for track IDs ──
def track_colour(tid):
    np.random.seed(tid * 7 + 3)
    return tuple(int(x) for x in np.random.randint(60, 255, 3))

# ── fast egg segmentation (based on T06: bgdist + centre prior) ──
def segment_egg_fast(img):
    """Fast segmentation: Lab bg-distance + Gaussian centre prior + morphology."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(3, int(0.05 * max(h, w)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    d = np.linalg.norm(lab - np.median(border, 0), axis=2)
    d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    # centre prior
    yy, xx = np.mgrid[0:h, 0:w]
    c = np.exp(-(((xx - w / 2) / (0.45 * w)) ** 2 + ((yy - h / 2) / (0.45 * h)) ** 2))
    s = cv2.normalize(d * c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    el = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(th, cv2.MORPH_OPEN, el(5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, el(9))
    # fill holes
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m); cv2.drawContours(filled, cnt, -1, 255, -1)
    return filled > 0

# ── split touching eggs using distance-transform watershed ──
def split_eggs(mask):
    """Return list of (binary_mask, cx, cy) for individual eggs."""
    mask8 = mask.astype(np.uint8)
    n0, cc0, stats, _ = cv2.connectedComponentsWithStats(mask8)
    results = []
    for k in range(1, n0):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            continue
        blob = (cc0 == k).astype(np.uint8)
        # try watershed split
        dt = distance_transform_edt(blob)
        thresh = max(dt.max() * 0.45, 2)
        peaks = (dt >= thresh).astype(np.uint8)
        n_peaks, markers = cv2.connectedComponents(peaks)
        if n_peaks <= 2:
            ys, xs = np.nonzero(blob)
            results.append((blob > 0, xs.mean(), ys.mean()))
        else:
            markers = markers.astype(np.int32)
            markers[blob == 0] = 0
            markers += 1
            markers[blob == 0] = 0
            bgr = cv2.cvtColor(blob * 255, cv2.COLOR_GRAY2BGR)
            cv2.watershed(bgr, markers)
            for lbl in range(2, markers.max() + 1):
                region = (markers == lbl) & (blob > 0)
                if region.sum() >= MIN_AREA:
                    ys, xs = np.nonzero(region)
                    results.append((region, xs.mean(), ys.mean()))
    return results

# ── tracking ──
class Track:
    __slots__ = ("tid", "mask", "cx", "cy", "missed", "colour")
    def __init__(self, tid, mask, cx, cy):
        self.tid = tid; self.mask = mask; self.cx = cx; self.cy = cy
        self.missed = 0; self.colour = track_colour(tid)

def iou(a, b):
    i = (a & b).sum(); u = (a | b).sum()
    return i / u if u > 0 else 0.0

def match(tracks, dets):
    if not tracks or not dets:
        return [], list(range(len(dets))), list(range(len(tracks)))
    nt, nd = len(tracks), len(dets)
    cost = np.full((nt, nd), 1e9)
    for i, tr in enumerate(tracks):
        for j, (dm, dx, dy) in enumerate(dets):
            iou_val = iou(tr.mask, dm)
            dist = np.hypot(tr.cx - dx, tr.cy - dy) / DIST_NORM
            cost[i, j] = IOU_W * (1.0 - iou_val) + DIST_W * dist
    ri, ci = linear_sum_assignment(cost)
    matched, udet, utrk = [], set(range(nd)), set(range(nt))
    for r, c in zip(ri, ci):
        if cost[r, c] < 1.5:
            matched.append((r, c))
            udet.discard(c); utrk.discard(r)
    return matched, sorted(udet), sorted(utrk)


def main():
    os.makedirs(os.path.dirname(OUT_VID), exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("ERROR: cannot open", VIDEO); return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W0     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale  = LONG_SIDE / max(W0, H0)
    nw, nh = int(W0 * scale), int(H0 * scale)
    print(f"Video {W0}x{H0} @ {fps:.1f}fps, {total} frames -> {nw}x{nh}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUT_VID, fourcc, fps, (nw, nh))
    if not out.isOpened():
        print("ERROR: cannot create writer"); return

    tracks = []
    next_id = 1
    unique_ids = set()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

        # segment
        mask = segment_egg_fast(small)
        dets = split_eggs(mask)

        # match
        matched, udet, utrk = match(tracks, dets)

        # update matched
        for ti, di in matched:
            dm, dx, dy = dets[di]
            tracks[ti].mask = dm; tracks[ti].cx = dx; tracks[ti].cy = dy
            tracks[ti].missed = 0

        # missed
        for ti in utrk:
            tracks[ti].missed += 1

        # new tracks
        for di in udet:
            dm, dx, dy = dets[di]
            tracks.append(Track(next_id, dm, dx, dy))
            unique_ids.add(next_id)
            next_id += 1

        # prune dead
        tracks = [t for t in tracks if t.missed <= MAX_MISS]

        # draw
        vis = small.copy()
        for t in tracks:
            if t.missed > 0:
                continue
            cnt, _ = cv2.findContours(t.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnt, -1, t.colour, 2)
            cv2.putText(vis, f"ID {t.tid}", (int(t.cx) - 15, int(t.cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, t.colour, 1, cv2.LINE_AA)

        out.write(vis)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  frame {frame_idx}/{total}  tracks={len(tracks)}  unique={len(unique_ids)}")

    cap.release()
    out.release()

    print(f"\nDone: {frame_idx} frames processed, {len(unique_ids)} unique IDs")
    print(f"Output: {OUT_VID}")

    # verify output
    cap2 = cv2.VideoCapture(OUT_VID)
    if cap2.isOpened():
        fc = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        vw = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Verify: {fc} frames, {vw}x{vh}")
        cap2.release()

if __name__ == "__main__":
    main()
