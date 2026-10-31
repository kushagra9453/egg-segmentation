"""Task 4: egg tracking. Classical only: HSV 'bright + unsaturated' blob detection,
shape filtering, Hungarian matching on centroid distance, persistent IDs.
Usage: python scripts/15_track.py [input_video] [output_video]"""
import sys, os, subprocess, cv2, numpy as np
from scipy.optimize import linear_sum_assignment

VID = sys.argv[1] if len(sys.argv) > 1 else "data/video/eggs_clip.mov"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/tracking_output.mp4"
W = 640                   # working width
PLAYER_H = 0.88           # keep top 88% of frame: YouTube player only (drops title bar/comments)
MAX_MISS, GATE, CONFIRM = 90, 0.12, 3   # frames kept alive, max jump (frac of width), hits to confirm

def detect(img):
    """Return list of egg masks: bright, low-saturation, compact, elliptical blobs."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 150) & (hsv[..., 1] < 60)).astype(np.uint8)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, el), cv2.MORPH_CLOSE, el)
    # scene gate: eggs sit on a dark table, so the lower half must be dark overall
    if hsv[h // 2:, :, 2].mean() > 110:
        return []
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        x, y, bw, bh, a = st[k]
        if not (0.003 < a / (h * w) < 0.05) or (y + bh / 2) < 0.45 * h:
            continue
        if x <= 1 or x + bw >= w - 1 or max(bw, bh) / max(1, min(bw, bh)) > 2.0:
            continue
        c = (lab == k).astype(np.uint8)
        cnt, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnt, key=cv2.contourArea)
        if cv2.contourArea(cnt) / max(cv2.contourArea(cv2.convexHull(cnt)), 1) < 0.88:
            continue
        out.append(c)
    return out

def cen(m):
    ys, xs = np.nonzero(m); return np.array([xs.mean(), ys.mean()])

cap = cv2.VideoCapture(VID)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
tracks, nid, frames, frame_counts, frames_out = {}, 1, 0, [], []
rng = np.random.RandomState(3); colors = {}
while True:
    ok, fr = cap.read()
    if not ok: break
    f = W / fr.shape[1]
    fr = cv2.resize(fr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    img = fr[: int(PLAYER_H * fr.shape[0])]
    dets = detect(img); H, Wd = img.shape[:2]
    tids = list(tracks); matched, usedd = set(), set()
    if tids and dets:
        C = np.array([[np.linalg.norm(tracks[t]["c"] - cen(d)) / Wd for d in dets] for t in tids])
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] < GATE:
                t = tids[i]; tracks[t].update(m=dets[j], c=cen(dets[j]), miss=0)
                tracks[t]["hits"] += 1; matched.add(t); usedd.add(j)
    for t in tids:
        if t not in matched: tracks[t]["miss"] += 1
    for t in [t for t in tracks if tracks[t]["miss"] > MAX_MISS or (tracks[t]["hits"] < CONFIRM and tracks[t]["miss"] > 3)]:
        del tracks[t]
    for j, d in enumerate(dets):
        if j not in usedd:
            tracks[nid] = dict(m=d, c=cen(d), miss=0, hits=1); colors[nid] = tuple(int(v) for v in rng.randint(80, 255, 3)); nid += 1
    vis = fr.copy(); shown = 0
    for t, tr in tracks.items():
        if tr["hits"] >= CONFIRM and tr["miss"] == 0:
            cnt, _ = cv2.findContours(tr["m"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnt, -1, colors[t], 2)
            x, y = tr["c"].astype(int)
            cv2.putText(vis, f"Egg {t}", (x - 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[t], 2); shown += 1
    frame_counts.append(shown); frames_out.append(vis); frames += 1
    if frames % 300 == 0: print("frames", frames, flush=True)

# --- trim to the egg section: longest run of frames with eggs (gaps < 2 s merged), +0.2 s padding
pad, gap = int(0.2 * fps), int(2 * fps)
runs, start, last = [], None, None
for i, c in enumerate(frame_counts):
    if c > 0:
        if start is None: start = i
        elif i - last > gap: runs.append((start, last)); start = i
        last = i
if start is not None: runs.append((start, last))
a, b = max(runs, key=lambda r: r[1] - r[0]) if runs else (0, len(frames_out) - 1)
a, b = max(0, a - pad), min(len(frames_out) - 1, b + pad)
frames_out, frame_counts = frames_out[a:b + 1], frame_counts[a:b + 1]
print(f"trimmed to frames {a}-{b} ({(b - a + 1) / fps:.1f} s)")

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
tmp = OUT + ".tmp.mp4"
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames_out[0].shape[1], frames_out[0].shape[0]))
for v in frames_out: vw.write(v)
vw.release()
try:
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    ff = "ffmpeg"
try:
    subprocess.run([ff, "-y", "-i", tmp, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT], check=True, capture_output=True)
    os.remove(tmp)
except Exception:
    os.replace(tmp, OUT); print("WARNING: no ffmpeg, saved as mp4v; run: pip install imageio-ffmpeg and rerun")
print(f"done. frames={frames} track IDs created={nid-1} max eggs in one frame={max(frame_counts)} frames with >=1 egg shown={sum(c>0 for c in frame_counts)} -> {OUT}")