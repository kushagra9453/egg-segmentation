"""Task 4 – Egg tracking in video using classical segmentation (T10).
- Imports final segmentation T10 without executing 14_classical.py main body
- Resizes to 320 px on longest side
- Evaluates every 2nd frame, reuses masks on skipped frames
- Watershed-splits touching eggs
- Hungarian algorithm matching with cost (1-IoU) + 0.3 * (dist / diag) and 0.8 gate
- Confirms tracks only after 3 consecutive detections
- Drops tracks after 30 missed frames
- Ignores blobs under 0.5% of the frame
- Exports H.264 MP4 using imageio-ffmpeg at source fps
"""
import os, sys, subprocess, cv2, numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import maximum_filter

def load_t10_segmentation():
    """Load T10 and its dependencies from 14_classical.py without running trials."""
    script_path = os.path.join(os.path.dirname(__file__), "14_classical.py")
    with open(script_path, "r") as f:
        code = f.read()
    code_helpers = code.split('split = json.load(open("outputs/split.json"))')[0]
    ns = {}
    exec(code_helpers, ns)
    return ns["t10"]

def get_track_color(tid):
    """Generate distinct, bright RGB colors for tracks."""
    palette = [
        (0, 255, 0),     # bright green
        (0, 165, 255),   # orange
        (255, 255, 0),   # cyan
        (203, 192, 255), # pink
        (0, 255, 255),   # yellow
        (255, 144, 30),  # dodger blue
        (180, 105, 255), # hot pink
        (50, 205, 50),   # lime green
        (0, 215, 255),   # gold
    ]
    return palette[(tid - 1) % len(palette)]

def compute_iou(m1, m2):
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter / union) if union > 0 else 0.0

def split_eggs_watershed(mask, min_area):
    """Split touching eggs using distance-transform watershed."""
    mask_u8 = mask.astype(np.uint8)
    n_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
    detections = []

    for k in range(1, n_cc):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        blob = (cc_labels == k).astype(np.uint8)

        dt = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        filt_size = max(9, int(min(w, h) * 0.25))
        if filt_size % 2 == 0:
            filt_size += 1

        local_max = (dt == maximum_filter(dt, size=filt_size)) & (dt >= max(dt.max() * 0.4, 4.0))
        n_peaks, peak_labels = cv2.connectedComponents(local_max.astype(np.uint8))

        if n_peaks <= 2:
            ys, xs = np.nonzero(blob)
            if len(xs) > 0:
                detections.append((blob > 0, float(xs.mean()), float(ys.mean())))
        else:
            markers = np.zeros_like(blob, dtype=np.int32)
            markers[local_max] = peak_labels[local_max] + 1
            sure_bg = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) == 0
            markers[sure_bg] = 1

            bgr = cv2.cvtColor(blob * 255, cv2.COLOR_GRAY2BGR)
            cv2.watershed(bgr, markers)

            for lbl in range(2, n_peaks + 1):
                reg = (markers == lbl) & (blob > 0)
                if reg.sum() >= min_area:
                    ys, xs = np.nonzero(reg)
                    if len(xs) > 0:
                        detections.append((reg, float(xs.mean()), float(ys.mean())))
    return detections

class Track:
    def __init__(self, mask, cx, cy):
        self.track_id = None
        self.is_confirmed = False
        self.consecutive_hits = 1
        self.missed = 0
        self.mask = mask
        self.cx = cx
        self.cy = cy
        self.color = (0, 255, 0)

def draw_tracks(vis, active_tracks):
    for mask, cx, cy, tid, color in active_tracks:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 2)
        label = f"ID {tid}"
        org = (max(5, int(cx) - 15), max(15, int(cy) - 10))
        cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

def main():
    video_path = "data/video/eggs_clip.mov"
    final_output = "outputs/tracking_output.mp4"
    temp_output = "outputs/tracking_temp.mp4"
    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        sys.exit(1)

    t10 = load_t10_segmentation()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize to 320 px on longest side
    scale = 320.0 / max(W0, H0)
    W = int(W0 * scale)
    H = int(H0 * scale)
    diag = float(np.hypot(W, H))
    min_area = 0.005 * W * H  # ignore blobs under 0.5% of the frame

    print(f"Input Video: {W0}x{H0} @ {fps:.2f} fps ({total_frames} frames)")
    print(f"Processing Resolution: {W}x{H} (scale {scale:.4f}), min_area: {min_area:.1f} px, diag: {diag:.1f} px")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_output, fourcc, fps, (W, H))
    if not writer.isOpened():
        print("Error: Could not open VideoWriter.")
        sys.exit(1)

    tracks = []
    next_track_id = 1
    unique_confirmed_ids = set()
    last_drawn_tracks = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

        # Process segmentation every 2nd frame; reuse on skipped frames
        if frame_idx % 2 == 0:
            mask = t10(small)
            dets = split_eggs_watershed(mask, min_area)

            # Hungarian matching with cost: (1 - IoU) + 0.3 * (dist / diag)
            if tracks and dets:
                cost_matrix = np.full((len(tracks), len(dets)), 1e6, dtype=np.float32)
                for i, tr in enumerate(tracks):
                    for j, (dm, dx, dy) in enumerate(dets):
                        iou_val = compute_iou(tr.mask, dm)
                        dist = np.hypot(tr.cx - dx, tr.cy - dy)
                        cost_matrix[i, j] = (1.0 - iou_val) + 0.3 * (dist / diag)

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                matched_tracks, matched_dets = set(), set()
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] <= 0.8:  # Gate: reject matches with cost > 0.8
                        tr = tracks[r]
                        dm, dx, dy = dets[c]
                        tr.mask = dm
                        tr.cx = dx
                        tr.cy = dy
                        tr.consecutive_hits += 1
                        tr.missed = 0
                        if not tr.is_confirmed and tr.consecutive_hits >= 3:
                            tr.is_confirmed = True
                            tr.track_id = next_track_id
                            unique_confirmed_ids.add(next_track_id)
                            tr.color = get_track_color(next_track_id)
                            next_track_id += 1
                        matched_tracks.add(r)
                        matched_dets.add(c)

                # Unmatched tracks
                for i, tr in enumerate(tracks):
                    if i not in matched_tracks:
                        tr.missed += 1
                        tr.consecutive_hits = 0

                # Unmatched detections -> new tentative tracks
                for j, (dm, dx, dy) in enumerate(dets):
                    if j not in matched_dets:
                        tracks.append(Track(dm, dx, dy))
            else:
                for tr in tracks:
                    tr.missed += 1
                    tr.consecutive_hits = 0
                for dm, dx, dy in dets:
                    tracks.append(Track(dm, dx, dy))

            # Prune tracks
            # Tentative dropped if not consecutive (missed > 0); confirmed dropped after 30 missed frames
            alive = []
            for tr in tracks:
                if not tr.is_confirmed:
                    if tr.missed == 0:
                        alive.append(tr)
                else:
                    if tr.missed <= 30:
                        alive.append(tr)
            tracks = alive

            # Active confirmed tracks currently visible in this frame
            last_drawn_tracks = [
                (tr.mask, tr.cx, tr.cy, tr.track_id, tr.color)
                for tr in tracks if tr.is_confirmed and tr.missed == 0
            ]

        # Draw confirmed tracks (reused on skipped frames)
        vis = small.copy()
        draw_tracks(vis, last_drawn_tracks)
        writer.write(vis)

        frame_idx += 1
        if frame_idx % 500 == 0 or frame_idx == total_frames:
            n_active = sum(1 for tr in tracks if tr.is_confirmed and tr.missed == 0)
            print(f"  Frame {frame_idx}/{total_frames} | Active Confirmed: {n_active} | Total Unique IDs: {len(unique_confirmed_ids)}")

    cap.release()
    writer.release()

    # Re-encode as H.264 using imageio-ffmpeg
    print("Re-encoding video with H.264 codec...")
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"], check=True)
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe, "-y",
        "-i", temp_output,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "22",
        final_output
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    if os.path.exists(temp_output):
        os.remove(temp_output)

    print("\n================ TRACKING SUMMARY ================")
    print(f"Total frames processed: {frame_idx}")
    print(f"Unique egg IDs tracked: {len(unique_confirmed_ids)} (IDs: {sorted(list(unique_confirmed_ids))})")
    print(f"Output saved to: {final_output}")
    print("==================================================")

if __name__ == "__main__":
    main()
