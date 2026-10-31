import cv2, glob, os, shutil
import numpy as np

fixes = {117: (0.57, 0.31, 0.11, 0.13), 135: (0.70, 0.50, 0.14, 0.17)}  # cx, cy, rx, ry as fractions
os.makedirs("outputs/sam_final/masks_ellipse_backup", exist_ok=True)
tiles = []
for i, (cx, cy, rx, ry) in fixes.items():
    img_p = [p for p in glob.glob("data/egg_v1i_folder/**/*.jpg", recursive=True) if os.path.basename(p).split("_")[2] == str(i)][0]
    im = cv2.imread(img_p)
    H, W = im.shape[:2]
    stem = os.path.splitext(os.path.basename(img_p))[0]
    t = glob.glob(f"outputs/sam_final/masks/{stem}*")[0]
    shutil.copy(t, "outputs/sam_final/masks_ellipse_backup/" + os.path.basename(t))
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (int(cx * W), int(cy * H)), (int(rx * W), int(ry * H)), 0, 0, 360, 255, -1)
    cv2.imwrite(t, m)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, cnts, -1, (0, 255, 0), 3)
    tiles.append(cv2.resize(im, (500, 500)))
    print("fixed", i)
cv2.imwrite("outputs/sam_final/ellipse_check.jpg", np.hstack(tiles))