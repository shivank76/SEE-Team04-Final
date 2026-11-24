"""SEE Assignment 4 - camera calibration (Zhang 2000) with OpenCV.

Inputs (this folder):
  data/images/img_*.png         checkerboard images (code/capture.py)
  data/board_measurements.csv   ruler readings of the printed squares
Outputs (this folder):
  results.tex                   every number quoted in the report
  data/calibration.json         camera parameters and their standard deviations
  data/per_view.csv             reprojection error and board pose per image
  data/poses_rows.tex           rows of the pose table
  data/detected/*.jpg           images with the detected corners drawn
  figures/*.pdf                 sample images, all images

Run:  python3 code/calibrate.py     (from this folder)
k3 is fixed at 0: no corners lie near the image corners, so k3 cannot be estimated (its
standard deviation is about as large as its value) and it does not lower the RMS error.
No randomness is used.
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG, IMAGES = ROOT / "data", ROOT / "figures", ROOT / "data" / "images"
DETECTED = DATA / "detected"
for d in (FIG, DETECTED):
    d.mkdir(exist_ok=True)

PATTERN = (9, 6)            # inner corners (columns, rows)
RULER_MM = 0.5              # reading error of one ruler length
DUP_PX = 20                 # consecutive images whose corners moved less are one pose
TURN_DEG = 10               # a pose turned or rotated by less than this counts as straight
NEAR_CM, FAR_CM = 40, 60    # poses closer / farther than this are called close / far
OPENCV_MIN = 10             # minimum number of images recommended by the OpenCV tutorial
SAMPLES = ["img_01.png", "img_06.png", "img_20.png"]
FLAGS = cv2.CALIB_FIX_K3


# ---------------------------------------------------------------- helpers
def taylor_decimals(u):
    """Decimal places for a value with uncertainty u (Lecture 1, Taylor): one
    significant figure, two if the leading digit is 1."""
    e = int(np.floor(np.log10(u)))
    mant = u / 10**e
    if mant >= 9.5:
        return max(0, -e)
    return max(0, -e + (1 if mant < 2 else 0))


def tex_num(v, d):
    """Number with d decimals, minus sign for LaTeX (text and math mode), no '-0.0'."""
    v = 0.0 if round(v, d) == 0 else v
    s = f"{v:.{d}f}"
    return s.replace("-", r"\ensuremath{-}") if s.startswith("-") else s


def val_unc(name, v, u):
    """Macros name (value) and nameU (uncertainty), both rounded per Taylor."""
    d = taylor_decimals(u)
    macros[name], macros[name + "U"] = tex_num(v, d), tex_num(u, d)


def detect(gray):
    found, c = cv2.findChessboardCornersSB(
        gray, PATTERN, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if found:
        return c
    found, c = cv2.findChessboardCorners(gray, PATTERN)
    if not found:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)


def calibrate(img_pts, flags=FLAGS):
    return cv2.calibrateCameraExtended([grid] * len(img_pts), img_pts, size, None, None,
                                       flags=flags)


def describe(vs):
    """Pose in words from the mean angles and distance of its images."""
    lr, ud, rot = (np.mean([v[k] for v in vs]) for k in ("turn_lr_deg", "turn_ud_deg", "rot_deg"))
    d = np.mean([v["distance_mm"] for v in vs]) / 10
    if abs(rot) >= TURN_DEG:
        text = "rotated in its plane"
    elif max(abs(lr), abs(ud)) < TURN_DEG:
        text = "straight"
    elif abs(lr) >= abs(ud):
        text = "turned, " + ("right" if lr > 0 else "left") + " edge away"
    else:
        text = "tilted, " + ("bottom" if ud > 0 else "top") + " edge away"
    return text + (", close" if d < NEAR_CM else ", far" if d > FAR_CM else "")


# ---------------------------------------------------------------- input
rows = list(csv.DictReader(open(DATA / "board_measurements.csv")))
n_sq = sum(int(r["n_squares"]) for r in rows)
square = sum(float(r["length_mm"]) for r in rows) / n_sq
square_u = RULER_MM * np.sqrt(len(rows)) / n_sq     # independent reading errors
grid = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
grid[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * square

names, img_pts, size, skipped = [], [], None, []
for path in sorted(IMAGES.glob("img_*.png")):
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = size or gray.shape[::-1]
    c = detect(gray) if gray.shape[::-1] == size else None
    if c is None:
        skipped.append(path.name)
        continue
    names.append(path.name)
    img_pts.append(c)
    cv2.drawChessboardCorners(img, PATTERN, c, True)
    cv2.imwrite(str(DETECTED / path.name.replace(".png", ".jpg")), img)
W, H = size
print(f"{len(names)} images used, size {size}, skipped: {skipped or 'none'}")

# ---------------------------------------------------------------- calibration
rms, K, dist, rvecs, tvecs, std_in, _, per_view = calibrate(img_pts)
std_in, dist, per_view = std_in.ravel(), dist.ravel(), per_view.ravel()
rms_k3, _, dist_k3, _, _, std_k3, _, _ = calibrate(img_pts, flags=0)
print(f"RMS {rms:.3f} px; fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}")
print("dist (k1 k2 p1 p2 k3):", np.round(dist[:5], 4), " std:", np.round(std_in[4:9], 4))

# Image poses: consecutive near-copies form one pose.
pose_id = [0]
for a, b in zip(img_pts, img_pts[1:]):
    shift = np.linalg.norm((b - a).reshape(-1, 2), axis=1).mean()
    pose_id.append(pose_id[-1] + (shift >= DUP_PX))
n_poses = pose_id[-1] + 1

view = []
for name, p, r, t, e in zip(names, pose_id, rvecs, tvecs, per_view):
    R, _ = cv2.Rodrigues(r)
    n = R[:, 2] * -np.sign(R[2, 2])      # board normal, pointing to the camera
    view.append({"image": name, "pose": p + 1, "rms_px": e,
                 "tx_mm": t[0, 0], "ty_mm": t[1, 0], "tz_mm": t[2, 0],
                 "distance_mm": np.linalg.norm(t),
                 "tilt_deg": np.degrees(np.arccos(abs(R[2, 2]))),     # normal vs optical axis
                 "turn_lr_deg": np.degrees(np.arctan2(n[0], -n[2])),  # > 0: right edge away
                 "turn_ud_deg": np.degrees(np.arctan2(n[1], -n[2])),  # > 0: bottom edge away
                 "rot_deg": (np.degrees(np.arctan2(R[1, 0], R[0, 0])) + 90) % 180 - 90})
poses = [[v for v in view if v["pose"] == p + 1] for p in range(n_poses)]
pose_text = [describe(vs) for vs in poses]

# ---------------------------------------------------------------- outputs
(DATA / "calibration.json").write_text(json.dumps({
    "image_size": [W, H], "square_mm": square, "n_images": len(names), "k3_fixed_zero": True,
    "rms_reprojection_px": rms, "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
    "std_fx": std_in[0], "std_fy": std_in[1], "std_cx": std_in[2], "std_cy": std_in[3],
    "dist_k1_k2_p1_p2_k3": dist[:5].tolist(), "std_k1_k2_p1_p2_k3": std_in[4:9].tolist()},
    indent=2))
with open(DATA / "per_view.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(view[0]))
    wr.writeheader()
    for v in view:
        wr.writerow({k: (f"{x:.4f}" if isinstance(x, float) else x) for k, x in v.items()})
with open(DATA / "poses_rows.tex", "w") as f:
    for vs, text in zip(poses, pose_text):
        nums = [v["image"][4:6] for v in vs]
        imgs = nums[0] if len(nums) == 1 else f"{nums[0]}--{nums[-1]}"
        f.write(f"{vs[0]['pose']} & {imgs} & {text}\\\\\n")

macros = {"opencvVersion": cv2.__version__,
          "pythonVersion": ".".join(map(str, sys.version_info[:3])),
          "imgWidth": str(W), "imgHeight": str(H),
          "patternCols": str(PATTERN[0]), "patternRows": str(PATTERN[1]),
          "squaresCols": str(PATTERN[0] + 1), "squaresRows": str(PATTERN[1] + 1),
          "nCorners": str(PATTERN[0] * PATTERN[1]), "openCVMin": str(OPENCV_MIN),
          "nImages": str(len(names)), "nPoses": str(n_poses), "dupPx": str(DUP_PX),
          "rms": f"{rms:.2f}", "rmsKThree": f"{rms_k3:.2f}",
          "rmsMin": f"{per_view.min():.2f}", "rmsMax": f"{per_view.max():.2f}",
          "distMin": f"{min(v['distance_mm'] for v in view) / 10:.0f}",
          "distMax": f"{max(v['distance_mm'] for v in view) / 10:.0f}",
          "tiltMax": f"{max(v['tilt_deg'] for v in view):.0f}",
          "ppOffX": f"{K[0, 2] - W / 2:.0f}", "ppOffY": f"{K[1, 2] - H / 2:.0f}"}
val_unc("squareMM", square, square_u)
for name, i in (("focalX", 0), ("focalY", 1), ("ppX", 2), ("ppY", 3)):
    val_unc(name, K.ravel()[[0, 4, 2, 5][i]], std_in[i])
for name, i in (("kOne", 0), ("kTwo", 1), ("pOne", 2), ("pTwo", 3)):
    val_unc(name, dist[i], std_in[4 + i])
val_unc("kThreeFree", dist_k3.ravel()[4], std_k3.ravel()[8])
with open(ROOT / "results.tex", "w") as f:
    f.write("% Generated by code/calibrate.py - do not edit by hand\n")
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})

# Sample images with the detected corners.
fig, axs = plt.subplots(1, len(SAMPLES), figsize=(3.5, 1.05))
for ax, name in zip(axs, SAMPLES):
    v = view[names.index(name)]
    img = cv2.imread(str(DETECTED / name.replace(".png", ".jpg")))
    ax.imshow(img[:, :, ::-1])
    ax.set_title(f"{name[:-4].replace('_', ' ')}: {pose_text[v['pose'] - 1].split(',')[0]}",
                 fontsize=6, pad=2)
    ax.axis("off")
fig.tight_layout(pad=0.1, w_pad=0.3)
fig.savefig(FIG / "samples.pdf", bbox_inches="tight", dpi=200)
plt.close(fig)

# All images (appendix).
cols = 5
fig, axs = plt.subplots(int(np.ceil(len(names) / cols)), cols, figsize=(7.0, 5.6))
for ax in axs.ravel():
    ax.axis("off")
for ax, name in zip(axs.ravel(), names):
    ax.imshow(cv2.imread(str(IMAGES / name))[:, :, ::-1])
    ax.set_title(name[:-4].replace("_", " "), fontsize=7, pad=2)
fig.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.4)
fig.savefig(FIG / "all_images.pdf", bbox_inches="tight", dpi=150)
plt.close(fig)
