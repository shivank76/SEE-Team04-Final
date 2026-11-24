"""Capture still images of the checkerboard for the camera calibration.

Opens the webcam at its highest resolution, tries to switch autofocus off, and
shows a live preview with the detected board (green = found, red = not found)
and the corners of all saved images, to check the coverage of the image area.

Keys: SPACE save frame, Q quit.
Usage: python3 code/capture.py [--camera N]    or    python3 code/capture.py --list
       python3 code/capture.py --test --camera N   (camera check, no preview)
Saves data/images/img_<nn>.png (lossless, full resolution) and
data/images/capture_log.csv.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

PATTERN = (9, 6)  # inner corners (columns, rows)
RESOLUTIONS = [(4096, 2160), (3840, 2160), (2592, 1944), (2560, 1440),
               (1920, 1080), (1600, 1200), (1280, 960), (1280, 720), (640, 480)]
PREVIEW_W = 1000

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "images"


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} not available")
    return cap


def set_max_resolution(cap):
    """Try the candidate resolutions, keep the largest one the camera accepts."""
    best = None
    for w, h in RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        ok, frame = cap.read()
        if ok and (best is None or frame.shape[0] * frame.shape[1] > best[0] * best[1]):
            best = (frame.shape[1], frame.shape[0])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, best[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best[1])
    return best


def list_cameras():
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ok, frame = cap.read()
            print(i, frame.shape[1::-1] if ok else "no frame")
        cap.release()


def test_camera(index):
    """Camera check without preview: resolution, autofocus, frame rate, auto
    settings. Writes data/camera_test_<index>.json and a snapshot .jpg."""
    cap = open_camera(index)
    ok = False
    for _ in range(50):  # the first frames can be empty while the camera starts
        ok, frame = cap.read()
        if ok:
            break
        time.sleep(0.1)
    if not ok:
        raise SystemExit(f"camera {index}: no frame")
    w, h = set_max_resolution(cap)
    res = {"camera": index, "width": w, "height": h,
           "autofocus_off_accepted": bool(cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)),
           "autofocus_readback": cap.get(cv2.CAP_PROP_AUTOFOCUS),
           "fps_reported": cap.get(cv2.CAP_PROP_FPS)}
    for name in ["AUTO_EXPOSURE", "EXPOSURE", "AUTO_WB", "FOCUS", "GAIN", "BRIGHTNESS"]:
        res[name.lower()] = cap.get(getattr(cv2, "CAP_PROP_" + name))
    t0, n = time.time(), 0
    while time.time() - t0 < 3:
        ok, frame = cap.read()
        n += ok
    res["fps_measured"] = round(n / (time.time() - t0), 1)
    res["mean_brightness"] = round(float(frame.mean()), 1)
    cap.release()

    out = ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"camera_test_{index}.jpg"), frame)
    (out / f"camera_test_{index}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.list:
        list_cameras()
        return
    if args.test:
        test_camera(args.camera)
        return

    cap = open_camera(args.camera)
    w, h = set_max_resolution(cap)
    af_set = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    af_now = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    print(f"resolution {w} x {h}")
    print(f"autofocus off accepted by driver: {af_set} (read back {af_now})")
    if not af_set:
        print("WARNING: autofocus could not be switched off from OpenCV; "
              "lock it with the camera's own settings tool before capturing.")

    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "capture_log.csv"
    new_log = not log.exists()
    n = len(list(OUT.glob("img_*.png")))
    saved_corners = []
    scale = PREVIEW_W / w

    with open(log, "a", newline="") as f:
        wr = csv.writer(f)
        if new_log:
            wr.writerow(["file", "time", "width", "height", "board_found",
                         "autofocus_off_accepted", "autofocus_readback"])
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            small = cv2.resize(frame, None, fx=scale, fy=scale)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, PATTERN, flags=cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH)

            view = small.copy()
            for c in saved_corners:
                for x, y in c:
                    cv2.circle(view, (int(x), int(y)), 2, (255, 160, 0), -1)
            if found:
                cv2.drawChessboardCorners(view, PATTERN, corners, found)
            cv2.putText(view, f"{'FOUND' if found else 'no board'}  saved: {n}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 200, 0) if found else (0, 0, 255), 2)
            cv2.imshow("capture (SPACE save, Q quit)", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                n += 1
                name = f"img_{n:02d}.png"
                cv2.imwrite(str(OUT / name), frame)
                wr.writerow([name, time.strftime("%Y-%m-%d %H:%M:%S"), frame.shape[1],
                             frame.shape[0], found, af_set, af_now])
                f.flush()
                if found:
                    saved_corners.append(corners.reshape(-1, 2))
                print(f"saved {name}  board found: {found}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
