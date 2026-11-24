"""Assignment 5: KUKA youBot placing experiment, preprocessing of the class data.

Reads every team's OptiTrack takes (object pose, external) and ROS bags (end-effector pose,
internal), ours renamed by run in data/raw/ and the other teams' as received in
data/class/team_<nn>/, filters them to one final pose per run, and writes the Table 3 exports,
the combined class table, the figures and results.tex.

Usage (from report_hw05/):  python3 code/process_youbot.py
"""
import csv
import datetime as dt
import re
import warnings
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"

SIZES = ["large", "medium", "small"]
MOTIONS = ["left", "straight", "right"]
OUR_TEAM = 4

# Where each team's takes, bags and origin take are, relative to data/: other teams' folders
# as received in class/, ours renamed by run in raw/ (raw/manifest.csv maps the names).
TEAMS = {
    1: dict(takes="class/team_01/notbags", bags="class/team_01/bags",
            origin="class/team_01/origin-small.csv",
            skip_bags="group1_SS26"),   # bags of an earlier semester (Sept. 2026), not this run
    2: dict(takes="class/team_02/Team2_task3/Team_2optitrack",
            bags="class/team_02/Team2_task3/WS26team2",
            origin="class/team_02/Team2_task3/Team_2optitrack/team2_Origin.csv"),
    3: dict(takes="class/team_03/Team_3_CSV_file", bags="class/team_03/Team_3_bagfile",
            origin="class/team_03/Team_3_CSV_file/Origin_Team_3.csv"),
    4: dict(takes="raw/optitrack", bags="raw/encoder", origin="raw/optitrack/team_04_origin.csv"),
    5: dict(takes="class/team_05/Team_5", bags="class/team_05/Team_5/SEE_Team5WS26",
            origin="class/team_05/Team_5/Origin_Team_5.csv"),
}

# Ground truth (manual, Tables 1 and 2), world frame, m and rad.
GT_OBJECT = {"pick": (0.143, -0.351, -1.65), "straight": (0.150, -0.212, -1.65),
             "left": (0.356, -0.283, -2.08), "right": (-0.064, -0.352, -1.13)}
GT_EE = {"pick": (0.152, -0.457, -1.65), "straight": (0.167, -0.317, -1.62),
         "left": (0.331, -0.380, -2.11), "right": (-0.01, -0.433, -1.11)}

FPS = 120                 # OptiTrack frame rate (take header)
WINDOW_S = 1.0            # a pose is the mean of 1 s of frames
REST_S = 0.5             # s, the object is at rest if it stays ...
REST_BOX = 0.02           # m, ... inside a 2 cm box for 0.5 s (tracking jitters up to 1 cm)
MOVE_TOL = 0.001          # m, a take whose object moves less than 1 mm shows no motion
PICK_RADIUS = 0.08        # m, the object counts as in the container within 8 cm of the pick point
REST_TOL = 0.005          # m, and at most 5 mm above its lowest height there
TABLE_TOL = 0.01          # m, a placed object rests within 1 cm of its lowest height
K_CHEB = 2                # Chebyshev outlier test, mu +- 2 sigma (Lecture 3)
BAG_TOL = 0.001           # m, the arm has stopped once the position equals the last reading
FEW_READINGS = 10         # a bag with fewer readings started late


def chebyshev_inliers(v, k=K_CHEB):
    """Single pass of Chebyshev's outlier removal (Lecture 3): keep values within mu +- k sigma."""
    mu, sigma = v.mean(), v.std()
    return (v >= mu - k * sigma) & (v <= mu + k * sigma)


def label(rel):
    """Object size and motion from a path relative to the team folder, or None."""
    parts = [p.lower() for p in Path(rel).parts[:-1]]
    text = " ".join(parts).replace("straigth", "straight").replace("straightmed", "straight medium")
    size = next((s for s in SIZES if s in text), None)
    motion = next((m for m in MOTIONS if m in text), None)
    for p in parts:   # Team 5 folders: ll, lr, ls, ml, ..., sr
        if len(p) == 2 and p[0] in "lms" and p[1] in "lrs":
            size = size or {"l": "large", "m": "medium", "s": "small"}[p[0]]
            motion = motion or {"l": "left", "r": "right", "s": "straight"}[p[1]]
    if motion is None:
        name = Path(rel).name.lower()
        motion = next((m for m in MOTIONS if m in name), None)
    return size, motion


# ---------------------------------------------------------------- OptiTrack takes
def read_take(path):
    """Start time, rigid-body name, time (s), quaternion (x, y, z, w) and position (m)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    meta = dict(zip(rows[0][::2], rows[0][1::2]))
    start = dt.datetime.strptime(meta["Capture Start Time"], "%Y-%m-%d %I.%M.%S.%f %p")
    typ, name = rows[2], rows[3]
    cols = [i for i, t in enumerate(typ) if t == "Rigid Body"]
    if len(cols) != 7:
        return start, None, None, None, None
    d = np.array([[float(r[i]) if i < len(r) and r[i] != "" else np.nan for i in [1] + cols]
                  for r in rows[8:] if r])
    return start, name[cols[0]], d[:, 0], d[:, 1:5], d[:, 5:8]


def read_marker(path):
    """Mean position (m) of the single marker (or rigid body) in an origin take."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    typ, sub = rows[2], rows[6]
    cols = [i for i, (t, s) in enumerate(zip(typ, sub)) if t == "Marker" and s == "Position"]
    d = np.array([[float(r[i]) for i in cols] for r in rows[8:] if r and all(r[i] for i in cols)])
    return d.mean(axis=0)


def qmul(a, b):
    ax, ay, az, aw = a.T
    bx, by, bz, bw = b.T
    return np.column_stack([aw * bx + ax * bw + ay * bz - az * by,
                            aw * by - ax * bz + ay * bw + az * bx,
                            aw * bz + ax * by - ay * bx + az * bw,
                            aw * bw - ax * bx - ay * by - az * bz])


def yaw_from(q, q0):
    """Rotation (rad) about the vertical OptiTrack Y axis from orientation q0 to each q.
    OptiTrack Y points up, world x = OptiTrack X and world y = OptiTrack Z, so a positive
    OptiTrack yaw is a clockwise (negative) world rotation; the sign is flipped here."""
    q0inv = np.array([-q0[0], -q0[1], -q0[2], q0[3]])
    r = qmul(q, np.tile(q0inv, (len(q), 1)))
    x, y, z, w = r.T
    return -np.arctan2(2 * (w * y + x * z), 1 - 2 * (x * x + y * y))


def mean_quat(q):
    q = q * np.sign(q @ q[0])[:, None]
    m = q.mean(axis=0)
    return m / np.linalg.norm(m)


def filtered_mean(v):
    """Mean of each column after removing Chebyshev outliers; also the outlier count."""
    ok = np.column_stack([chebyshev_inliers(v[:, k]) for k in range(v.shape[1])]).all(axis=1)
    return v[ok].mean(axis=0), int((~ok).sum())


def take_windows(path, origin, motion):
    """Frames of one take that show the object at rest in the container (pick) and at its
    place (final), in world coordinates (m) with their orientations.

    Frames where the marker was lost (empty cells) or frozen (exactly the same value as the
    frame before or after: Motive repeats the last pose when it cannot track the object) are
    dropped. The object is at rest where it stays inside a REST_BOX for 0.5 s. The final
    window is the last 1 s of the last rest away from the container at table height (later
    frames, e.g. a person lifting the object off for the next run, are ignored). The pick
    window is the last 1 s (at least 0.5 s) in the container: within PICK_RADIUS of the
    ground-truth pick point and not lifted. A take without it cannot be corrected and is left
    out (status "no pick"), as is a take whose object never moved ("not tracked")."""
    start, body, t, q, pos = read_take(path)
    res = dict(start=start, body=body, frames=0 if t is None else len(t),
               duration=0.0 if t is None else float(t[-1]))
    if t is None:
        return res | dict(status="no rigid body")
    same = np.r_[False, (pos[1:] == pos[:-1]).all(axis=1)]
    frozen = same | np.r_[same[1:], False]
    pos[frozen] = np.nan
    seen = ~np.isnan(pos).any(axis=1)
    res["lost"] = int((~seen).sum())
    res["frozen"] = int(frozen.sum())
    win, box = int(WINDOW_S * FPS), int(REST_S * FPS)
    if seen.sum() < box:
        return res | dict(status="not tracked")
    xy = np.column_stack([pos[:, 0] - origin[0], pos[:, 2] - origin[2]])
    h = pos[:, 1]
    target = np.array(GT_OBJECT[motion][:2])
    if np.nanmax(np.ptp(xy[seen], axis=0)) < MOVE_TOL:      # object never moved
        if np.hypot(*(xy[seen][-1] - target)) > PICK_RADIUS:
            return res | dict(status="not tracked")         # parked away from the robot
        return res | dict(status="no pick")                 # take started after the placing
    # at rest: x, y and height stay inside a REST_BOX for REST_S
    v = np.column_stack([xy, h])
    wv = np.lib.stride_tricks.sliding_window_view(v, box, axis=0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        span = np.nanmax(wv, axis=2) - np.nanmin(wv, axis=2)
    still = np.zeros(len(v), bool)
    still[:len(span)] = (span < REST_BOX).all(axis=1)
    near = seen & (np.hypot(*(xy - GT_OBJECT["pick"][:2]).T) < PICK_RADIUS)
    in_box = near & (h < np.min(h[near]) + REST_TOL) if near.any() else near
    away = still & seen & ~near
    if not away.any():
        return res | dict(status="not placed")
    placed = away & (h < np.min(h[away]) + TABLE_TOL)     # resting on the table
    last = np.where(placed)[0][-1]                 # last rest at the place: frames s to end
    s = last
    while s > 0 and placed[s - 1]:
        s -= 1
    end = last + box
    fin = np.where(seen[s:end])[0][-win:] + s
    res |= dict(fin_xy=xy[fin], fin_q=q[fin])
    ini = np.where(in_box[:fin[0]])[0][-win:]
    if len(ini) < box:
        return res | dict(status="no pick")
    return res | dict(status="ok", ini_xy=xy[ini], ini_q=q[ini])


def object_pose(w):
    """Final object pose (m, m, rad) from the windows of take_windows, after removing the
    Chebyshev outliers of x, y and theta in each window.

    The container holds the object at the ground-truth pick pose. The difference between
    the measured and the ground-truth pick position is the offset of the marker centre from
    the object centre; it turns with the object and is subtracted from the final position.
    theta = ground-truth pick angle + rotation from pick to place."""
    out = {}
    q0 = mean_quat(w["ini_q"])
    v = np.column_stack([w["ini_xy"], yaw_from(w["ini_q"], q0)])
    m0, out["out_ini"] = filtered_mean(v)
    out["n_ini"] = len(v)
    v = np.column_stack([w["fin_xy"], yaw_from(w["fin_q"], q0)])
    m1, out["out_fin"] = filtered_mean(v)
    out["n_fin"] = len(v)
    dth = m1[2] - m0[2]
    off = m0[:2] - GT_OBJECT["pick"][:2]
    c, s = np.cos(dth), np.sin(dth)
    out["x"] = m1[0] - (c * off[0] - s * off[1])
    out["y"] = m1[1] - (s * off[0] + c * off[1])
    out["theta"] = GT_OBJECT["pick"][2] + dth
    out["pick_xy"], out["pick_q"], out["offset"] = m0[:2], q0, off
    out["raw_x"], out["raw_y"] = m1[0], m1[1]
    return out


# ---------------------------------------------------------------- ROS bags
TYPESTORE = get_typestore(Stores.ROS1_NOETIC)


def bag_pose(path):
    """Final end-effector pose (m, m, rad) from /end_effector_pose: the mean of the readings
    at the end of the bag whose position equals the last reading within BAG_TOL, after the
    Chebyshev outliers are removed."""
    rows = []
    with Reader(path) as r:
        start = dt.datetime.fromtimestamp(r.start_time / 1e9)
        for conn, _, raw in r.messages():
            if conn.topic == "/end_effector_pose":
                m = TYPESTORE.deserialize_ros1(raw, conn.msgtype)
                rows.append((m.x, m.y, m.theta))
    a = np.array(rows)
    if len(a) == 0:
        return dict(readings=0, start=start, error="no end-effector readings")
    last = a[-1, :2]
    i = len(a)
    while i > 0 and np.all(np.abs(a[i - 1, :2] - last) <= BAG_TOL + 1e-9):
        i -= 1
    rest = a[i:]
    m, n_out = filtered_mean(rest)
    steps = [np.diff(np.unique(a[:, k])) for k in (0, 2)]          # reading resolution
    res = [d.min() if len(d) else np.inf for d in steps]
    return dict(readings=len(a), static=len(rest), outliers=n_out, final=m, res=res, start=start)


# ---------------------------------------------------------------- all teams
def collect():
    """Object poses (one per take) and end-effector poses (one per bag) of all teams. Runs
    are numbered in time order per team, object and motion."""
    takes, bags = [], []
    for team, c in TEAMS.items():
        origin = read_marker(OUT / c["origin"])
        root = OUT / c["takes"]
        for p in root.rglob("*.csv"):
            if p.parent == root:                 # origin take at the top level
                continue
            size, motion = label(p.relative_to(root))
            w = take_windows(p, origin, motion)
            takes.append(w | dict(team=team, size=size, motion=motion,
                                  file=str(p.relative_to(OUT))))
        root = OUT / c["bags"]
        for p in root.rglob("*.bag"):
            if c.get("skip_bags") and c["skip_bags"] in str(p):
                continue
            size, motion = label(p.relative_to(root))
            bags.append(bag_pose(p) | dict(team=team, size=size, motion=motion,
                                           file=str(p.relative_to(OUT))))
    for rows in (takes, bags):
        rows.sort(key=lambda r: (r["team"], r["size"], r["motion"], r["start"]))
        n = {}
        for r in rows:
            k = (r["team"], r["size"], r["motion"])
            n[k] = r["run"] = n.get(k, 0) + 1
    for w in takes:
        if w["status"] == "ok":
            w |= object_pose(w)
    return takes, bags


def pose_cm_deg(r):
    if r.get("final") is not None:                    # end effector
        x, y, th = r["final"]
    elif "x" in r:                                    # object
        x, y, th = r["x"], r["y"], r["theta"]
    else:
        return None
    return 100 * x, 100 * y, np.degrees(th)


def write_table3(rows, path):
    """One team's poses in the layout of the manual's Table 3 (youBot_table_template.csv):
    three header rows (object, motion, variable) and one row per run, 2 decimals."""
    cols = [(s, m) for s in reversed(SIZES) for m in ("left", "straight", "right")]
    runs = {k: [pose_cm_deg(r) for r in rows if (r["size"], r["motion"]) == k] for k in cols}
    n = max(len(v) for v in runs.values())
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([s.upper() for s, m in cols for _ in range(3)])
        wr.writerow([m.capitalize() for s, m in cols for _ in range(3)])
        wr.writerow(["X Translation (cm)", "Y Translation (cm)", "Orientation (deg)"] * len(cols))
        for i in range(n):
            line = []
            for k in cols:
                p = runs[k][i] if i < len(runs[k]) else None
                line += ["" if p is None else f"{v:.2f}" for v in (p or (0, 0, 0))]
            wr.writerow(line)


def write_combined(rows, path):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["team", "object", "motion", "run", "x_cm", "y_cm", "theta_deg", "used",
                     "source_file"])
        for r in rows:
            p = pose_cm_deg(r)
            vals = ["", "", ""] if p is None else [f"{v:.4f}" for v in p]
            note = "yes" if p is not None else r.get("status", r.get("error"))
            wr.writerow([r["team"], r["size"], r["motion"], r["run"], *vals, note, r["file"]])


def appendix_rows(rows):
    """LaTeX rows per team and object: run, then x, y, theta for left, straight, right."""
    out = {}
    for t in TEAMS:
        for s in SIZES:
            cols = [[pose_cm_deg(r) for r in rows if (r["team"], r["size"], r["motion"]) == (t, s, m)]
                    for m in MOTIONS]
            lines = []
            for i in range(max(len(c) for c in cols)):
                cells = [str(i + 1)]
                for c in cols:
                    p = c[i] if i < len(c) else None
                    cells += ["--"] * 3 if p is None else [f"{v:.2f}".replace("-", "$-$")
                                                            for v in p]
                lines.append(" & ".join(cells) + r"\\")
            out[(t, s)] = "\n".join(lines) + "\n"
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    takes, bags = collect()
    used = [w for w in takes if "x" in w]
    bag_ok = [b for b in bags if b.get("final") is not None]
    export = OUT / "export"
    export.mkdir(exist_ok=True)
    write_table3([w for w in takes if w["team"] == OUR_TEAM], export / "team_04_object_poses.csv")
    write_table3([b for b in bags if b["team"] == OUR_TEAM],
                 export / "team_04_end_effector_poses.csv")
    write_combined(takes, OUT / "combined_object.csv")
    write_combined(bags, OUT / "combined_end_effector.csv")
    tab = OUT / "tables"
    tab.mkdir(exist_ok=True)
    for name, rows in (("object", takes), ("ee", bags)):
        for (t, s), text in appendix_rows(rows).items():
            (tab / f"{name}_team{t}_{s}.tex").write_text(text)

    # ------------------------------------------------------------ numbers for the text
    M = {}

    def put(name, value, fmt="{}"):
        M[name] = fmt.format(value)

    full = [w for w in takes if w["status"] == "ok"]
    put("nTeams", len(TEAMS))
    put("nRunsPlanned", len(SIZES) * len(MOTIONS) * 25)
    put("nRunsCombo", 25)
    put("nTakes", len(takes))
    put("nTakesOurs", sum(w["team"] == OUR_TEAM for w in takes))
    put("nObjUsed", len(used))
    put("nNoPick", sum(w["status"] == "no pick" for w in takes))
    put("nNotTracked", sum(w["status"] == "not tracked" for w in takes))
    put("nBags", len(bags))
    put("nBagUsed", len(bag_ok))
    put("nFrozenTakes", sum(w.get("frozen", 0) > 0 for w in takes))
    put("frozenMean", np.mean([w["frozen"] for w in takes if w.get("frozen", 0) > 0]), "{:.0f}")
    put("fps", FPS)
    put("windowS", WINDOW_S, "{:g}")
    put("restS", REST_S, "{:g}")
    put("restBoxCm", REST_BOX * 100, "{:g}")
    put("pickRadiusCm", PICK_RADIUS * 100, "{:g}")
    put("restTolMm", REST_TOL * 1000, "{:g}")
    put("tableTolCm", TABLE_TOL * 100, "{:g}")
    put("kCheb", K_CHEB)
    put("frameRotDeg", 90)          # OptiTrack -> robot frame: -90 deg about X (yaw_from)
    put("winFrames", int(WINDOW_S * FPS))
    out_take = np.array([w["out_ini"] + w["out_fin"] for w in full])
    n_take = np.array([w["n_ini"] + w["n_fin"] for w in full])
    put("outPerTake", out_take.mean(), "{:.0f}")
    put("outPct", 100 * out_take.sum() / n_take.sum(), "{:.0f}")
    put("outIni", np.mean([w["out_ini"] for w in full]), "{:.0f}")
    put("outFin", np.mean([w["out_fin"] for w in used]), "{:.0f}")
    put("bagReadings", np.mean([b["readings"] for b in bag_ok]), "{:.0f}")
    put("bagStatic", np.mean([b["static"] for b in bag_ok]), "{:.0f}")
    put("bagOut", np.mean([b["outliers"] for b in bag_ok]), "{:.1f}")
    put("bagFew", sum(b["readings"] < FEW_READINGS for b in bag_ok))
    put("fewReadings", FEW_READINGS)
    put("bagTolMm", BAG_TOL * 1000, "{:g}")
    put("eeResMm", 1000 * min(b["res"][0] for b in bag_ok), "{:.0f}")
    put("eeResRad", min(b["res"][1] for b in bag_ok), "{:.2f}")
    off = np.array([np.hypot(*w["offset"]) for w in full]) * 100
    put("offMin", off.min(), "{:.1f}")
    put("offMax", off.max(), "{:.1f}")
    sess = {"A": (1, 3), "B": (2, 4, 5)}          # lab days 28-29 Oct and 2-3 Nov
    gap = 0.0
    for m in MOTIONS:
        mean = {k: np.mean([[w["x"], w["y"]] for w in used
                            if w["motion"] == m and w["team"] in v], axis=0) for k, v in sess.items()}
        gap = max(gap, np.hypot(*(mean["A"] - mean["B"])))
    put("sessionGapCm", 100 * gap, "{:.1f}")
    ee_xy = {m: np.array([b["final"][:2] for b in bag_ok if b["motion"] == m]) for m in MOTIONS}
    put("eeSpreadMm", 1000 * max(np.ptp(v, axis=0).max() for v in ee_xy.values()), "{:.0f}")
    for t in TEAMS:
        key = "ABCDE"[t - 1]
        put(f"takes{key}", sum(w["team"] == t for w in takes))
        put(f"objUsed{key}", sum(w["team"] == t for w in used))
        put(f"bags{key}", sum(b["team"] == t for b in bags))
        put(f"bagUsed{key}", sum(b["team"] == t for b in bag_ok))

    with open(ROOT / "results.tex", "w") as f:
        f.write("% Generated by code/process_youbot.py - do not edit by hand\n")
        for k, v in M.items():
            assert re.fullmatch(r"[A-Za-z]+", k), k
            v = v.replace("-", "\\ensuremath{-}", 1) if v.startswith("-") else v
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # ------------------------------------------------------------ figures
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "pdf.fonttype": 42, "axes.grid": True,
                         "grid.alpha": 0.4})
    tab10 = plt.get_cmap("tab10").colors
    color = {t: tab10[(t - 1) % 10] for t in TEAMS}
    marker = {"large": "o", "medium": "s", "small": "^"}
    draw = [t for t in TEAMS if t != OUR_TEAM] + [OUR_TEAM]
    title = {"left": "Left", "straight": "Straight", "right": "Right"}

    def team_label(t):
        return f"Team {t}" + (" (ours)" if t == OUR_TEAM else "")

    # a) overview of the workspace: pick pose, ground truth and the mean final poses
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    arrow = 4.0
    for gt, mk, c, lab in ((GT_OBJECT, "o", "k", "object"), (GT_EE, "x", "0.4", "end effector")):
        for k, (x, y, th) in gt.items():
            ax.plot(100 * x, 100 * y, mk, mfc="none", color=c, ms=8)
    for m in MOTIONS:
        o = np.array([pose_cm_deg(w) for w in used if w["motion"] == m]).mean(axis=0)
        e = np.array([pose_cm_deg(b) for b in bag_ok if b["motion"] == m]).mean(axis=0)
        for p, c, mk in ((o, "tab:blue", "o"), (e, "tab:orange", "x")):
            ax.plot(p[0], p[1], mk, color=c, ms=5)
            ax.annotate("", xy=(p[0] + arrow * np.cos(np.radians(p[2])),
                                p[1] + arrow * np.sin(np.radians(p[2]))), xytext=p[:2],
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.0))
        ax.text(100 * GT_OBJECT[m][0], 100 * GT_OBJECT[m][1] + 2.4, title[m], ha="center",
                fontsize=8)
    ax.text(100 * GT_OBJECT["pick"][0], 100 * GT_OBJECT["pick"][1] - 3.9, "Pick", ha="center",
            fontsize=8)
    h = [ax.plot([], [], "o", mfc="none", color="k", label="Ground truth, object")[0],
         ax.plot([], [], "x", color="0.4", label="Ground truth, end effector")[0],
         ax.plot([], [], "o-", color="tab:blue", ms=5, label="Object, OptiTrack (mean)")[0],
         ax.plot([], [], "x-", color="tab:orange", ms=5, label="End effector, encoders (mean)")[0]]
    ax.legend(handles=h, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              frameon=False, columnspacing=0.8, handletextpad=0.4)
    ax.set_aspect("equal")
    ax.set_xlim(-12, 42)
    ax.set_ylim(-50, -16)
    ax.set_xlabel("$x$ [cm]")
    ax.set_ylabel("$y$ [cm]")
    fig.tight_layout()
    fig.savefig(fig_dir / "overview.pdf", bbox_inches="tight")
    plt.close(fig)

    def grid_2x2(height):
        """Three motion panels in a 2x2 grid; the fourth cell holds the legend."""
        fig, axs = plt.subplots(2, 2, figsize=(3.5, height))
        axs = axs.ravel()
        axs[3].axis("off")
        return fig, axs

    # b) final object positions per motion, zoomed, coloured by team, shape by object
    half = 1.5
    fig, axs = grid_2x2(3.3)
    for ax, m in zip(axs, MOTIONS):
        sel = [w for w in used if w["motion"] == m]
        c0 = np.array([[w["x"], w["y"]] for w in sel]).mean(axis=0) * 100
        for t in draw:
            for s in SIZES:
                p = np.array([[w["x"], w["y"]] for w in sel if (w["team"], w["size"]) == (t, s)])
                if len(p):
                    ax.plot(100 * p[:, 0], 100 * p[:, 1], marker[s], ms=3, mfc="none",
                            mew=0.6, color=color[t])
        ax.plot(100 * GT_OBJECT[m][0], 100 * GT_OBJECT[m][1], "k+", ms=9, mew=1.3)
        ax.set_aspect("equal")
        ax.set_xlim(c0[0] - half, c0[0] + half)
        ax.set_ylim(c0[1] - half, c0[1] + half)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_title(title[m])
        ax.set_xlabel("$x$ [cm]")
        ax.set_ylabel("$y$ [cm]")
    h = [axs[3].plot([], [], "s", ms=5, color=color[t], label=team_label(t))[0] for t in TEAMS]
    h += [axs[3].plot([], [], marker[s], ms=5, mfc="none", color="k",
                      label=s.capitalize())[0] for s in SIZES]
    h += [axs[3].plot([], [], "k+", ms=8, mew=1.3, label="Ground truth")[0]]
    axs[3].legend(handles=h, fontsize=7.5, loc="center", ncol=2, frameon=False,
                  columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    fig.savefig(fig_dir / "object_xy.pdf", bbox_inches="tight")
    plt.close(fig)

    # c) final orientation per motion: object (OptiTrack) per size and end effector (encoders)
    fig, axs = grid_2x2(3.4)
    rng = np.random.default_rng(0)
    row = {"large": 3, "medium": 2, "small": 1}
    for i, (ax, m) in enumerate(zip(axs, MOTIONS)):
        for t in draw:
            for s in SIZES:
                th = [pose_cm_deg(w)[2] for w in used
                      if (w["team"], w["size"], w["motion"]) == (t, s, m)]
                ax.plot(th, row[s] + rng.uniform(-0.25, 0.25, len(th)), ".", ms=2.5,
                        color=color[t])
            th = [pose_cm_deg(b)[2] for b in bag_ok if (b["team"], b["motion"]) == (t, m)]
            ax.plot(th, rng.uniform(-0.25, 0.25, len(th)), ".", ms=2.5, color=color[t])
        ax.axvline(np.degrees(GT_OBJECT[m][2]), color="k", lw=0.9, ls="--")
        ax.axvline(np.degrees(GT_EE[m][2]), color="0.4", lw=0.9, ls=":")
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax.set_title(title[m])
        ax.set_xlabel(r"$\theta$ [deg]")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, 3.6)
        ax.set_yticks([3, 2, 1, 0])
        ax.set_yticklabels(["Large", "Medium", "Small", "End eff."] if i % 2 == 0 else [])
    h = [axs[3].plot([], [], "s", ms=5, color=color[t], label=team_label(t))[0] for t in TEAMS]
    h += [axs[3].plot([], [], "k--", lw=0.9, label="GT object")[0],
          axs[3].plot([], [], ":", color="0.4", lw=0.9, label="GT end effector")[0]]
    axs[3].legend(handles=h, fontsize=7.5, loc="center", frameon=False, handletextpad=0.4)
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    fig.savefig(fig_dir / "theta.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"takes {len(takes)}, object poses {len(used)}, bags {len(bags)}, "
          f"end-effector poses {len(bag_ok)}")
    for w in takes:
        if w["status"] != "ok":
            print(f"  {w['status']:12s} team {w['team']} {w['size']} {w['motion']} run {w['run']}: "
                  f"{w['file']}")


if __name__ == "__main__":
    main()
