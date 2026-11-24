"""SEE Assignment 2 - end poses from the pencil marks, encoder paths, combined class data.

Inputs (relative to the report folder):
  data/raw/marks.csv           our raw readings in cm: run, brick_run, motion, xL, yL, xR, yR,
                               note  (L, R = pencil marks; empty rows are skipped)
  data/encoder/runs/run_<nnn>_<Motion>_both_motors_path.csv
                               written by EV3_Control_runs.py, copied once (sh code/fetch_all.sh);
                               the k-th file of a motion belongs to run ID <M>k (e.g. L07) unless
                               the column brick_run in marks.csv names another file number
  data/encoder/<run>_both_motors_path.csv
                               alternative: copied by hand after every run (sh code/fetch_run.sh)
  data/class/team_<nn>/*_poses_<left|straight|right>.csv
                               other teams' end poses as received (class format: header
                               x,y,theta; cm, cm, deg), see data/class/README.md
  data/class/<group>.csv       alternative: other groups' end poses in the Table 3 format
  data/class/team_<nn>/*_poses_<motion>.xlsx
                               the same as an Excel file (first sheet, header row)
  data/class/<group>/**/*robot_path*.csv  or  *both_motors_path*.csv
                               other groups' encoder paths in any subfolder (optional); the
                               motion comes from the folder or file name, the run number from
                               the file name (or the order of the time stamps in it)
  data/class/<group>/*encoder*/*_poses_<motion>*.csv
                               encoder end poses of a group that shared no paths (class format)

Outputs:
  data/end_poses.csv           our axle poses per run, with the pencil spacing check
  data/export/Team_<TEAM>_Poses_<2decimals|sigfigs>.zip, Team_<TEAM>_Encoder.zip
                               our end poses in the class format and our encoder files - to share
  data/export/table3_<GROUP>.csv
                               our end poses in the Table 3 format, 2 decimals (manual)
  data/paths/<run>.csv         our encoder paths recomputed from the motor angles
  data/combined_manual.csv, data/combined_encoder.csv
                               manual and encoder end poses of all teams (team, motion, run,
                               x, y, theta), the input of the Assignment 3 statistics
  figures/*.pdf, results.tex   figures and every number quoted in the report

Run:  python code/process_runs.py      (from the report folder)
No randomness is used.
"""
import csv
import itertools
import re
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG = ROOT / "data", ROOT / "figures"
(DATA / "paths").mkdir(parents=True, exist_ok=True)
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- parameters
GROUP = "Sharma"            # our group label in combined plots and file names
TEAM = 4                    # our number in the class numbering (teams 1-5)
WHEEL_DIAMETER = 5.6        # cm, as in EV3_Control.py
MAIN_AXIS_LENGTH = 12.0     # cm, as in EV3_Control.py
W_PEN = 8.1                 # cm, distance between the pencil tips (measured for Assignment 2)
B_OFF = 3.8                 # cm, pencil line behind the axle (measured for Assignment 2)
DELTA = 0.15                # cm, max. error per read coordinate (Assignment 1)
MOTIONS = ("Left", "Straight", "Right")
COLORS = {"Left": "tab:blue", "Straight": "tab:green", "Right": "tab:red"}


# ---------------------------------------------------------------- pose from the marks
def orientation(p):
    """theta from the pencil coordinates p = (xl, yl, xr, yr)."""
    xl, yl, xr, yr = p
    return np.arctan2(xr - xl, yl - yr)


def position(p):
    """Axle midpoint (x, y) from the pencil coordinates p = (xl, yl, xr, yr)."""
    th = orientation(p)
    return np.array([(p[0] + p[2]) / 2 + B_OFF * np.cos(th),
                     (p[1] + p[3]) / 2 + B_OFF * np.sin(th)])


def spacing_offset(p):
    """Deviation of the read pencil spacing |LR| from the known spacing W_PEN."""
    return abs(np.hypot(p[2] - p[0], p[3] - p[1]) - W_PEN)


def spacing_tolerance(p):
    """Largest spacing deviation that reading errors alone can cause (3^4 combos),
    evaluated around marks exactly W_PEN apart at the orientation given by p."""
    th = orientation(p)
    hx, hy = W_PEN / 2 * np.sin(th), W_PEN / 2 * np.cos(th)   # half of R - L
    p0 = np.array([-hx, hy, hx, -hy])
    return max(spacing_offset(p0 + np.array(e))
               for e in itertools.product((-DELTA, 0.0, DELTA), repeat=4))


def pose_error(p):
    """Largest change of (x, y, theta in deg) that reading errors of up to DELTA in the
    four pencil coordinates can cause (3^4 combinations), for the marks p."""
    x0, th0 = position(p), orientation(p)
    d = [np.r_[position(p + np.array(e)) - x0,
               np.degrees(orientation(p + np.array(e)) - th0)]
         for e in itertools.product((-DELTA, 0.0, DELTA), repeat=4)]
    return np.abs(d).max(axis=0)


def taylor_decimals(u):
    """Decimal places of a value with uncertainty u (Lecture 1, Taylor): u keeps one
    significant figure, two if its leading digit is 1; the value ends at the same place."""
    e = int(np.floor(np.log10(u)))
    mant = u / 10**e                            # 1 <= mant < 10
    if mant >= 9.5:                             # e.g. 0.96 -> 1.0, leading digit 1
        return max(0, -e)
    return max(0, -e + (1 if mant < 2 else 0))


# ---------------------------------------------------------------- encoder paths
def odometry(m1, m2):
    """Path (x, y, theta) from the motor angles in rad, with the update rule of
    EV3_Control.py. Its first sample is the raw encoder value in degrees; it is
    converted here, so the path starts at the origin without a jump."""
    m1, m2 = np.array(m1, float), np.array(m2, float)
    m1[0], m2[0] = np.radians(m1[0]), np.radians(m2[0])
    d1 = WHEEL_DIAMETER / 2 * np.diff(m1)               # left wheel, cm
    d2 = WHEEL_DIAMETER / 2 * np.diff(m2)               # right wheel, cm
    phi = np.cumsum((d1 - d2) / MAIN_AXIS_LENGTH)       # clockwise from +Y
    ds = (d1 + d2) / 2
    x = np.concatenate([[0], np.cumsum(ds * np.sin(phi))])
    y = np.concatenate([[0], np.cumsum(ds * np.cos(phi))])
    return x, y, np.pi / 2 - np.concatenate([[0], phi])


def read_path(f):
    """Encoder path of one run from a copied brick file."""
    a = np.loadtxt(f, ndmin=2)
    if f.name.endswith("both_motors_path.csv"):
        return odometry(a[:, 0], a[:, 1])
    return a[:, 0], a[:, 1], a[:, 2]                    # robot_path.csv as written


def motion_of(run):
    return {"L": "Left", "S": "Straight", "R": "Right"}[run[0].upper()]


# ---------------------------------------------------------------- our runs
poses, brick_run, run_ids = [], {}, []
with open(DATA / "raw" / "marks.csv", newline="") as f:
    for row in csv.DictReader(f):
        run_ids.append(row["run"])
        if (row.get("brick_run") or "").strip():
            brick_run[row["run"]] = int(row["brick_run"])
        vals = [row[k].strip() for k in ("xL", "yL", "xR", "yR")]
        if not all(vals):
            continue
        p = np.array(vals, float)
        x, y = position(p)
        off, tol = spacing_offset(p), spacing_tolerance(p)
        poses.append(dict(run=row["run"], motion=row["motion"], x=x, y=y,
                          theta=np.degrees(orientation(p)), err=pose_error(p),
                          spacing_mm=off * 10,
                          tol_mm=tol * 10, ok=off <= tol, note=row.get("note") or ""))

with open(DATA / "end_poses.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "motion", "x_cm", "y_cm", "theta_deg", "spacing_offset_mm",
                "spacing_tolerance_mm", "spacing_check", "note"])
    for r in poses:
        w.writerow([r["run"], r["motion"], f"{r['x']:.2f}", f"{r['y']:.2f}",
                    f"{r['theta']:.2f}", f"{r['spacing_mm']:.2f}", f"{r['tol_mm']:.2f}",
                    "ok" if r["ok"] else "FAILED", r["note"]])


def write_table3(path, by_motion):
    """End poses in the common format of the manual (Table 3), 2 decimals."""
    n = max((len(v) for v in by_motion.values()), default=0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([m.upper() for m in MOTIONS for _ in range(3)])
        w.writerow(["X Translation (cm)", "Y Translation (cm)", "Orientation (deg)"] * 3)
        for i in range(n):
            w.writerow([f"{v:.2f}" if i < len(by_motion[m]) else ""
                        for m in MOTIONS for v in (by_motion[m][i] if i < len(by_motion[m])
                                                   else (0, 0, 0))])


def read_table3(path):
    """End poses per motion from a file in the common format (Table 3)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    out = {m: [] for m in MOTIONS}
    for row in rows[2:]:
        for j, m in enumerate(MOTIONS):
            cell = row[3 * j:3 * j + 3]
            if len(cell) == 3 and all(c.strip() for c in cell):
                out[m].append(tuple(float(c) for c in cell))
    return out


ours = {m: [(r["x"], r["y"], r["theta"]) for r in poses if r["motion"] == m]
        for m in MOTIONS}
EXPORT = DATA / "export"
EXPORT.mkdir(exist_ok=True)
write_table3(EXPORT / f"table3_{GROUP}.csv", ours)

# class export, two zips with the same file names: 2 decimals as the manual asks (step 7),
# and rounded to the resolution the reading errors justify (Lecture 1, Taylor; QA 1.2.12)
u_pos = max(max(r["err"][0], r["err"][1]) for r in poses)            # cm
u_theta = max(r["err"][2] for r in poses)                            # deg
dec_pos, dec_theta = taylor_decimals(u_pos), taylor_decimals(u_theta)


def poses_csv(m, decimals):
    """One motion in the class format: header x,y,theta, one row per run."""
    lines = ["x,y,theta"] + [",".join(f"{v:.{d}f}" for v, d in zip(pose, decimals))
                             for pose in ours[m]]
    return "\n".join(lines) + "\n"


for tag, decimals in (("2decimals", (2, 2, 2)),
                      ("sigfigs", (dec_pos, dec_pos, dec_theta))):
    with zipfile.ZipFile(EXPORT / f"Team_{TEAM:02d}_Poses_{tag}.zip", "w",
                         zipfile.ZIP_DEFLATED) as z:
        files = {f"team_{TEAM:02d}_poses_{m.lower()}.csv": poses_csv(m, decimals)
                 for m in MOTIONS}
        for name, text in files.items():               # fixed date: same zip every run
            z.writestr(zipfile.ZipInfo(name, (2026, 10, 16, 0, 0, 0)), text,
                       zipfile.ZIP_DEFLATED)

# numbered files of EV3_Control_runs.py -> run IDs
numbered = {}
for f in sorted((DATA / "encoder" / "runs").glob("run_*_both_motors_path.csv")):
    _, n, m = f.name.split("_")[:3]
    numbered[int(n)] = (m, f)
by_motion = {m: [n for n in sorted(numbered) if numbered[n][0] == m] for m in MOTIONS}
taken = set(brick_run.values())
encoder_files = {run: numbered[n][1] for run, n in brick_run.items() if n in numbered}
unmatched = []
marked = [r["run"] for r in poses] or run_ids
for m in MOTIONS:                       # remaining files fill the free IDs of their motion
    free_ids = [i for i in marked if i[0] == m[0] and i not in encoder_files]
    free_files = [n for n in by_motion[m] if n not in taken]
    if free_ids and len(free_files) != len(free_ids):
        print(f"  WARNING {m}: {len(free_files)} brick files but {len(free_ids)} run IDs "
              f"without brick_run - fill in brick_run in marks.csv")
    for run, n in zip(free_ids, free_files):
        encoder_files[run] = numbered[n][1]
    unmatched += [numbered[n][1].name for n in free_files[len(free_ids):]]
for f in sorted((DATA / "encoder").glob("*_both_motors_path.csv")):
    encoder_files.setdefault(f.name.split("_")[0], f)

our_paths = {}
for run, f in sorted(encoder_files.items()):
    x, y, th = read_path(f)
    our_paths[(motion_of(run), int(run[1:]))] = (x, y, th)
    np.savetxt(DATA / "paths" / f"{run}.csv", np.column_stack([x, y, np.degrees(th)]),
               delimiter=",", fmt="%.2f", header="x_cm,y_cm,theta_deg", comments="")

# encoder export for the class: the brick files unchanged, named by our run ID, so that L07
# belongs to row 7 of team_<TEAM>_poses_left.csv
with zipfile.ZipFile(EXPORT / f"Team_{TEAM:02d}_Encoder.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for run, f in sorted(encoder_files.items()):
        for src in (f, f.with_name(f.name.replace("both_motors_path", "robot_path"))):
            if src.exists():
                name = f"team_{TEAM:02d}_encoder/{run}_{src.name.split('_', 3)[-1]}"
                z.writestr(zipfile.ZipInfo(name, (2026, 10, 16, 0, 0, 0)), src.read_bytes(),
                           zipfile.ZIP_DEFLATED)

# ---------------------------------------------------------------- combined class data
# manual end poses of a team that are not in the agreed frame stay out of the combined data
# (their encoder paths are still used); empty: every team's data are used as received
EXCLUDE_POSES = {}
TIME_STAMP = re.compile(r"\d\d-\d\d-\d\d")


def read_xlsx(path):
    """Numbers of the first sheet of an .xlsx file, header row skipped (standard library
    only: the file is a zip of XML)."""
    z = zipfile.ZipFile(path)
    rows = re.findall(r"<row[^>]*>(.*?)</row>", z.read("xl/worksheets/sheet1.xml").decode())
    return [tuple(float(v) for a, v in re.findall(r"<c([^>]*)>(?:<f>.*?</f>)?<v>(.*?)</v></c>",
                                                  row) if 't="s"' not in a)
            for row in rows[1:]]


def class_paths(d):
    """Encoder paths of another team, searched in all subfolders, keyed by (motion, run).
    The brick's robot_path.csv is preferred: it uses that team's own axle length
    (MAIN_AXIS_LENGTH may differ from ours). Files are paired by their name without the file
    kind. The motion comes from the folder or file name, the run number is the last number
    in the file name, or the rank of the time stamp if the files are named by time (Team 1).
    Byte-identical files of one motion are copies, not runs: all copies are left out and
    returned in the second value as (motion, run numbers)."""
    files = {}
    for f in sorted(d.rglob("*.csv")):
        for kind in ("robot_path", "both_motors_path"):
            if kind in f.name:
                key = str(f.relative_to(d)).replace(kind, "")
                if key not in files or kind == "robot_path":
                    files[key] = f
                break
    paths, copies = {}, []
    for m in MOTIONS:
        fs = [f for f in files.values() if m.lower() in str(f.relative_to(d)).lower()]
        if fs and all(TIME_STAMP.search(f.name) for f in fs):
            numbered = list(enumerate(sorted(fs, key=lambda f: TIME_STAMP.search(f.name)[0]),
                                      1))
        else:
            numbered = sorted((int(re.findall(r"\d+", f.stem)[-1]), f) for f in fs)
        same = {}
        for n, f in numbered:
            same.setdefault(f.read_bytes(), []).append(n)
        copies += [(m, ns) for ns in same.values() if len(ns) > 1]
        paths.update({(m, n): read_path(f) for n, f in numbered
                      if len(same[f.read_bytes()]) == 1})
    return paths, copies


def class_encoder_poses(d):
    """Encoder end poses that a team shared as a table instead of paths (class format, in a
    subfolder whose name contains 'encoder'), per motion."""
    out = {}
    for m in MOTIONS:
        for f in d.glob(f"*encoder*/*_poses_{m.lower()}*.csv"):
            out[m] = [tuple(r) for r in np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)]
    return out


groups = {GROUP: dict(poses=ours, paths=our_paths, copies=[], enc_table={},
                     runs={m: [int(r["run"][1:]) for r in poses if r["motion"] == m]
                           for m in MOTIONS})}
for f in sorted((DATA / "class").glob("*.csv")):
    groups.setdefault(f.stem, dict(poses=read_table3(f), paths={}, copies=[], enc_table={}))
for d in sorted(p for p in (DATA / "class").iterdir() if p.is_dir()):
    g = groups.setdefault(d.name, dict(poses={m: [] for m in MOTIONS}, paths={}))
    g["paths"], g["copies"] = class_paths(d)
    g["enc_table"] = class_encoder_poses(d)
    if d.name in EXCLUDE_POSES:
        print(f"  {d.name}: manual poses left out - {EXCLUDE_POSES[d.name]}")
        continue
    for m in MOTIONS:                                  # class format: x,y,theta per motion
        for f in d.glob(f"*_poses_{m.lower()}.csv"):
            g["poses"][m] = [tuple(r) for r in
                             np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)]
        for f in d.glob(f"*_poses_{m.lower()}*.xlsx"):  # any suffix after the motion
            g["poses"][m] = read_xlsx(f)
with_poses = [g for g in groups if any(groups[g]["poses"][m] for m in MOTIONS)]
for g in groups.values():                   # other teams: row i of a pose file is run i
    g.setdefault("runs", {m: list(range(1, len(g["poses"][m]) + 1)) for m in MOTIONS})


def team_no(gname):
    """Class number of a group: ours is TEAM, team_<nn> is nn, other names keep their name."""
    if gname == GROUP:
        return TEAM
    tail = gname.split("_")[-1]
    return int(tail) if gname.startswith("team_") and tail.isdigit() else gname


# all end poses in one long table each, input of the Assignment 3 statistics. Encoder end
# poses carry the run number only if they can be paired with a manual run.
with open(DATA / "combined_manual.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["team", "motion", "run", "x_cm", "y_cm", "theta_deg"])
    for gname, g in groups.items():
        for m in MOTIONS:
            for n, pose in zip(g["runs"][m], g["poses"][m]):
                w.writerow([team_no(gname), m, n] + [f"{v:.6f}" for v in pose])
with open(DATA / "combined_encoder.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["team", "motion", "run", "x_cm", "y_cm", "theta_deg"])
    for gname, g in groups.items():
        for (m, n), (x, y, th) in sorted(g["paths"].items(), key=lambda kv:
                                         (MOTIONS.index(kv[0][0]), kv[0][1])):
            w.writerow([team_no(gname), m, n, f"{x[-1]:.6f}", f"{y[-1]:.6f}",
                        f"{np.degrees(th[-1]):.6f}"])
        for m, rows in g["enc_table"].items():
            pair = len(rows) == len(g["poses"][m])          # one row per manual run
            for i, pose in enumerate(rows, 1):
                w.writerow([team_no(gname), m, i if pair else ""] +
                           [f"{v:.6f}" for v in pose])

# ---------------------------------------------------------------- statistics
macros = {"team": str(TEAM), "uPosCm": f"{u_pos:.2f}", "uThetaDeg": f"{u_theta:.1f}",
          "uPosRound": f"{u_pos:.{dec_pos}f}", "uThetaRound": f"{u_theta:.{dec_theta}f}",
          "stepPosCm": f"{10**-dec_pos:.{dec_pos}f}",
          "stepThetaDeg": f"{10**-dec_theta:.{dec_theta}f}",
          "nGroups": str(len(with_poses)),
          "groupsText": "one group" if len(with_poses) == 1 else f"{len(with_poses)} groups",
          "nPathGroups": str(sum(bool(g["paths"]) for g in groups.values())),
          "nSpacingFailed": str(sum(not r["ok"] for r in poses))}
for m in MOTIONS:
    own = np.array(ours[m]).reshape(-1, 3)
    comb = np.array([p for g in groups.values() for p in g["poses"][m]]).reshape(-1, 3)
    macros[f"nOwn{m}"] = str(len(own))
    macros[f"nAll{m}"] = str(len(comb))
    for tag, a in (("Own", own), ("All", comb)):
        if len(a) < 2:
            continue
        for k, name in enumerate(("X", "Y", "Theta")):
            mean, sd = a[:, k].mean(), a[:, k].std(ddof=1)
            macros[f"mean{name}{tag}{m}"] = f"{mean:.2f}"
            macros[f"std{name}{tag}{m}"] = f"{sd:.2f}"
            # table cell: the standard deviation sets the decimal place (Taylor)
            d = taylor_decimals(sd)
            mean = mean if round(mean, d) != 0 else 0.0           # no "-0.0"
            macros[f"pm{name}{tag}{m}"] = f"\\ensuremath{{{mean:.{d}f} \\pm {sd:.{d}f}}}"

for k, name in enumerate(("X", "Y", "Theta")):   # range of our spread over the motions
    sds = [np.array(ours[m])[:, k].std(ddof=1) for m in MOTIONS]
    macros[f"std{name}Min"], macros[f"std{name}Max"] = f"{min(sds):.2f}", f"{max(sds):.2f}"
    lo, hi = (f"{v:.{taylor_decimals(v)}f}" for v in (min(sds), max(sds)))   # as in Table II
    macros[f"std{name}Range"] = lo if lo == hi else f"{lo}--{hi}"

# battery voltage of every brick run (runs_log.csv: run number, motion, volts)
battery = {}
log = DATA / "encoder" / "runs" / "runs_log.csv"
if log.exists():
    for n, _, volts in csv.reader(open(log)):
        battery[int(n)] = float(volts)
brick_no = {run: int(f.name.split("_")[1]) for run, f in encoder_files.items()
            if f.name.startswith("run_")}
if battery:
    macros["batteryFirst"] = f"{battery[min(battery)]:.2f}"
    macros["batteryLast"] = f"{battery[max(battery)]:.2f}"
    macros["nBrickRuns"] = str(len(battery))
macros["nSamplesMin"] = str(min(len(p[0]) for p in our_paths.values()))
macros["nSamplesMax"] = str(max(len(p[0]) for p in our_paths.values()))
macros["spacingFailedRuns"] = ", ".join(r["run"] for r in poses if not r["ok"]) or "none"
macros["spacingWorstMm"] = f"{max(r['spacing_mm'] - r['tol_mm'] for r in poses):.1f}"

# manual end pose vs encoder end pose of the same run
EXPECTED_DIR = ROOT.parent / "report" / "data"       # ideal end poses from Assignment 1
compare = {}
for m in MOTIONS:
    key = lambda r: (r["motion"], int(r["run"][1:]))
    runs = [r for r in poses if r["motion"] == m and key(r) in our_paths]
    man = np.array([(r["x"], r["y"], r["theta"]) for r in runs]).reshape(-1, 3)
    enc = np.array([(our_paths[key(r)][0][-1], our_paths[key(r)][1][-1],
                     np.degrees(our_paths[key(r)][2][-1])) for r in runs]).reshape(-1, 3)
    if len(man) < 2:
        continue
    d_man, d_enc = np.hypot(man[:, 0], man[:, 1]), np.hypot(enc[:, 0], enc[:, 1])
    compare[m] = dict(runs=[r["run"] for r in runs], man=man, enc=enc, d_man=d_man,
                      d_enc=d_enc)
    for k, name in enumerate(("X", "Y", "Theta")):
        macros[f"meanEnc{name}{m}"] = f"{enc[:, k].mean():.2f}"
        macros[f"stdEnc{name}{m}"] = f"{enc[:, k].std(ddof=1):.2f}"
    macros[f"distMan{m}"] = f"{d_man.mean():.2f}"
    macros[f"distEnc{m}"] = f"{d_enc.mean():.2f}"
    macros[f"distDiff{m}"] = f"{(d_enc - d_man).mean():.2f}"
    macros[f"corrDist{m}"] = f"{np.corrcoef(d_enc, d_man)[0, 1]:.2f}"
    volts = [battery.get(brick_no.get(r, -1)) for r in compare[m]["runs"]]
    if all(v is not None for v in volts):
        compare[m]["volts"] = np.array(volts)
        macros[f"corrBattery{m}"] = f"{np.corrcoef(volts, d_enc)[0, 1]:.2f}"
    ideal = EXPECTED_DIR / f"expected_path_{m.lower()}.csv"
    if ideal.exists():
        ex, ey, eth = np.loadtxt(ideal, delimiter=",", skiprows=1)[-1]
        compare[m]["ideal"] = (ex, ey, eth)
        macros[f"idealX{m}"], macros[f"idealY{m}"] = f"{ex:.2f}", f"{ey:.2f}"
        macros[f"idealTheta{m}"] = f"{eth:.2f}"
        macros[f"idealDist{m}"] = f"{np.hypot(ex, ey):.2f}"
# effective wheel diameter: distance the marks show versus the distance the encoders compute
if "Straight" in compare:
    c = compare["Straight"]
    macros["wheelEff"] = f"{WHEEL_DIAMETER * c['d_man'].mean() / c['d_enc'].mean():.2f}"
    d = taylor_decimals(c["d_man"].std(ddof=1))           # distances rounded as in Table II
    macros["distManStraightRound"] = f"{c['d_man'].mean():.{d}f}"
    macros["idealDistStraightRound"] = f"{np.hypot(*c['ideal'][:2]):.{d}f}"
# signed pencil spacing |LR| - w of all runs: an empirical check of the reading error
dev = np.array([np.hypot(float(r["xR"]) - float(r["xL"]), float(r["yR"]) - float(r["yL"]))
                - W_PEN for r in csv.DictReader(open(DATA / "raw" / "marks.csv"))
                if r["xL"].strip()]) * 10
macros["spacingMeanMm"] = f"{dev.mean():.1f}"
macros["spacingSdMm"] = f"{dev.std(ddof=1):.1f}"
macros["spacingMaxMm"] = f"{abs(dev).max():.1f}"
# |LR| depends on the coordinate errors with gain 1 along the line L-R (two points)
macros["sigmaCoordMm"] = f"{dev.std(ddof=1) / np.sqrt(2):.1f}"
# effective wheel track: the turn the marks show versus the turn the encoders compute
for m in ("Left", "Right"):
    if m in compare:
        turn_man = abs(compare[m]["man"][:, 2].mean() - 90)
        turn_enc = abs(compare[m]["enc"][:, 2].mean() - 90)
        macros[f"turnMan{m}"], macros[f"turnEnc{m}"] = f"{turn_man:.1f}", f"{turn_enc:.1f}"
        macros["idealTurn"] = f"{abs(compare[m]['ideal'][2] - 90):.1f}"
        macros[f"trackEff{m}"] = f"{MAIN_AXIS_LENGTH * turn_enc / turn_man:.1f}"


def tex_num(v, fmt):
    s = format(v, fmt)
    return s.replace("-", "$-$") if s.startswith("-") else s


# raw data tables for the appendix, one per motion (QA Table 1 layout)
raw = {}
with open(DATA / "raw" / "marks.csv", newline="") as f:
    for row in csv.DictReader(f):
        raw[row["run"]] = row
for m in MOTIONS:
    lines = []
    for i, r in enumerate((r for r in poses if r["motion"] == m), 1):
        n = brick_no.get(r["run"])
        cells = [str(i), r["run"], str(n) if n else "--"]
        cells += [tex_num(float(raw[r["run"]][k]), ".1f") for k in ("xL", "yL", "xR", "yR")]
        cells += [tex_num(r["x"], ".2f"), tex_num(r["y"], ".2f"), tex_num(r["theta"], ".2f"),
                  f"{r['spacing_mm']:.1f}", "ok" if r["ok"] else "fail"]
        lines.append(" & ".join(cells) + r"\\")
    (DATA / f"raw_table_{m.lower()}.tex").write_text("\n".join(lines) + "\n")

with open(ROOT / "results.tex", "w") as f:
    f.write("% Generated by code/process_runs.py - do not edit by hand\n")
    for k, val in macros.items():
        val = val.replace("-", "\\ensuremath{-}", 1) if val.startswith("-") else val
        f.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.grid": True,
                     "grid.alpha": 0.4})
MOTION_MARKER = {"Left": "<", "Straight": "^", "Right": ">"}
MOTION_TITLE = {"Left": "Arc to the left", "Straight": "Straight ahead",
                "Right": "Arc to the right"}


def team_number(gname):
    """Class number of a group (None for a group without one)."""
    n = team_no(gname)
    return n if isinstance(n, int) else None


def team_label(gname):
    n = team_number(gname)
    return gname if n is None else f"Team {n}" + (" (ours)" if gname == GROUP else "")


# QA 1.2.7: every team gets its own colour, fixed by its number; the motion is told apart
# by the marker. Teams are listed by number; ours is drawn last, on top of the others.
TAB10 = plt.get_cmap("tab10").colors
LEGEND_ORDER = sorted(groups, key=lambda g: (team_number(g) is None, team_number(g) or 0, g))
GROUP_COLOR = {g: TAB10[(team_number(g) - 1) % 10] if team_number(g) else TAB10[5 + i % 5]
               for i, g in enumerate(LEGEND_ORDER)}
DRAW_ORDER = [g for g in LEGEND_ORDER if g != GROUP] + [GROUP]


def team_handles(ax, names):
    """Legend entries for the teams (empty plots, so they do not change the axes)."""
    return [ax.plot([], [], "s", ms=5, color=GROUP_COLOR[g], label=team_label(g))[0]
            for g in names]


def robot_outline(ax, x=0.0, y=0.0, th=np.pi / 2, **kw):
    """Top view of the drive base and the two pencil positions at a pose."""
    c, s = np.cos(th), np.sin(th)
    to_world = lambda u, v: (x + u * c - v * s, y + u * s + v * c)   # u forward, v left
    body = np.array([to_world(u, v) for u, v in
                     ((2.5, 4.6), (2.5, -4.6), (-11, -4.6), (-11, 4.6), (2.5, 4.6))])
    ax.plot(body[:, 0], body[:, 1], "k-", lw=1, **kw)
    for v in (MAIN_AXIS_LENGTH / 2, -MAIN_AXIS_LENGTH / 2):
        wheel = np.array([to_world(u, v) for u in (-2.8, 2.8)])
        ax.plot(wheel[:, 0], wheel[:, 1], "k-", lw=4, solid_capstyle="butt")
    for (u, v), lab in (((-B_OFF, W_PEN / 2), "L"), ((-B_OFF, -W_PEN / 2), "R")):
        px, py = to_world(u, v)
        ax.plot(px, py, "o", ms=6, mfc="none", mec="tab:red")
        ax.annotate(lab, (px, py), xytext=(4, 2), textcoords="offset points",
                    color="tab:red")


# 1) start pose
fig, ax = plt.subplots(figsize=(3.5, 2.0))
robot_outline(ax)
ax.quiver(0, 0, 0, 1, angles="xy", scale_units="xy", scale=1 / 6, color="tab:green",
          width=0.01, label="Heading at start")
ax.plot(0, 0, "ks", ms=5, label=r"Start pose $(0, 0, 90^\circ)$")
ax.plot([], [], "o", mfc="none", mec="tab:red", label="Start marks L, R")
ax.set_aspect("equal")
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 8)
ax.set_xlabel("X [cm]")
ax.set_ylabel("Y [cm]")
ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
fig.savefig(FIG / "start_pose.pdf", bbox_inches="tight")
plt.close(fig)

def scale_circles(ax, step, r):
    """QA 1.2.7: light circles behind the data; they look round only if both axes
    have the same scale. Call after the limits are final."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    for cx in np.arange(np.ceil(x0 / step) * step, x1, step):
        for cy in np.arange(np.ceil(y0 / step) * step, y1, step):
            ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec="0.85", lw=0.6, zorder=0))


def pose_arrows(ax, a, color, length, **kw):
    """End poses as arrows (QA 1.2.8): tail at (x, y), pointing along theta."""
    ax.quiver(a[:, 0], a[:, 1], np.cos(np.radians(a[:, 2])), np.sin(np.radians(a[:, 2])),
              color=color, angles="xy", scale_units="xy", scale=1 / length, width=0.004,
              **kw)


# 2) end poses of all groups (manual measurements), whole sheet
fig, ax = plt.subplots(figsize=(3.3, 3.6))
for gname in DRAW_ORDER:
    for m in MOTIONS:
        a = np.array(groups[gname]["poses"][m]).reshape(-1, 3)
        if len(a):
            pose_arrows(ax, a, GROUP_COLOR[gname], 3, alpha=0.6)
            ax.plot(a[:, 0], a[:, 1], MOTION_MARKER[m], ms=2.5, color=GROUP_COLOR[gname])
team_hs = team_handles(ax, [g for g in LEGEND_ORDER if g in with_poses])
motion_hs = [ax.plot([], [], MOTION_MARKER[m], color="0.3", label=MOTION_TITLE[m])[0]
             for m in MOTIONS]
pose_arrows(ax, np.array([[0, 0, 90]]), "k", 3)
motion_hs += ax.plot(0, 0, "ks", ms=5, label="Start pose")
ax.set_aspect("equal")
ax.set_xlim(-25, 25)
ax.set_ylim(-5, 52)
scale_circles(ax, 10, 1.5)
ax.set_xlabel("X [cm]")
ax.set_ylabel("Y [cm]")
# legends in the empty space below the arcs
ax.add_artist(ax.legend(handles=team_hs, fontsize=6, loc="lower left", framealpha=0.9))
ax.legend(handles=motion_hs, fontsize=6, loc="lower right", framealpha=0.9)
fig.savefig(FIG / "end_poses.pdf", bbox_inches="tight")
plt.close(fig)

# 2b) end poses per motion, zoomed in, same scale in every panel, 2x2 grid (legend in the free cell) for one column
all_poses = {m: np.array([p for g in groups.values() for p in g["poses"][m]]).reshape(-1, 3)
             for m in MOTIONS}
half_x, half_y = (max(np.ptp(all_poses[m][:, k]) for m in MOTIONS) / 2 + 1.0 for k in (0, 1))
fig, axs = plt.subplots(2, 2, figsize=(3.5, 2.48))
axs = axs.ravel()
for ax, m in zip(axs, MOTIONS):
    for gname in DRAW_ORDER:
        a = np.array(groups[gname]["poses"][m]).reshape(-1, 3)
        if len(a):
            pose_arrows(ax, a, GROUP_COLOR[gname], 0.8, alpha=0.7)
            ax.plot(a[:, 0], a[:, 1], ".", ms=3, color=GROUP_COLOR[gname])
    handles = team_handles(ax, [g for g in LEGEND_ORDER if g in with_poses])
    mean = all_poses[m].mean(axis=0)
    handles += ax.plot(*mean[:2], "k+", ms=10, mew=1.6, zorder=5, label="Mean (all teams)")
    cx, cy = (all_poses[m][:, 0].max() + all_poses[m][:, 0].min()) / 2, \
             (all_poses[m][:, 1].max() + all_poses[m][:, 1].min()) / 2
    ax.set_aspect("equal")
    ax.set_xlim(cx - half_x, cx + half_x)
    ax.set_ylim(cy - half_y, cy + half_y)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    scale_circles(ax, 1, 0.15)
    ax.set_xlabel("X [cm]", fontsize=7, labelpad=1)
    ax.set_ylabel("Y [cm]", fontsize=7, labelpad=1)
    ax.set_title(MOTION_TITLE[m], fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
axs[3].axis("off")
axs[3].legend(handles=handles, fontsize=7, loc="center", frameon=False)
fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6)
fig.savefig(FIG / "end_poses_zoom.pdf", bbox_inches="tight")
plt.close(fig)

# 3) encoder paths of all groups
fig, ax = plt.subplots(figsize=(3.5, 4.0))
with_paths = [g for g in DRAW_ORDER if groups[g]["paths"]]
for gname in with_paths:
    for run, (x, y, _) in groups[gname]["paths"].items():
        ax.plot(x, y, "-", color=GROUP_COLOR[gname], lw=0.6, alpha=0.7)
handles = [ax.plot([], [], "-", lw=1.5, color=GROUP_COLOR[g], label=team_label(g))[0]
           for g in LEGEND_ORDER if g in with_paths]
pose_arrows(ax, np.array([[0, 0, 90]]), "k", 3)
handles += ax.plot(0, 0, "ks", ms=5, label="Start pose")
ax.set_aspect("equal")
ax.set_xlim(-25, 25)
ax.set_ylim(-5, 52)
scale_circles(ax, 10, 1.5)
ax.set_xlabel("X [cm]")
ax.set_ylabel("Y [cm]")
ax.legend(handles=handles, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13),
          ncol=3)
fig.savefig(FIG / "encoder_paths.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- summary
for name in unmatched:
    print(f"  not used (no run ID points to it): {name}")
for gname, g in groups.items():
    for m, ns in g["copies"]:
        print(f"  {gname} {m}: runs {', '.join(map(str, ns))} have identical encoder files, "
              f"left out")
print(f"{len(poses)} runs read, {macros['nSpacingFailed']} failed the spacing check, "
      f"{len(our_paths)} encoder paths, {len(groups)} group(s)")
for r in poses:
    if not r["ok"]:
        print(f"  spacing check FAILED: {r['run']} offset {r['spacing_mm']:.1f} mm "
              f"> {r['tol_mm']:.1f} mm")
