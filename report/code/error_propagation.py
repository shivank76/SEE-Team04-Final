"""SEE Assignment 1 - expected end poses and Jacobian error propagation.

Two pencils on the sides of the brick mark the stop pose:
  L = (xl, yl), R = (xr, yr): left / right pencil, W_PEN apart on a line parallel
      to the axle, a distance B_OFF behind it
The pose of the axle midpoint is computed from the two pencil marks as
  theta = atan2(xr - xl, yl - yr)
  x = (xl + xr)/2 + B_OFF cos(theta),  y = (yl + yr)/2 + B_OFF sin(theta)
Each of the four read coordinates has a maximum measurement error of +-DELTA.

Outputs (all relative to the report folder):
  figures/*.pdf   vector figures used in the report
  data/*.csv      raw numbers behind every figure (Sandve et al., rule 7)
  results.tex     LaTeX macros with every number quoted in the text (rule 9)

Run:  python code/error_propagation.py      (from the report folder)
Tested with Python 3.9.6, NumPy 2.0.2, Matplotlib 3.9.4. No randomness is used.
"""
import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# ---------------------------------------------------------------- parameters
# From EV3_Control.py (provided program, must not be changed)
WHEEL_DIAMETER = 5.6        # cm
MAIN_AXIS_LENGTH = 12.0     # cm, distance between wheel contact points
RUN_TIME = 2.2              # s
MOTOR_MAX_SPEED = 1050.0    # deg/s, ev3dev2 max_speed of an EV3 large motor
SPEEDS = {                  # (left motor A, right motor D) in % of max speed
    "Left": (30, 40),
    "Straight": (40, 40),
    "Right": (40, 30),
}
COLORS = {"Left": "tab:blue", "Straight": "tab:green", "Right": "tab:red"}

# Measurement facility (our design choices)
W_PEN = 8.1                 # cm, distance between the pencil tips (measured)
B_OFF = 3.8                 # cm, pencil line behind the axle (measured)
DELTA = 0.15                # cm, max. error per read coordinate (see report)
DELTA_PARTS = {             # cm, contributions to DELTA (worst-case sum)
    "ruler reading (half least count)": 0.05,
    "mark diameter (centre estimate)": 0.05,
    "pencil play in its holder": 0.05,
}
assert abs(sum(DELTA_PARTS.values()) - DELTA) < 1e-12


# ---------------------------------------------------------------- kinematics
def wheel_speed(percent):
    """Wheel rim speed in cm/s for a motor speed given in percent."""
    return percent / 100 * MOTOR_MAX_SPEED / 360 * np.pi * WHEEL_DIAMETER


def ideal_path(v_left, v_right, n=200):
    """Ideal unicycle path. Coordinate convention of the SEE manual:
    Y = initial heading, X to the right, theta from +X counter-clockwise."""
    v = (v_left + v_right) / 2
    omega = (v_right - v_left) / MAIN_AXIS_LENGTH
    t = np.linspace(0, RUN_TIME, n)
    theta = np.pi / 2 + omega * t
    if abs(omega) < 1e-12:
        x, y = np.zeros_like(t), v * t
    else:
        r = v / omega
        x = r * (np.sin(theta) - np.sin(np.pi / 2))     # integral of v cos(theta)
        y = -r * (np.cos(theta) - np.cos(np.pi / 2))    # integral of v sin(theta)
    return x, y, theta, v, omega


# ---------------------------------------------------------------- orientation
def orientation(p):
    """theta from the pencil coordinates p = (xl, yl, xr, yr)."""
    xl, yl, xr, yr = p
    return np.arctan2(xr - xl, yl - yr)


def jacobian(p):
    """d theta / d(xl, yl, xr, yr), analytically."""
    xl, yl, xr, yr = p
    u, v = xr - xl, yl - yr
    d2 = u ** 2 + v ** 2
    return np.array([-v / d2, -u / d2, v / d2, u / d2])


def marks(x, y, theta, w=W_PEN, b=B_OFF):
    """Pencil marks (xl, yl, xr, yr) for an axle-midpoint pose (x, y, theta)."""
    hx, hy = np.cos(theta), np.sin(theta)          # heading
    mx, my = x - b * hx, y - b * hy                # midpoint of the pencils
    p = np.array([mx - w / 2 * hy, my + w / 2 * hx,  # left = heading turned +90
                  mx + w / 2 * hy, my - w / 2 * hx])
    return p


def position(p, b=B_OFF):
    """Axle midpoint (x, y) from the pencil coordinates p = (xl, yl, xr, yr)."""
    th = orientation(p)
    return np.array([(p[0] + p[2]) / 2 + b * np.cos(th),
                     (p[1] + p[3]) / 2 + b * np.sin(th)])


def propagate_position(p, delta=DELTA, b=B_OFF):
    """Jacobian worst case and numerical bounds of the axle position."""
    th, J = orientation(p), jacobian(p)
    Jx = np.array([0.5, 0, 0.5, 0]) - b * np.sin(th) * J
    Jy = np.array([0, 0.5, 0, 0.5]) + b * np.cos(th) * J
    combos = itertools.product((-delta, 0.0, delta), repeat=4)
    dev = np.array([position(p + np.array(c)) for c in combos]) - position(p)
    return dict(lin_x=np.sum(np.abs(Jx)) * delta, lin_y=np.sum(np.abs(Jy)) * delta,
                num_x=np.abs(dev[:, 0]).max(), num_y=np.abs(dev[:, 1]).max())


def spacing_offset(p, w=W_PEN):
    """Deviation of the read pencil spacing |LR| from the known spacing w."""
    return abs(np.hypot(p[2] - p[0], p[3] - p[1]) - w)


def spacing_tolerance(p, delta=DELTA):
    """Largest spacing deviation that reading errors alone can cause (3^4 combos)."""
    return max(spacing_offset(p + np.array(e))
               for e in itertools.product((-delta, 0.0, delta), repeat=4))


def propagate(p, delta=DELTA):
    """Return the linear (Jacobian) and numerical bounds of theta."""
    th = orientation(p)
    J = jacobian(p)
    lin = np.sum(np.abs(J)) * delta                  # worst case, |J| . delta
    sigma = np.sqrt(np.sum(J ** 2)) * delta / np.sqrt(3)  # 1 sigma, uniform err.
    # numerical: every combination of {-delta, 0, +delta} on 4 coordinates
    combos = np.array(list(itertools.product((-delta, 0.0, delta), repeat=4)))
    ths = np.array([orientation(p + c) for c in combos])
    ths = th + np.angle(np.exp(1j * (ths - th)))     # unwrap around th
    return dict(theta=th, J=J, lin=lin, sigma=sigma,
                lo=ths.min(), hi=ths.max(), combos=combos, thetas=ths)


# ---------------------------------------------------------------- rounding
def sig_unc(u):
    """Round an uncertainty as in the lecture (Taylor): 1 significant figure,
    2 if the leading digit is 1. Returns (string, number of decimals)."""
    exp = int(np.floor(np.log10(abs(u))))
    n_sig = 2 if f"{abs(u):e}"[0] == "1" else 1
    n_dec = max(0, n_sig - 1 - exp)
    return f"{u:.{n_dec}f}", n_dec


# ---------------------------------------------------------------- compute
results, macros = {}, {}
for name, (pl, pr) in SPEEDS.items():
    vl, vr = wheel_speed(pl), wheel_speed(pr)
    x, y, th, v, om = ideal_path(vl, vr)
    p = marks(x[-1], y[-1], th[-1])
    res = propagate(p)
    res.update(vl=vl, vr=vr, v=v, omega=om, path=(x, y, th), marks=p,
               pos=propagate_position(p), tol=spacing_tolerance(p))
    results[name] = res

    np.savetxt(DATA / f"expected_path_{name.lower()}.csv",
               np.column_stack([x, y, np.degrees(th)]), delimiter=",", fmt="%.4f",
               header="x_cm,y_cm,theta_deg", comments="")
    np.savetxt(DATA / f"theta_combinations_{name.lower()}.csv",
               np.column_stack([res["combos"], np.degrees(res["thetas"])]),
               delimiter=",", fmt="%.5f",
               header="dxl_cm,dyl_cm,dxr_cm,dyr_cm,theta_deg", comments="")

    k = name  # LaTeX macro suffix
    d_lin, n_dec = sig_unc(np.degrees(res["lin"]))   # angles share these decimals
    macros.update({
        f"vL{k}": f"{vl:.1f}", f"vR{k}": f"{vr:.1f}", f"v{k}": f"{v:.1f}",
        f"omega{k}": f"{np.degrees(om):.1f}",
        f"x{k}": f"{x[-1]:.1f}", f"y{k}": f"{y[-1]:.1f}",
        f"theta{k}": f"{np.degrees(th[-1]):.1f}",
        f"Jxl{k}": f"{res['J'][0]:.4f}", f"Jyl{k}": f"{res['J'][1]:.4f}",
        f"Jxr{k}": f"{res['J'][2]:.4f}", f"Jyr{k}": f"{res['J'][3]:.4f}",
        f"dLin{k}": d_lin,
        f"sigma{k}": sig_unc(np.degrees(res["sigma"]))[0],
        f"lo{k}": f"{np.degrees(res['lo']):.{n_dec}f}",
        f"hi{k}": f"{np.degrees(res['hi']):.{n_dec}f}",
        f"dMinus{k}": sig_unc(np.degrees(res["theta"] - res["lo"]))[0],
        f"dPlus{k}": sig_unc(np.degrees(res["hi"] - res["theta"]))[0],
        f"range{k}": f"{np.degrees(res['hi'] - res['lo']):.{n_dec}f}",
        f"dPosX{k}": sig_unc(res['pos']['num_x'] * 10)[0],
        f"dPosY{k}": sig_unc(res['pos']['num_y'] * 10)[0],
    })

pos_all = [max(r["pos"]["num_x"], r["pos"]["num_y"]) for r in results.values()]
tol_all = max(r["tol"] for r in results.values())
macros.update({"wPen": f"{W_PEN:.1f}", "halfW": f"{W_PEN / 2:.2f}",
               "bOff": f"{B_OFF:.1f}",
               "deltaCm": f"{DELTA:.2f}", "trackCm": f"{MAIN_AXIS_LENGTH:.1f}",
               "posMaxMm": sig_unc(max(pos_all) * 10)[0],
               "spacingTolMm": f"{np.ceil(tol_all * 100) / 10:.1f}",
               "dPosBound": sig_unc((DELTA + B_OFF * 2 * np.sqrt(2) * DELTA / W_PEN) * 10)[0],
               "deltaMm": f"{DELTA * 10:.1f}",
               "deltaRssMm": f"{np.sqrt(sum(v**2 for v in DELTA_PARTS.values())) * 10:.1f}",
               "maxSpeed": f"{MOTOR_MAX_SPEED:.0f}",
               # closed form over all orientations: 2 delta/w (|sin|+|cos|)
               "dMinAll": sig_unc(np.degrees(2 * DELTA / W_PEN))[0],
               "dMaxAll": sig_unc(np.degrees(2 * np.sqrt(2) * DELTA / W_PEN))[0]})
with open(ROOT / "results.tex", "w") as f:
    f.write("% Generated by code/error_propagation.py - do not edit by hand\n")
    for k, val in macros.items():
        if val.startswith("-") and float(val) == 0:   # avoid "-0.0000"
            val = val[1:]
        val = val.replace("-", "\\ensuremath{-}", 1) if val.startswith("-") else val
        f.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.grid": True,
                     "grid.alpha": 0.4})


def scale_circles(ax, rmax, step):
    """Light grey circles that make the equal axis scale visible (QA 1.2.7)."""
    for r in np.arange(step, rmax + step, step):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="0.85", lw=0.8,
                                zorder=0))


# 1) expected paths and end poses
fig, ax = plt.subplots(figsize=(3.5, 4.0))
scale_circles(ax, 50, 10)
for name, r in results.items():
    x, y, th = r["path"]
    p = r["marks"]
    ax.plot(x, y, color=COLORS[name], lw=1.8, label=f"{name}: expected path")
    ax.quiver(x[-1], y[-1], np.cos(th[-1]), np.sin(th[-1]), color=COLORS[name],
              angles="xy", scale_units="xy", scale=1 / 8, width=0.006, zorder=4)
    ax.plot([p[0], p[2]], [p[1], p[3]], "o", ms=3, color="k", zorder=5)
ax.quiver(0, 0, 0, 1, color="k", angles="xy", scale_units="xy", scale=1 / 8,
          width=0.006, zorder=4, label="Pose (arrow = orientation)")
ax.plot(0, 0, "ks", ms=5, label=r"Start pose (0, 0, 90$^\circ$)")
ax.plot([], [], "o", ms=3, color="k", label="Pencil marks L, R")
ax.set_aspect("equal")
ax.set_xlim(-35, 35)
ax.set_ylim(-15, 55)
ax.set_xlabel("X [cm]")
ax.set_ylabel("Y [cm]")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(FIG / "expected_poses.pdf", bbox_inches="tight")
plt.close(fig)

# 2) worst-case orientation error vs. pencil separation
ls = np.linspace(4, 30, 200)
fig, ax = plt.subplots(figsize=(3.5, 2.4))
rows = [ls]
for d in (0.10, 0.15, 0.20):
    worst = [np.degrees(propagate(marks(0, 0, np.radians(135), w), d)["lin"])
             for w in ls]
    rows.append(worst)
    ax.plot(ls, worst, lw=1.8, label=rf"$\delta$ = {d * 10:.1f} mm")
ax.axvline(W_PEN, color="k", ls="--", lw=1, label=f"built w = {W_PEN:.1f} cm")
ax.set_xlabel("Pencil separation w [cm]")
ax.set_ylabel(r"Worst-case $|\Delta\theta|$ [deg]")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "dtheta_vs_l.pdf", bbox_inches="tight")
plt.close(fig)
np.savetxt(DATA / "dtheta_vs_l.csv", np.column_stack(rows), delimiter=",",
           fmt="%.4f", header="w_cm,d1.0mm_deg,d1.5mm_deg,d2.0mm_deg",
           comments="")

# 3) orientation error vs. heading (linear vs numerical)
heads = np.linspace(0, 180, 361)
lin, lo, hi = [], [], []
for h in heads:
    r = propagate(marks(0, 0, np.radians(h)))
    lin.append(np.degrees(r["lin"]))
    lo.append(np.degrees(r["lo"] - r["theta"]))
    hi.append(np.degrees(r["hi"] - r["theta"]))
fig, ax = plt.subplots(figsize=(3.5, 3.2))
ax.fill_between(heads, lo, hi, color="0.85", label="Numerical bounds (81 combinations)")
ax.plot(heads, lin, "k-", lw=1.5, label=r"Jacobian worst case $\pm|J|\cdot\delta$")
ax.plot(heads, -np.array(lin), "k-", lw=1.5)
for name, r in results.items():
    ax.axvline(np.degrees(r["theta"]), color=COLORS[name], ls="--", lw=1.2,
               label=rf"expected $\theta$, {name}")
ax.set_xlabel(r"End orientation $\theta$ [deg]")
ax.set_ylabel(r"$\Delta\theta$ [deg]")
ax.set_xlim(0, 180)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
fig.tight_layout()
fig.savefig(FIG / "dtheta_vs_heading.pdf", bbox_inches="tight")
plt.close(fig)
np.savetxt(DATA / "dtheta_vs_heading.csv", np.column_stack([heads, lin, lo, hi]),
           delimiter=",", fmt="%.4f",
           header="theta_deg,jacobian_deg,num_lower_deg,num_upper_deg", comments="")

# ---------------------------------------------------------------- summary
for name, r in results.items():
    print(f"{name:8s} end=({macros['x'+name]}, {macros['y'+name]}, "
          f"{macros['theta'+name]} deg)  Jacobian +-{macros['dLin'+name]} deg  "
          f"numerical [{macros['lo'+name]}, {macros['hi'+name]}] deg  "
          f"sigma {macros['sigma'+name]} deg  "
          f"pos +-({macros['dPosX'+name]}, {macros['dPosY'+name]}) mm")
print(f"spacing check tolerance {macros['spacingTolMm']} mm")
