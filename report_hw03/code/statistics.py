"""SEE Assignment 3 - statistical evaluation of the combined end poses of all teams.

Inputs (from the Assignment 2 folder ../report_hw02, written by its code/process_runs.py):
  data/combined_manual.csv     manual end poses of all teams: team, motion, run, x, y, theta
  data/combined_encoder.csv    encoder end poses of all teams (run empty if it cannot be
                               paired with a manual run)
  data/raw/marks.csv           our raw pencil readings (random reading error)
  results.tex                  worst-case reading errors, effective wheel diameter and track
and from the Assignment 1 folder ../report: results.tex (pencil geometry) and
data/expected_path_<motion>.csv (ideal end poses, reference for the accuracy).

Outputs (in this folder):
  results.tex                  every number quoted in the report
  data/*.tex                   table rows (Table 4, encoder comparison, uncertainties, class data)
  data/outliers.csv            the runs removed by the Chebyshev test
  figures/*.pdf                encoder differences, outliers, histograms, PCA

Run:  python code/statistics.py     (from this folder, after ../report_hw02/code/process_runs.py)
No randomness is used.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG = ROOT / "data", ROOT / "figures"
A1, A2 = ROOT.parent / "report", ROOT.parent / "report_hw02"     # earlier assignments
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
MOTIONS = ("Left", "Straight", "Right")
VARS = ("x", "y", "theta")
VAR_TEX = {"x": "$x$", "y": "$y$", "theta": r"$\theta$"}
VAR_UNIT = {"x": "cm", "y": "cm", "theta": "deg"}
OURS = 4                    # our team number
K_CHEB = 2.0                # Chebyshev threshold: mu +- 2 sigma (manual, Assignment 3)
ALPHA = 0.05                # significance level of all tests
POS_TOL = 1.0               # cm; a mean position difference below this counts as a fit
WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}   # team number in macro names


# ---------------------------------------------------------------- input
def read_poses(path):
    """{(team, motion): [(run or None, x, y, theta), ...]} from a combined table."""
    out = {}
    for r in csv.DictReader(open(path)):
        run = int(r["run"]) if r["run"].strip() else None
        out.setdefault((int(r["team"]), r["motion"]), []).append(
            (run, float(r["x_cm"]), float(r["y_cm"]), float(r["theta_deg"])))
    return out


def tex_macros(path):
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", path.read_text()))


manual = read_poses(A2 / "data" / "combined_manual.csv")
encoder = read_poses(A2 / "data" / "combined_encoder.csv")
TEAMS = sorted({t for t, _ in manual})
a1, a2 = tex_macros(A1 / "results.tex"), tex_macros(A2 / "results.tex")
ideal = {m: np.loadtxt(A1 / "data" / f"expected_path_{m.lower()}.csv",
                       delimiter=",", skiprows=1)[-1] for m in MOTIONS}


def taylor_decimals(u):
    """Decimal places for a value with uncertainty u (Lecture 1, Taylor), as in
    process_runs.py: one significant figure, two if the leading digit is 1."""
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


def sig(v, n=2):
    """v with n significant figures (for variances and test statistics)."""
    if v == 0:
        return "0"
    d = max(0, n - 1 - int(np.floor(np.log10(abs(v)))))
    return tex_num(v, d)


def p_text(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def pct(v, d=0):
    return f"{v * 100:.{d}f}"


macros = {"kCheb": f"{K_CHEB:g}", "chebBound": pct(1 - 1 / K_CHEB**2),
          "alphaLevel": f"{ALPHA:g}", "nTeams": str(len(TEAMS)),
          "numpyVersion": np.__version__, "scipyVersion": scipy.__version__,
          "mplVersion": matplotlib.__version__,
          "pythonVersion": ".".join(map(str, sys.version_info[:3]))}


# ---------------------------------------------------------------- 1) Chebyshev outliers
def chebyshev_inliers(v, k=K_CHEB):
    """Single pass of Chebyshev's outlier removal (Lecture 3, get_inliers in
    OutlierFilteringbyChebyshev.ipynb; Amidan et al.): keep values within mu +- k sigma.
    k = 2 corresponds to pp = 1/k^2 = 0.25 in the lecture's notation; sigma is np.std as
    in the lecture."""
    mu, sigma = v.mean(), v.std()
    return (v >= mu - k * sigma) & (v <= mu + k * sigma)


combined, bounds = {}, {}
outlier_rows = []
for m in MOTIONS:
    rows = [(t, run, x, y, th) for t in TEAMS for run, x, y, th in manual.get((t, m), [])]
    a = np.array([r[2:] for r in rows])
    team = np.array([r[0] for r in rows])
    run = np.array([r[1] for r in rows])
    ok_var = np.column_stack([chebyshev_inliers(a[:, k]) for k in range(3)])
    ok = ok_var.all(axis=1)                 # a run is kept only if x, y and theta are inliers
    combined[m] = dict(a=a, team=team, run=run, ok=ok, ok_var=ok_var)
    bounds[m] = [(a[:, k].mean() - K_CHEB * a[:, k].std(), a[:, k].mean() + K_CHEB * a[:, k].std())
                 for k in range(3)]
    for i in np.where(~ok)[0]:
        outlier_rows.append((m, team[i], run[i], *a[i],
                             "+".join(v for k, v in enumerate(VARS) if not ok_var[i, k])))

with open(DATA / "outliers.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["motion", "team", "run", "x_cm", "y_cm", "theta_deg", "beyond_2sigma_in"])
    for r in outlier_rows:
        w.writerow([r[0], r[1], r[2], f"{r[3]:.2f}", f"{r[4]:.2f}", f"{r[5]:.2f}", r[6]])

for m in MOTIONS:
    c = combined[m]
    macros[f"nRaw{m}"] = str(len(c["a"]))
    macros[f"nOut{m}"] = str((~c["ok"]).sum())
    macros[f"nIn{m}"] = str(c["ok"].sum())
macros["nOutAll"] = str(sum((~combined[m]["ok"]).sum() for m in MOTIONS))
macros["nRawAll"] = str(sum(len(combined[m]["a"]) for m in MOTIONS))
out_team = {t: sum(((combined[m]["team"] == t) & ~combined[m]["ok"]).sum() for m in MOTIONS)
            for t in TEAMS}
macros["outByTeamText"] = ", ".join(f"Team~{t}: {n}" for t, n in out_team.items())


# ---------------------------------------------------------------- 2) Gaussian fit, chi-square
def chi2_gauss(v, n_par=2):
    """Chi-square goodness-of-fit test of v against the Gaussian N(mean, sd) fitted to v.
    Bins of equal probability under the fit, n // 10 of them but at least 5, so that every
    bin expects at least 5 values; degrees of freedom = bins - 1 - n_par (Lecture 5).
    Also returns the inner bin edges and the observed counts."""
    n = len(v)
    k = max(5, n // 10)
    mu, sd = v.mean(), v.std(ddof=1)
    inner = stats.norm.ppf(np.arange(1, k) / k, mu, sd)        # k - 1 inner bin edges
    obs = np.bincount(np.searchsorted(inner, v), minlength=k)
    exp = n / k
    chi2 = ((obs - exp)**2 / exp).sum()
    dof = k - 1 - n_par
    return dict(chi2=chi2, dof=dof, p=stats.chi2.sf(chi2, dof), bins=k, inner=inner, obs=obs,
                exp=exp)


def chi2_gauss_2d(a):
    """Chi-square test of a bivariate Gaussian fit to the points a (n x 2): for Gaussian data
    the squared Mahalanobis distance d^2 follows a chi-square distribution with 2 degrees of
    freedom. Bins of equal probability of chi2(2); 5 fitted parameters (2 means, 3 entries
    of the covariance matrix)."""
    n = len(a)
    k = max(7, n // 10)
    d = a - a.mean(axis=0)
    d2 = np.einsum("ij,jk,ik->i", d, np.linalg.inv(np.cov(a.T)), d)
    inner = stats.chi2.ppf(np.arange(1, k) / k, 2)
    obs = np.bincount(np.searchsorted(inner, d2), minlength=k)
    exp = n / k
    chi2 = ((obs - exp)**2 / exp).sum()
    dof = k - 1 - 5
    return dict(chi2=chi2, dof=dof, p=stats.chi2.sf(chi2, dof), bins=k)


fits = {}
t4_rows = []
for m in MOTIONS:
    c = combined[m]
    a = c["a"][c["ok"]]
    for k, v in enumerate(VARS):
        col = a[:, k]
        f = chi2_gauss(col)
        f.update(mean=col.mean(), sd=col.std(ddof=1), acc=col.mean() - ideal[m][k])
        fits[(m, v)] = f
        d = taylor_decimals(f["sd"])
        if k == 0:
            t4_rows.append(rf"\multicolumn{{8}}{{l}}{{{m} ($n = {len(col)}$)}}\\")
        t4_rows.append(" & ".join([VAR_TEX[v], tex_num(f["mean"], d), sig(f["sd"]**2),
                                   tex_num(f["acc"], d), f"{f['chi2']:.1f}", str(f["dof"]),
                                   p_text(f["p"]),
                                   "Reject" if f["p"] < ALPHA else "Fail to reject"]) + r"\\")
        macros[f"acc{v.capitalize()}{m}"] = tex_num(f["acc"], d)
    f = chi2_gauss_2d(a[:, :2])
    fits[(m, "xy")] = f
    t4_rows.append(" & ".join(["$(x, y)$", "--", "--", "--", f"{f['chi2']:.1f}",
                               str(f["dof"]), p_text(f["p"]),
                               "Reject" if f["p"] < ALPHA else "Fail to reject"]) + r"\\")
    if m != MOTIONS[-1]:
        t4_rows.append(r"\midrule")
(DATA / "table4.tex").write_text("\n".join(t4_rows) + "\n")
macros["nRejected"] = str(sum(f["p"] < ALPHA for f in fits.values()))
macros["nTests"] = str(len(fits))
MOTION_TEXT = {"Left": "to the left", "Straight": "straight ahead", "Right": "to the right"}
VAR_TEXT = {**VAR_TEX, "xy": "$(x, y)$"}
p_math = lambda p: "$p < 0.001$" if p < 0.001 else f"$p = {p:.3f}$"
macros["rejectedText"] = "; ".join(f"{VAR_TEXT[v]} {MOTION_TEXT[m]} ({p_math(f['p'])})"
                                   for (m, v), f in fits.items() if f["p"] < ALPHA)
exp_bin = [fits[(m, "x")]["exp"] for m in MOTIONS]
macros["expBinMin"], macros["expBinMax"] = f"{min(exp_bin):.0f}", f"{max(exp_bin):.0f}"
macros["binsText"] = " or ".join(map(str, sorted({f["bins"] for (m, v), f in fits.items()
                                                  if v != "xy"})))

# theta straight ahead: the readings take discrete values (1 mm over the marker spacing), so a
# bin narrower than one step can stay empty; quote the emptiest bin
f = fits[("Straight", "theta")]
i = int(f["obs"].argmin())
macros["emptyBinLo"] = f"{f['inner'][i - 1]:.2f}"
macros["emptyBinHi"] = f"{f['inner'][i]:.2f}"
macros["emptyBinObs"] = str(f["obs"][i])
macros["emptyBinExp"] = f"{f['exp']:.1f}"
macros["thetaStepDeg"] = f"{np.degrees(np.arctan(0.1 / float(a1['wPen']))):.1f}"


# ---------------------------------------------------------------- 3) team effect
# theta to the right: Team 2 and the other teams form two separate peaks
c = combined["Right"]
macros["meanThetaTwoRight"] = f"{c['a'][c['ok'] & (c['team'] == 2), 2].mean():.0f}"
macros["meanThetaOthersRight"] = f"{c['a'][c['ok'] & (c['team'] != 2), 2].mean():.0f}"


# ---------------------------------------------------------------- 4) PCA
pca = {}
for m in MOTIONS:
    c = combined[m]
    a = c["a"][c["ok"]][:, :2]
    mean = a.mean(axis=0)
    lam, vec = np.linalg.eigh(np.cov(a.T))
    order = lam.argsort()[::-1]
    lam, vec = lam[order], vec[:, order]
    if vec[0, 0] < 0:                       # PC1 points to +X (sign of an eigenvector is free)
        vec[:, 0] *= -1
    if np.linalg.det(vec) < 0:              # right-handed: PC2 is PC1 turned by +90 deg
        vec[:, 1] *= -1
    proj = (a - mean) @ vec
    r2 = (proj[:, 0]**2 / lam[0]) + (proj[:, 1]**2 / lam[1])
    inside = [(r2 <= s**2).mean() for s in (1, 2, 3)]
    angle = np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))
    pca[m] = dict(mean=mean, lam=lam, vec=vec, proj=proj, team=c["team"][c["ok"]],
                  inside=inside, angle=angle)
    macros[f"pcaSdOne{m}"] = f"{np.sqrt(lam[0]):.2f}"
    macros[f"pcaSdTwo{m}"] = f"{np.sqrt(lam[1]):.2f}"
    macros[f"pcaAngle{m}"] = f"{angle:.0f}"
    macros[f"pcaVar{m}"] = pct(lam[0] / lam.sum())
    for s, frac in zip(("One", "Two", "Three"), inside):
        macros[f"pcaIn{s}{m}"] = pct(frac)
for s, n in zip(("One", "Two", "Three"), (1, 2, 3)):
    macros[f"gaussIn{s}"] = f"{stats.chi2.cdf(n**2, 2) * 100:.1f}"


# ---------------------------------------------------------------- 5) manual vs. encoder
enc_rows, compare = [], {}
for t in TEAMS:
    for m in MOTIONS:
        man = {r[0]: np.array(r[1:]) for r in manual.get((t, m), [])}
        enc_all = encoder.get((t, m), [])
        if not enc_all:
            continue
        enc = {r[0]: np.array(r[1:]) for r in enc_all if r[0] is not None}
        pairs = sorted(set(man) & set(enc))
        e_arr = np.array([r[1:] for r in enc_all])
        m_arr = np.array(list(man.values()))
        if pairs:
            d = np.array([man[n] - enc[n] for n in pairs])
            dm, ds = d.mean(axis=0), d.std(axis=0, ddof=1)
            dec = [taylor_decimals(u) for u in ds]          # the SD sets the decimals (Taylor)
            cells = [tex_num(dm[k], dec[k]) + " $\\pm$ " + tex_num(ds[k], dec[k])
                     for k in range(3)]
            n_text = str(len(pairs))
        else:                               # no pairing possible: compare the means only
            d, dec = None, [1, 1, 1]
            dm = m_arr.mean(axis=0) - e_arr.mean(axis=0)
            cells = [tex_num(dm[k], 1) + r"\textsuperscript{a}" for k in range(3)]
            n_text = f"{len(m_arr)}/{len(e_arr)}"
        compare[(t, m)] = dict(dm=dm, d=d, runs=pairs)
        macros[f"nPairs{WORD[t]}{m}"] = str(len(pairs))
        enc_rows.append(" & ".join([f"Team~{t}" if m == MOTIONS[0] else "", m, n_text] + cells)
                        + r"\\")
        for k, v in enumerate(VARS):
            macros[f"d{v.capitalize()}{WORD[t]}{m}"] = tex_num(dm[k], dec[k])
    if t != TEAMS[-1]:
        enc_rows.append(r"\midrule")
(DATA / "encoder_compare.tex").write_text("\n".join(enc_rows) + "\n")

macros["nRowsEnc"] = str(len(compare))
macros["nPosMeans"] = str(2 * len(compare))
macros["posTol"] = f"{POS_TOL:g}"
macros["nPosBelowTol"] = str(sum((abs(c["dm"][:2]) < POS_TOL).sum() for c in compare.values()))
macros["maxDPosAll"] = f"{max(abs(c['dm'][:2]).max() for c in compare.values()):.1f}"
macros["nNegY"] = str(sum(c["dm"][1] < 0 for c in compare.values()))
macros["maxDThetaOthers"] = f"{max(abs(c['dm'][2]) for (t, m), c in compare.items() if t != 2):.0f}"
# the largest disagreement in theta among the paired runs
worst = max(((t, m, n, man_r, enc_r) for (t, m) in compare
             for n, man_r, enc_r in
             [(r[0], np.array(r[1:]), np.array(e[1:])) for r in manual[(t, m)]
              for e in encoder[(t, m)] if e[0] == r[0]]),
            key=lambda z: abs(z[3][2] - z[4][2]))
macros["worstTeam"], macros["worstMotion"], macros["worstRun"] = (str(worst[0]),
                                                                  MOTION_TEXT[worst[1]],
                                                                  str(worst[2]))
macros["worstThetaMan"] = f"{worst[3][2]:.0f}"
macros["worstThetaEnc"] = f"{worst[4][2]:.0f}"
for k in ("trackEffLeft", "trackCm", "wheelEff"):
    macros[k] = a1[k] if k in a1 else a2[k]
macros["wheelNom"] = "5.6"                  # nominal wheel diameter (LEGO tyre 56 mm)


# ---------------------------------------------------------------- 6) uncertainty comparison
# random reading error of our end poses. The two marks of a run lie w apart, so the scatter of
# |LR| - w over all runs gives the error of one read coordinate (two points, gain 1 along
# L-R): sigma_c = sd / sqrt(2). It is propagated linearly (Gaussian error propagation)
# through the pose equations of Assignment 2 at the marks of every run; mean over the runs.
W_PEN, B_OFF = float(a1["wPen"]), float(a1["bOff"])
marks = np.array([[float(r[k]) for k in ("xL", "yL", "xR", "yR")]
                  for r in csv.DictReader(open(A2 / "data" / "raw" / "marks.csv")) if r["xL"].strip()])


def pose(p):
    """(x, y, theta in deg) of the axle midpoint from the marks p = (xl, yl, xr, yr)."""
    th = np.arctan2(p[2] - p[0], p[1] - p[3])
    return np.array([(p[0] + p[2]) / 2 + B_OFF * np.cos(th),
                     (p[1] + p[3]) / 2 + B_OFF * np.sin(th), np.degrees(th)])


def pose_jacobian(p, h=1e-4):
    """d(x, y, theta) / d(xl, yl, xr, yr), central differences."""
    return np.column_stack([(pose(p + h * e) - pose(p - h * e)) / (2 * h) for e in np.eye(4)])


sigma_c = (np.hypot(marks[:, 2] - marks[:, 0], marks[:, 3] - marks[:, 1]) - W_PEN) \
    .std(ddof=1) / np.sqrt(2)                                          # cm per coordinate
sig_read = np.mean([np.sqrt((pose_jacobian(p)**2).sum(axis=1)) * sigma_c for p in marks],
                   axis=0)
macros["sigmaCoordMm"] = f"{sigma_c * 10:.1f}"

unc_rows = []
for k, v in enumerate(VARS):
    own = [combined[m]["a"][combined[m]["team"] == OURS][:, k].std(ddof=1) for m in MOTIONS]
    comb = [fits[(m, v)]["sd"] for m in MOTIONS]
    dec = 1 if v == "theta" else 2
    unc_rows.append(" & ".join([f"{VAR_TEX[v]} [{VAR_UNIT[v]}]",
                                a2["uPosCm"] if v != "theta" else a2["uThetaDeg"],
                                f"{sig_read[k]:.{dec}f}"]
                               + [f"{s:.{dec}f}" for s in own + comb]) + r"\\")
(DATA / "uncertainty.tex").write_text("\n".join(unc_rows) + "\n")


# ---------------------------------------------------------------- 7) accuracy vs. the ideal pose
macros["maxAccPos"] = f"{max(abs(fits[(m, v)]['acc']) for m in MOTIONS for v in ('x', 'y')):.1f}"
macros["maxAccTheta"] = f"{max(abs(fits[(m, 'theta')]['acc']) for m in MOTIONS):.0f}"
macros["posMaxMm"], macros["dMaxAll"] = a1["posMaxMm"], a1["dMaxAll"]


# ---------------------------------------------------------------- raw class data (appendix)
removed = {(r[0], r[1], r[2]) for r in outlier_rows}


def class_rows(poses, mark_removed):
    """Table rows: row number, then x, y, theta for each motion (blank where missing).
    Paired data are placed at their run number, unpaired rows in the order received."""
    out = {}
    for t in TEAMS:
        cols = {}
        for m in MOTIONS:
            rows = poses.get((t, m), [])
            cols[m] = {(r[0] if r[0] is not None else i + 1): r for i, r in enumerate(rows)}
        n_rows = max(max(c, default=0) for c in cols.values())
        lines = []
        for i in range(1, n_rows + 1):
            cells = [str(i)]
            for m in MOTIONS:
                r = cols[m].get(i)
                if r is None:
                    cells += ["--", "--", "--"]
                    continue
                mark = (r"\rlap{\textsuperscript{*}}"
                        if mark_removed and (m, t, r[0]) in removed else "")   # no shift
                cells += [tex_num(r[1], 2), tex_num(r[2], 2), tex_num(r[3], 2) + mark]
            lines.append(" & ".join(cells) + r"\\")
        out[t] = "\n".join(lines) + "\n"
    return out


for t, text in class_rows(manual, True).items():
    (DATA / f"class_team{t}.tex").write_text(text)
for t, text in class_rows(encoder, False).items():
    (DATA / f"encoder_team{t}.tex").write_text(text)

with open(ROOT / "results.tex", "w") as f:
    f.write("% Generated by code/statistics.py - do not edit by hand\n")
    for k, val in macros.items():
        assert re.fullmatch(r"[A-Za-z]+", k), k
        val = val.replace("-", "\\ensuremath{-}", 1) if val.startswith("-") else val
        f.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")


# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.grid": True,
                     "grid.alpha": 0.4})
TAB10 = plt.get_cmap("tab10").colors
COLOR = {t: TAB10[(t - 1) % 10] for t in TEAMS}         # as in process_runs.py
TITLE = {"Left": "Arc to the left", "Straight": "Straight ahead", "Right": "Arc to the right"}
DRAW = [t for t in TEAMS if t != OURS] + [OURS]          # our team is drawn last, on top


def team_label(t):
    return f"Team {t}" + (" (ours)" if t == OURS else "")


def team_handles(ax):
    return [ax.plot([], [], "s", ms=5, color=COLOR[t], label=team_label(t))[0] for t in TEAMS]


# a) manual minus encoder end pose, per run, grouped by team and motion
fig, axs = plt.subplots(3, 1, figsize=(3.5, 3.2), sharex=True)      # one column wide
TH_LIM = (-20, 36)                                     # Delta theta axis; larger values clipped
xpos = {(t, m): (t - 1) * 4 + j for t in TEAMS for j, m in enumerate(MOTIONS)}
for k, (ax, v) in enumerate(zip(axs, VARS)):
    ax.axhline(0, color="k", lw=0.7)
    for (t, m), c in compare.items():
        x0 = xpos[(t, m)]
        if c["d"] is None:                             # not paired: difference of the means
            ax.plot(x0, c["dm"][k], "D", ms=4, mfc="none", mec=COLOR[t], mew=1)
            continue
        dv = c["d"][:, k]
        jit = np.linspace(-0.25, 0.25, len(dv))        # spread the runs sideways, no randomness
        if v == "theta":
            off = dv > TH_LIM[1]
            ax.plot(x0 + jit[~off], dv[~off], ".", ms=3, color=COLOR[t], alpha=0.7)
            for xj, val in zip(x0 + jit[off], dv[off]):
                ax.plot(xj, TH_LIM[1] - 4, "^", ms=4, color=COLOR[t])
                ax.annotate(f"{val:.0f}" + r"$^\circ$", (xj, TH_LIM[1] - 4), xytext=(5, -2.5),
                            textcoords="offset points", fontsize=6)
        else:
            ax.plot(x0 + jit, dv, ".", ms=3, color=COLOR[t], alpha=0.7)
        ax.plot(x0, c["dm"][k], "_", ms=12, mew=1.5, color="k")
    ax.set_ylabel(rf"$\Delta${VAR_TEX[v]} [{VAR_UNIT[v]}]")
    ax.grid(axis="x", visible=False)
axs[2].set_ylim(*TH_LIM)
axs[2].set_xticks(list(xpos.values()), [m[0] for t, m in xpos], minor=True)
axs[2].set_xticks([(t - 1) * 4 + 1 for t in TEAMS],
                  [f"Team {t}" + ("\n(ours)" if t == OURS else "") for t in TEAMS])
axs[2].tick_params(axis="x", which="major", pad=11, length=0)
axs[2].tick_params(axis="x", which="minor", labelsize=6)
handles = axs[0].plot([], [], "_", ms=12, mew=1.5, color="k", label="Mean of the paired runs")
handles += axs[0].plot([], [], "D", ms=4, mfc="none", mec="0.3", label="Means only (not paired)")
fig.legend(handles=handles, fontsize=7, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.6)
fig.savefig(FIG / "manual_vs_encoder.pdf", bbox_inches="tight")
plt.close(fig)

# b) outliers: end positions per motion, Chebyshev bounds in x and y, removed runs crossed.
# One column wide: the three motions side by side.
fig, axs = plt.subplots(1, 3, figsize=(3.5, 1.85))
for ax, m in zip(axs, MOTIONS):
    c = combined[m]
    for t in DRAW:
        sel = c["team"] == t
        ax.plot(c["a"][sel & c["ok"], 0], c["a"][sel & c["ok"], 1], ".", ms=2, color=COLOR[t])
        ax.plot(c["a"][sel & ~c["ok"], 0], c["a"][sel & ~c["ok"], 1], "o", ms=3.5, mfc="none",
                mec=COLOR[t], mew=0.7)
    ax.plot(c["a"][~c["ok"], 0], c["a"][~c["ok"], 1], "kx", ms=2.5, mew=0.6)
    (x0, x1), (y0, y1) = bounds[m][0], bounds[m][1]
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k--", lw=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(m, fontsize=7, pad=2)
    ax.set_xlabel("X [cm]", fontsize=7, labelpad=1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(4))       # same tick step on both axes
    ax.yaxis.set_major_locator(plt.MultipleLocator(4))
    ax.tick_params(labelsize=6, pad=1)
axs[0].set_ylabel("Y [cm]", fontsize=7, labelpad=1)
handles = team_handles(axs[0])
handles += axs[0].plot([], [], "kx", ms=4, label="Outlier")
handles += axs[0].plot([], [], "k--", lw=0.8, label=r"$\mu \pm 2\sigma$ in $x$, $y$")
fig.legend(handles=handles, fontsize=6, loc="lower center", ncol=4, frameon=False,
           handletextpad=0.3, columnspacing=0.8)
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.3)
fig.savefig(FIG / "outliers.pdf", bbox_inches="tight")
plt.close(fig)

# c) histograms of the inliers with the fitted Gaussian and the mixture of the teams' Gaussians.
# One column wide: rows = motions, columns = x, y, theta; p value of the chi-square test inside.
fig, axs = plt.subplots(3, 3, figsize=(3.5, 3.6))
for i, m in enumerate(MOTIONS):
    c = combined[m]
    a, team = c["a"][c["ok"]], c["team"][c["ok"]]
    for k, v in enumerate(VARS):
        ax = axs[i, k]
        col = a[:, k]
        f = fits[(m, v)]
        ax.hist(col, bins=f["bins"], density=True, color="0.8", ec="0.45", lw=0.4)
        xs = np.linspace(col.min() - f["sd"], col.max() + f["sd"], 300)
        mix = sum((team == t).mean() * stats.norm.pdf(xs, col[team == t].mean(),
                                                      col[team == t].std(ddof=1))
                  for t in TEAMS)
        ax.plot(xs, stats.norm.pdf(xs, f["mean"], f["sd"]), "k-", lw=0.9)
        ax.plot(xs, mix, "--", color="tab:blue", lw=0.9)
        ax.set_title("$p$ " + ("< 0.001" if f["p"] < 0.001 else f"= {f['p']:.3f}"),
                     fontsize=6.5, pad=2)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(labelsize=6.5)
        ax.grid(False)
        if i == len(MOTIONS) - 1:
            ax.set_xlabel(f"{VAR_TEX[v]} [{VAR_UNIT[v]}]")
    axs[i, 0].set_ylabel(m)
handles = [axs[0, 0].plot([], [], "k-", lw=1, label="Fitted Gaussian")[0],
           axs[0, 0].plot([], [], "--", color="tab:blue", lw=1,
                          label="Mixture of the teams' Gaussians")[0]]
fig.legend(handles=handles, fontsize=7, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=1.2)
fig.savefig(FIG / "histograms.pdf", bbox_inches="tight")
plt.close(fig)

# d) PCA: inliers in the principal axes with their 1, 2, 3 sigma ellipses, same scale.
# One column wide: 2 x 2 grid, the fourth cell holds the legend.
half = max(np.abs(pca[m]["proj"]).max() for m in MOTIONS) * 1.1
fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.6))
for ax, m in zip(axs.flat, MOTIONS):
    p = pca[m]
    for t in DRAW:
        sel = p["team"] == t
        ax.plot(p["proj"][sel, 0], p["proj"][sel, 1], ".", ms=2.5, color=COLOR[t])
    for s, ls in zip((1, 2, 3), ("-", "--", ":")):
        ax.add_patch(Ellipse((0, 0), 2 * s * np.sqrt(p["lam"][0]), 2 * s * np.sqrt(p["lam"][1]),
                             fill=False, ec="k", lw=0.8, ls=ls))
    ax.set_aspect("equal")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_title(f"{TITLE[m]} (PC1 at ${p['angle']:.0f}" + r"^\circ$)", fontsize=7)
    ax.set_xlabel("PC1 [cm]")
    ax.set_ylabel("PC2 [cm]")
axs[1, 1].axis("off")
handles = team_handles(axs[0, 0])
handles += [axs[0, 0].plot([], [], "k", ls=ls, lw=0.9, label=f"{s}$\\sigma$ ellipse")[0]
            for s, ls in zip((1, 2, 3), ("-", "--", ":"))]
axs[1, 1].legend(handles=handles, fontsize=7, loc="center", frameon=False)
fig.tight_layout(w_pad=0.8, h_pad=0.8)
fig.savefig(FIG / "pca.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- summary
print(f"{macros['nOutAll']} of {macros['nRawAll']} runs removed as outliers ({macros['outByTeamText']}); "
      f"{macros['nRejected']} of {macros['nTests']} Gaussian fits rejected (alpha {ALPHA})")
for m in MOTIONS:
    print(f"  {m}: removed {macros[f'nOut{m}']}; "
          + ", ".join(f"p_{v} = {fits[(m, v)]['p']:.3f}" for v in (*VARS, "xy")))
