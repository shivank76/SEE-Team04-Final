"""SEE Assignment 6 - statistical evaluation of the KUKA youBot placing experiment.

Input (from the Assignment 5 folder ../report_hw05, written by its code/process_youbot.py):
  data/combined_object.csv        final object poses of all teams (OptiTrack), one row per run
  data/combined_end_effector.csv  final end-effector poses of all teams (encoders)

Outputs (in this folder):
  results.tex          every number quoted in the report
  data/table5.tex      rows of Table 5 of the manual (mean, variance, accuracy, Gaussian test)
  data/tests.tex       rows of the significance tests (object and placing-pose effects)
  figures/*.pdf        positions with PCA ellipses, example histograms, effects

Run:  python3 code/evaluate_youbot.py     (from this folder). No randomness is used.
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
A5 = ROOT.parent / "report_hw05"
DATA, FIG = ROOT / "data", ROOT / "figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SIZES = ("large", "medium", "small")
MOTIONS = ("left", "straight", "right")
VARS = ("x", "y", "theta")
VAR_TEX = {"x": "$x$", "y": "$y$", "theta": r"$\theta$", "pc1": "PC1", "pc2": "PC2"}
VAR_UNIT = {"x": "cm", "y": "cm", "theta": "deg", "pc1": "cm", "pc2": "cm"}
ALPHA = 0.05            # significance level of all tests
WEEK1 = (1, 3)          # teams of the first lab week (Assignment 5); the others ran a week later

# Ground truth of the final object pose (manual, Table 1), world frame, m and rad, as in
# ../report_hw05/code/process_youbot.py; converted to cm and degrees.
GT_OBJECT = {"straight": (0.150, -0.212, -1.65), "left": (0.356, -0.283, -2.08),
             "right": (-0.064, -0.352, -1.13)}
GT = {m: np.array([100 * x, 100 * y, np.degrees(t)]) for m, (x, y, t) in GT_OBJECT.items()}


# ---------------------------------------------------------------- input
def read(path):
    """{(size, motion): {"a": n x 3 array of x [cm], y [cm], theta [deg], "team": n}}, used
    poses only."""
    out = {}
    for r in csv.DictReader(open(path)):
        if r["used"] != "yes":
            continue
        g = out.setdefault((r["object"], r["motion"]), {"a": [], "team": []})
        g["a"].append([float(r["x_cm"]), float(r["y_cm"]), float(r["theta_deg"])])
        g["team"].append(int(r["team"]))
    for g in out.values():
        g["a"], g["team"] = np.array(g["a"]), np.array(g["team"])
    return out


obj = read(A5 / "data" / "combined_object.csv")
ee = read(A5 / "data" / "combined_end_effector.csv")
GROUPS = [(s, m) for s in SIZES for m in MOTIONS]


def tex_num(v, d=2):
    """Number with d decimals, minus sign for LaTeX (text and math mode), no '-0.00'."""
    v = 0.0 if round(v, d) == 0 else v
    s = f"{v:.{d}f}"
    return s.replace("-", r"\ensuremath{-}") if s.startswith("-") else s


def sig(v, n=2):
    """v with n significant figures (variances)."""
    if v == 0:
        return "0"
    d = max(0, n - 1 - int(np.floor(np.log10(abs(v)))))
    return tex_num(v, d)


def p_text(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


macros = {"alphaLevel": f"{ALPHA:g}", "nPoses": str(sum(len(g["a"]) for g in obj.values())),
          "nGroups": str(len(GROUPS)),
          "nMin": str(min(len(g["a"]) for g in obj.values())),
          "nMax": str(max(len(g["a"]) for g in obj.values())),
          "nTeams": str(len({t for g in obj.values() for t in g["team"]})),
          "numpyVersion": np.__version__, "scipyVersion": scipy.__version__,
          "mplVersion": matplotlib.__version__,
          "pythonVersion": ".".join(map(str, sys.version_info[:3]))}


# ---------------------------------------------------------------- 1) Gaussian test (chi^2)
def chi2_gauss(v, n_par=2):
    """Chi-square goodness-of-fit test of v against the Gaussian N(mean, sd) fitted to v
    (Lecture 5), as in Assignment 3: bins of equal probability under the fit, n // 10 of them,
    so that every bin expects 10 values; degrees of freedom = bins - 1 - n_par."""
    n = len(v)
    k = max(5, n // 10)
    inner = stats.norm.ppf(np.arange(1, k) / k, v.mean(), v.std(ddof=1))
    obs = np.bincount(np.searchsorted(inner, v), minlength=k)
    exp = n / k
    chi2 = ((obs - exp)**2 / exp).sum()
    return chi2, stats.chi2.sf(chi2, k - 1 - n_par)


# ---------------------------------------------------------------- 2) Table 5
fit = {}
rows = []
for s in SIZES:
    rows.append(rf"\multicolumn{{8}}{{l}}{{{s.capitalize()} object}}\\")
    for m in MOTIONS:
        a = obj[(s, m)]["a"]
        for k, v in enumerate(VARS):
            col = a[:, k]
            chi2, p = chi2_gauss(col)
            fit[(s, m, v)] = dict(mean=col.mean(), var=col.var(ddof=1),
                                  acc=col.mean() - GT[m][k], chi2=chi2, p=p)
            f = fit[(s, m, v)]
            rows.append(" & ".join([m.capitalize() if k == 0 else "", VAR_TEX[v],
                                    tex_num(f["mean"]), sig(f["var"]), tex_num(f["acc"]),
                                    f"{chi2:.1f}", p_text(p),
                                    "Reject" if p < ALPHA else "Fail to reject"]) + r"\\")
    if s != SIZES[-1]:
        rows.append(r"\midrule")
(DATA / "table5.tex").write_text("\n".join(rows) + "\n")

for v in VARS:
    rej = sum(fit[(s, m, v)]["p"] < ALPHA for s, m in GROUPS)
    macros[f"nRej{v.capitalize()}"] = str(rej)
macros["nRejAll"] = str(sum(f["p"] < ALPHA for f in fit.values()))
macros["nTests"] = str(len(fit))

# accuracy: distance of the mean position from the ground truth, and the theta error
dist = {g: np.hypot(fit[(*g, "x")]["acc"], fit[(*g, "y")]["acc"]) for g in GROUPS}
macros["accPosMin"] = f"{min(dist.values()):.1f}"
macros["accPosMax"] = f"{max(dist.values()):.1f}"
acc_th = [abs(fit[(*g, "theta")]["acc"]) for g in GROUPS]
macros["accThetaMax"] = f"{max(acc_th):.1f}"
for m in MOTIONS:
    macros[f"accPos{m.capitalize()}"] = f"{np.mean([dist[(s, m)] for s in SIZES]):.1f}"


# ---------------------------------------------------------------- 3) PCA of (x, y)
def pca(a):
    """Principal axes of the positions a (n x 2): eigenvectors of the covariance matrix,
    largest variance first, PC1 pointing to +x. Returns the scores (n x 2), the variances
    along PC1 and PC2, the eigenvectors (columns) and the PC1 angle from +x in degrees."""
    lam, vec = np.linalg.eigh(np.cov(a.T))
    lam, vec = lam[::-1], vec[:, ::-1]
    if vec[0, 0] < 0:
        vec[:, 0] *= -1
    if np.linalg.det(vec) < 0:              # PC2 is PC1 turned by +90 deg
        vec[:, 1] *= -1
    return (a - a.mean(axis=0)) @ vec, lam, vec, np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))


pc = {}
for g in GROUPS:
    a = obj[g]["a"]
    score, lam, vec, ang = pca(a[:, :2])
    pc[g] = dict(score=score, lam=lam, vec=vec, ang=ang, mean=a[:, :2].mean(axis=0))
sd = {(*g, v): (np.sqrt(pc[g]["lam"][i]) if v in ("pc1", "pc2") else obj[g]["a"][:, 2].std(ddof=1))
      for g in GROUPS for i, v in enumerate(("pc1", "pc2", "theta"))}
macros["sdPcOneMin"] = f"{min(sd[(*g, 'pc1')] for g in GROUPS):.2f}"
macros["sdPcOneMax"] = f"{max(sd[(*g, 'pc1')] for g in GROUPS):.2f}"
macros["sdPcTwoMin"] = f"{min(sd[(*g, 'pc2')] for g in GROUPS):.2f}"
macros["sdPcTwoMax"] = f"{max(sd[(*g, 'pc2')] for g in GROUPS):.2f}"
macros["sdThetaMin"] = f"{min(sd[(*g, 'theta')] for g in GROUPS):.1f}"
macros["sdThetaMax"] = f"{max(sd[(*g, 'theta')] for g in GROUPS):.1f}"
macros["sdThetaSmallMax"] = f"{max(sd[('small', m, 'theta')] for m in ('left', 'right')):.1f}"
macros["sdThetaBigMin"] = f"{min(sd[(s, m, 'theta')] for s in ('large', 'medium') for m in MOTIONS):.1f}"
macros["pcaAngMin"] = f"{min(pc[g]['ang'] % 180 for g in GROUPS):.0f}"
macros["pcaAngMax"] = f"{max(pc[g]['ang'] % 180 for g in GROUPS):.0f}"


# ---------------------------------------------------------------- 4) significance tests
# Precision: Levene's test (variances, median-centred, robust to non-Gaussian data) on the
# PC1 and PC2 scores and on theta. Accuracy: Kruskal-Wallis test (medians, on ranks) on the
# errors x - x_GT, y - y_GT, theta - theta_GT. Object effect: the three objects at one place;
# placing-pose effect: the three places for one object.
def samples(groups, v):
    out = []
    for g in groups:
        if v in ("pc1", "pc2"):
            out.append(pc[g]["score"][:, 0 if v == "pc1" else 1])
        else:
            k = VARS.index(v)
            out.append(obj[g]["a"][:, k] - GT[g[1]][k])
    return out


comparisons = [("object", m, [(s, m) for s in SIZES]) for m in MOTIONS] + \
              [("place", s, [(s, m) for m in MOTIONS]) for s in SIZES]
test = {}
for effect, fixed, groups in comparisons:
    for v in ("pc1", "pc2", "theta"):
        test[(effect, fixed, "prec", v)] = stats.levene(*samples(groups, v))
    for v in VARS:
        test[(effect, fixed, "acc", v)] = stats.kruskal(*samples(groups, v))

rows = []
for effect, fixed, groups in comparisons:
    if effect == "place" and fixed == SIZES[0]:
        rows.append(r"\midrule")
    if fixed in (MOTIONS[0], SIZES[0]):
        head = "Objects compared, at place" if effect == "object" else \
            "Places compared, for object"
        rows.append(rf"\multicolumn{{7}}{{l}}{{{head}}}\\")
    cells = [p_text(test[(effect, fixed, "prec", v)].pvalue) for v in ("pc1", "pc2", "theta")]
    cells += [p_text(test[(effect, fixed, "acc", v)].pvalue) for v in VARS]
    rows.append(" & ".join([fixed.capitalize()] + cells) + r"\\")
(DATA / "tests.tex").write_text("\n".join(rows) + "\n")


def n_sig(effect, kind):
    keys = [k for k in test if k[0] == effect and k[2] == kind]
    return sum(test[k].pvalue < ALPHA for k in keys), len(keys)


for effect in ("object", "place"):
    for kind in ("prec", "acc"):
        n, tot = n_sig(effect, kind)
        macros[f"nSig{effect.capitalize()}{kind.capitalize()}"] = str(n)
        macros["nPerEffect"] = str(tot)


def spread(effect, values):
    """Largest difference between the groups of one comparison, over all comparisons of an
    effect: values maps a group (size, motion) to a number."""
    return max(max(values[g] for g in groups) - min(values[g] for g in groups)
               for e, _, groups in comparisons if e == effect)


eff = {}
for effect in ("object", "place"):
    E = effect.capitalize()
    for v in VARS:
        eff[(effect, "acc", v)] = spread(effect, {g: fit[(*g, v)]["acc"] for g in GROUPS})
    for v in ("pc1", "pc2", "theta"):
        eff[(effect, "sd", v)] = spread(effect, {g: sd[(*g, v)] for g in GROUPS})
    macros[f"accPos{E}"] = f"{max(eff[(effect, 'acc', 'x')], eff[(effect, 'acc', 'y')]):.1f}"
    macros[f"accTheta{E}"] = f"{eff[(effect, 'acc', 'theta')]:.1f}"
    macros[f"sdPos{E}"] = f"{max(eff[(effect, 'sd', 'pc1')], eff[(effect, 'sd', 'pc2')]):.2f}"
    macros[f"sdTheta{E}"] = f"{eff[(effect, 'sd', 'theta')]:.1f}"
# The text states which variation has the larger effect; check that it still holds.
assert max(eff[("place", "acc", v)] for v in ("x", "y")) > \
       max(eff[("object", "acc", v)] for v in ("x", "y"))
assert max(eff[("place", "sd", v)] for v in ("pc1", "pc2")) > \
       max(eff[("object", "sd", v)] for v in ("pc1", "pc2"))
assert eff[("object", "sd", "theta")] > eff[("place", "sd", "theta")]

# lab weeks: the example of the histograms (large object, left)
EX = ("large", "left")
a, team = obj[EX]["a"], obj[EX]["team"]
w1 = np.isin(team, WEEK1)
macros["weekGapX"] = f"{abs(a[w1, 0].mean() - a[~w1, 0].mean()):.1f}"
macros["weekGapY"] = f"{abs(a[w1, 1].mean() - a[~w1, 1].mean()):.1f}"

# end effector: x and y are the same in every run (encoder pose rounded to 1 mm)
macros["eeSdXyMax"] = f"{max(ee[g]['a'][:, :2].std(ddof=1, axis=0).max() for g in ee):.1f}"

with open(ROOT / "results.tex", "w") as f:
    f.write("% Generated by code/evaluate_youbot.py - do not edit by hand\n")
    for k, v in macros.items():
        assert re.fullmatch(r"[A-Za-z]+", k), k
        v = v.replace("-", r"\ensuremath{-}", 1) if v.startswith("-") else v
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "pdf.fonttype": 42, "axes.grid": True,
                     "grid.alpha": 0.4})
OBJ_COLOR = {"large": "#1b9e77", "medium": "#d95f02", "small": "#7570b3"}
OBJ_MARK = {"large": "o", "medium": "s", "small": "^"}
TITLE = {"left": "Left", "straight": "Straight", "right": "Right"}

# a) positions per place with the 2 sigma ellipse of each object (axes along PC1, PC2)
half = 0.0
for m in MOTIONS:
    pts = np.vstack([obj[(s, m)]["a"][:, :2] for s in SIZES] + [GT[m][:2]])
    half = max(half, (pts.max(axis=0) - pts.min(axis=0)).max() / 2 * 1.1)
fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.5))
axs = axs.ravel()
for ax, m in zip(axs, MOTIONS):
    pts = np.vstack([obj[(s, m)]["a"][:, :2] for s in SIZES] + [GT[m][:2]])
    c0 = (pts.max(axis=0) + pts.min(axis=0)) / 2
    for s in SIZES:
        a, p = obj[(s, m)]["a"], pc[(s, m)]
        ax.plot(a[:, 0], a[:, 1], OBJ_MARK[s], ms=2.2, mfc="none", mew=0.5,
                color=OBJ_COLOR[s], alpha=0.7)
        ax.add_patch(Ellipse(p["mean"], 4 * np.sqrt(p["lam"][0]), 4 * np.sqrt(p["lam"][1]),
                             angle=p["ang"], fill=False, ec=OBJ_COLOR[s], lw=1.1))
    ax.plot(*GT[m][:2], "k+", ms=9, mew=1.4)
    ax.set_aspect("equal")
    ax.set_xlim(c0[0] - half, c0[0] + half)
    ax.set_ylim(c0[1] - half, c0[1] + half)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_title(TITLE[m])
    ax.set_xlabel("$x$ [cm]")
    ax.set_ylabel("$y$ [cm]")
axs[3].axis("off")
h = [axs[3].plot([], [], OBJ_MARK[s], ms=5, mfc="none", color=OBJ_COLOR[s],
                 label=f"{s.capitalize()} object")[0] for s in SIZES]
h += [axs[3].plot([], [], "-", color="0.3", lw=1.1, label=r"2$\sigma$ ellipse (PCA)")[0],
      axs[3].plot([], [], "k+", ms=8, mew=1.3, label="Ground truth")[0]]
axs[3].legend(handles=h, fontsize=7.5, loc="center", frameon=False, handletextpad=0.4)
fig.tight_layout(w_pad=0.8, h_pad=0.8)
fig.savefig(FIG / "positions.pdf", bbox_inches="tight")
plt.close(fig)

# b) example histogram (large object, left, y) split by lab week, with the fitted Gaussian
fig, ax = plt.subplots(figsize=(3.5, 1.15))
a, team = obj[EX]["a"], obj[EX]["team"]
w1 = np.isin(team, WEEK1)
col = a[:, 1]
bins = np.linspace(col.min(), col.max(), 21)
ax.hist([col[w1], col[~w1]], bins=bins, stacked=True, density=True,
        color=["tab:gray", "tab:cyan"], edgecolor="white", linewidth=0.3,
        label=["Teams 1, 3\n(week 1)", "Teams 2, 4, 5\n(week 2)"])
xs = np.linspace(col.min(), col.max(), 200)
ax.plot(xs, stats.norm.pdf(xs, col.mean(), col.std(ddof=1)), "k-", lw=1.0,
        label="Fitted Gaussian")
ax.set_xlabel("$y$ [cm]")
ax.set_ylabel("Density")
ax.set_yticks([])
ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
ax.grid(False)
ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
          handlelength=1.2)
fig.tight_layout()
fig.savefig(FIG / "histograms.pdf", bbox_inches="tight")
plt.close(fig)

# c) effects: accuracy (left column) and SD (right column) per place, bars per object
fig, axs = plt.subplots(3, 2, figsize=(3.5, 4.1), sharex=True)
xpos = np.arange(len(MOTIONS))
wbar = 0.26
for r, (va, vs) in enumerate((("x", "pc1"), ("y", "pc2"), ("theta", "theta"))):
    for j, s in enumerate(SIZES):
        off = (j - 1) * wbar
        axs[r, 0].bar(xpos + off, [fit[(s, m, va)]["acc"] for m in MOTIONS], wbar,
                      color=OBJ_COLOR[s], label=f"{s.capitalize()}")
        axs[r, 1].bar(xpos + off, [sd[(s, m, vs)] for m in MOTIONS], wbar, color=OBJ_COLOR[s])
    axs[r, 0].axhline(0, color="k", lw=0.6)
    axs[r, 0].set_ylabel(f"Acc. {VAR_TEX[va]} [{VAR_UNIT[va]}]")
    axs[r, 1].set_ylabel(f"SD {VAR_TEX[vs]} [{VAR_UNIT[vs]}]")
for ax in axs[-1]:
    ax.set_xticks(xpos)
    ax.set_xticklabels(["L", "S", "R"])
for ax in axs.flat:
    ax.grid(axis="x", visible=False)
axs[0, 0].set_title("Accuracy (mean $-$ GT)")
axs[0, 1].set_title("Precision (SD)")
fig.legend(*axs[0, 0].get_legend_handles_labels(), fontsize=7.5, loc="lower center", ncol=3,
           frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.8, h_pad=0.5)
fig.savefig(FIG / "effects.pdf", bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------- summary
print(f"{macros['nPoses']} object poses in {len(GROUPS)} groups; Gaussian rejected in "
      f"{macros['nRejAll']} of {macros['nTests']} tests")
for (effect, fixed, kind, v), r in test.items():
    print(f"  {effect:6s} {fixed:8s} {kind:4s} {v:5s} stat {r.statistic:7.1f}  p {r.pvalue:.3g}")
print({k: round(v, 2) for k, v in ((" ".join(k), v) for k, v in eff.items())})
