"""Report figures from the outputs of phase1_checks.py (reports/checks).

Run from the repo root: python reports/checks/plot_checks.py
Writes reports/figures/th_learning_curves.png and th_calibration.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

IN = Path("reports/checks")
FIG = Path("reports/figures")
TRAIN, EVAL, INK, MUTED, GRID = "#E8590C", "#1C7ED6", "#212529", "#868E96", "#E9ECEF"
N_TREES = 216  # reports/best_params.json, xgboost n_estimators

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300,
})


def label(ax, x, y, text, color, **kw):
    ax.annotate(text, (x, y), color=color, fontsize=7.5, fontweight="bold", **kw)


# Learning curve (more data) and trees curve (more trees), both in ROC AUC so train and eval are comparable
lc = pd.read_csv(IN / "learning_curve.csv")
tc = pd.read_csv(IN / "trees_curve.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True, gridspec_kw={"wspace": 0.12})
rows = lc.train_rows / 1e6
a.plot(rows, lc.train_roc_auc, "-o", color=TRAIN, lw=1.5, ms=3.5)
a.plot(rows, lc.eval_roc_auc, "-o", color=EVAL, lw=1.5, ms=3.5)
a.set(xscale="log", xlabel="training rows (millions)", ylabel="ROC AUC", title="(a) more data", ylim=(0.855, 0.98))
a.set_xticks([0.2, 0.5, 1, 2, 4], ["0.2", "0.5", "1", "2", "4"])
a.minorticks_off()
label(a, rows.iloc[0], lc.train_roc_auc.iloc[0] + 0.004, "train", TRAIN)
label(a, rows.iloc[0], lc.eval_roc_auc.iloc[0] - 0.012, "eval", EVAL)
b.plot(tc.trees, tc.train_auc, color=TRAIN, lw=1.5)
b.plot(tc.trees, tc.eval_auc, color=EVAL, lw=1.5)
b.axvline(N_TREES, ls="--", lw=0.9, color=MUTED)
b.text(N_TREES + 25, 0.862, f"used: {N_TREES}", color=MUTED, fontsize=7)
b.set(xlabel="trees", title="(b) more trees", xlim=(0, 1000))
label(b, 620, tc.train_auc.iloc[620] + 0.005, "train", TRAIN)
label(b, 620, tc.eval_auc.iloc[620] - 0.013, "eval", EVAL)
fig.savefig(FIG / "th_learning_curves.png", bbox_inches="tight")
plt.close(fig)

# Calibration after undoing the prior shift: deciles of the corrected score, log-log so all ten are visible
cal = pd.read_csv(IN / "calibration.csv")
fig, ax = plt.subplots(figsize=(2.6, 2.3))
lims = (1.5e-4, 0.15)
ax.plot(lims, lims, "--", lw=0.9, color=MUTED)
ax.text(0.03, 0.015, "perfect", color=MUTED, fontsize=7, rotation=38)
ax.plot(cal.mean_corrected_p, cal.observed_rate, "-o", color=EVAL, lw=1.5, ms=3.5)
ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims, aspect="equal",
       xlabel="mean corrected probability", ylabel="observed fire rate", title="Tuned XGBoost, test deciles")
ax.minorticks_off()
fig.savefig(FIG / "th_calibration.png", bbox_inches="tight")
plt.close(fig)

print("done")
