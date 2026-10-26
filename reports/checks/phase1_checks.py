"""Kaggle job: rebuild the Phase 1 pipeline from the repo code, then run the extra analyses for the report.

Inputs (attached Kaggle datasets): the NDWS TFRecords and a private dataset with the repo's src/ and reports/.
Outputs (/kaggle/working/out, small files only):
  env.txt                library versions
  repro_check.csv        rebuilt test metrics vs the committed reports/results.csv
  ci_events.csv          95% bootstrap intervals of PR AUC, resampling fire events (Sachin's event grouping)
  ci_patches.csv         the same, resampling patches
  ci_diffs.csv           intervals of paired PR AUC differences between models
  calibration.csv/.png   XGBoost probabilities after undoing the subsampling/weighting shift vs observed fire rate
  learning_curve.csv/.png XGBoost with best_params on growing shares of the training patches
  trees_curve.csv/.png   train vs eval score as trees are added (best_params, up to 1000 trees)
  prob_maps_selected.png 4 larger example patches for the Results section
Every step uses the repo code (src/) unchanged; seeds are the repo's (0).
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
OUT = Path(os.environ.get("NDWS_OUT", "/kaggle/working/out"))
OUT.mkdir(parents=True, exist_ok=True)
WORK = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp")
REPO = WORK / "repo"


def log(msg):
    line = f"[{time.time() - T0:7.0f}s] {msg}"
    print(line, flush=True)
    with open(OUT / "log.txt", "a") as f:
        f.write(line + "\n")


def sh(*cmd):
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


# ---------------------------------------------------------------------------------------------
# 1. Repo code and data in place
REPO = Path(os.environ.get("NDWS_REPO", REPO))  # the launcher unpacks the repo code here
assert (REPO / "src" / "features.py").exists(), f"repo code missing in {REPO}"
(REPO / "data").mkdir(exist_ok=True)
tfr = glob.glob(os.environ.get("NDWS_INPUT", "/kaggle/input") + "/**/next_day_wildfire_spread_*.tfrecord", recursive=True)
assert len(tfr) == 19, f"expected 19 TFRecords, found {len(tfr)}"
for f in tfr:
    dst = REPO / "data" / Path(f).name
    if not dst.exists():
        os.symlink(f, dst)
shutil.copy(REPO / "reports" / "results.csv", OUT / "results_committed.csv")
log(f"repo at {REPO}, {len(tfr)} TFRecords linked")

import importlib.metadata as md  # noqa: E402

with open(OUT / "env.txt", "w") as f:
    for pkg in ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "scipy", "pyarrow", "matplotlib"]:
        try:
            f.write(f"{pkg}=={md.version(pkg)}\n")
        except md.PackageNotFoundError:
            f.write(f"{pkg} missing\n")
log("env: " + (OUT / "env.txt").read_text().replace("\n", " "))

# ---------------------------------------------------------------------------------------------
# 2. Rebuild data, features and baselines with the repo's own entry points
sh(sys.executable, "-m", "src.data")
sh(sys.executable, "-m", "src.features")
sh(sys.executable, "-m", "src.baselines")

# ---------------------------------------------------------------------------------------------
# 3. Refit the two tuned models with the committed best_params (as notebook 02 does), score as src.train does
os.chdir(REPO)
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.data import CHANNELS, SIZE, load_split  # noqa: E402
from src.evaluate import evaluate, metrics  # noqa: E402
from src.features import ALL_FEATURES, load_pixels, make_pixel_table  # noqa: E402

best = json.loads((REPO / "reports" / "best_params.json").read_text())
train, ev, test = load_pixels("train"), load_pixels("eval"), load_pixels("test")
y_test_map = test["label"].to_numpy().reshape(-1, SIZE, SIZE)
models = {}
for name, cls in (("xgboost", xgb.XGBClassifier), ("lightgbm", lgb.LGBMClassifier)):
    m = cls(**best[name]).fit(train[ALL_FEATURES], train["label"])
    thr = metrics(ev["label"], m.predict_proba(ev[ALL_FEATURES])[:, 1])["best_f1_threshold"]
    evaluate(y_test_map, m.predict_proba(test[ALL_FEATURES])[:, 1].reshape(-1, SIZE, SIZE), name, threshold=thr)
    models[name] = m
    log(f"refit {name} done")

new = pd.read_csv(REPO / "reports" / "results.csv").set_index("model")
old = pd.read_csv(OUT / "results_committed.csv").set_index("model")
cols = ["pr_auc", "roc_auc", "precision", "recall", "f1", "threshold", "recall_at_1fpr", "recall_at_5fpr"]
check = pd.concat({"committed": old.loc[new.index, cols], "rebuilt": new[cols]}, axis=1)
for c in cols:
    check[("abs_diff", c)] = (check[("rebuilt", c)] - check[("committed", c)]).abs()
check.to_csv(OUT / "repro_check.csv", float_format="%.4f")
new.to_csv(OUT / "results_rebuilt.csv", float_format="%.4f")
log("max |PR AUC diff| rebuilt vs committed: " + f"{check[('abs_diff', 'pr_auc')].max():.4f}")

# ---------------------------------------------------------------------------------------------
# 4. Bootstrap intervals (patches and fire events), paired differences
from joblib import Parallel, delayed  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

PREDS = REPO / "reports" / "preds"
MODELS = ["persistence", "logreg_raw", "rf_raw", "xgb_raw", "xgboost", "lightgbm"]
probs = {m: np.load(PREDS / f"{m}.npz")["y_prob"] for m in MODELS}
y_map = np.load(PREDS / "xgboost.npz")["y_true"]
n = len(y_map)

# event grouping: identical to notebooks/02_evaluation.ipynb, Task 6
X_test, _ = load_split("test")
elev_map = X_test[:, CHANNELS.index("elevation")]
prev_map = X_test[:, CHANNELS.index("PrevFireMask")]
th_map = X_test[:, CHANNELS.index("th")] % 360
vs_map = np.clip(X_test[:, CHANNELS.index("vs")], 0, None)
del X_test
K, STARTS = 16, (0, 24, 48)
blocks = {}
for a in range(n):
    for r0 in STARTS:
        for c0 in STARTS:
            block = elev_map[a, r0:r0 + K, c0:c0 + K]
            if block.std() > 0:
                blocks.setdefault(block.tobytes(), set()).add(a)
links = {i: set() for i in range(n)}
for b in range(n):
    for w in sliding_window_view(elev_map[b], (K, K)).reshape(-1, K, K):
        for a in blocks.get(w.tobytes(), ()):
            if a != b:
                links[a].add(b)
                links[b].add(a)
event_of = np.full(n, -1)
for start in range(n):
    if event_of[start] != -1:
        continue
    event_of[start] = start
    stack = [start]
    while stack:
        i = stack.pop()
        for j in links[i]:
            if event_of[j] == -1:
                event_of[j] = start
                stack.append(j)
events = np.unique(event_of)
log(f"{n} test patches -> {len(events)} events")

lab = y_map != -1
patch_y = [y_map[i][lab[i]].astype(np.int8) for i in range(n)]
patch_p = {m: [probs[m][i][lab[i]] for i in range(n)] for m in MODELS}
members = {e: np.flatnonzero(event_of == e) for e in events}
N_BOOT = 500
rng = np.random.default_rng(0)
patch_sets = [rng.integers(0, n, n) for _ in range(N_BOOT)]
event_sets = [np.concatenate([members[e] for e in rng.choice(events, len(events))]) for _ in range(N_BOOT)]


def ap_all(idx):
    y = np.concatenate([patch_y[i] for i in idx])
    return [average_precision_score(y, np.concatenate([patch_p[m][i] for i in idx])) for m in MODELS]


def ap_batch(batch):
    return [ap_all(idx) for idx in batch]  # one task per worker, so the data is sent to each worker once


for label, sets in (("patches", patch_sets), ("events", event_sets)):
    n_workers = os.cpu_count() or 1
    batches = [sets[k::n_workers] for k in range(n_workers)]
    out = Parallel(n_jobs=n_workers)(delayed(ap_batch)(b) for b in batches)
    res = np.array([row for b in out for row in b])  # (N_BOOT, models)
    point = {m: new.loc[m, "pr_auc"] for m in MODELS}
    ci = pd.DataFrame({"model": MODELS, "pr_auc": [point[m] for m in MODELS],
                       "lo95": np.percentile(res, 2.5, axis=0), "hi95": np.percentile(res, 97.5, axis=0)})
    ci.to_csv(OUT / f"ci_{label}.csv", index=False, float_format="%.4f")
    if label == "events":
        rows = []
        for a, b in (("xgboost", "lightgbm"), ("xgboost", "rf_raw"), ("lightgbm", "rf_raw"), ("rf_raw", "persistence")):
            d = res[:, MODELS.index(a)] - res[:, MODELS.index(b)]
            rows.append({"a": a, "b": b, "diff": point[a] - point[b], "lo95": np.percentile(d, 2.5),
                         "hi95": np.percentile(d, 97.5), "share_a_better": (d > 0).mean()})
        pd.DataFrame(rows).to_csv(OUT / "ci_diffs.csv", index=False, float_format="%.4f")
    log(f"bootstrap over {label} done")

# ---------------------------------------------------------------------------------------------
# 5. Calibration of the tuned XGBoost after undoing the prior shift (Theory section)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

y_all = y_map[lab].astype(int)
p_all = probs["xgboost"][lab]
full_counts = np.load(REPO / "data" / "processed" / "train.npz")["y"]
pi_train = (full_counts == 1).sum() / (full_counts != -1).sum()
kept_ratio = (train["label"] == 0).sum() / (train["label"] == 1).sum()
w = best["xgboost"]["scale_pos_weight"]
r = kept_ratio / ((1 - pi_train) / pi_train)
factor = w / r
p_corr = p_all / (p_all + factor * (1 - p_all))
edges = np.quantile(p_corr, np.linspace(0, 1, 11))
which = np.clip(np.digitize(p_corr, edges[1:-1]), 0, 9)
cal = pd.DataFrame([{"bin": b, "mean_corrected_p": p_corr[which == b].mean(), "observed_rate": y_all[which == b].mean(),
                     "pixels": int((which == b).sum())} for b in range(10)])
cal.to_csv(OUT / "calibration.csv", index=False, float_format="%.5f")
with open(OUT / "calibration_meta.json", "w") as f:
    json.dump({"pi_train": float(pi_train), "kept_neg_per_pos": float(kept_ratio), "w": w, "r": float(r),
               "w_over_r": float(factor), "threshold": float(new.loc["xgboost", "threshold"]),
               "threshold_corrected": float(new.loc["xgboost", "threshold"] / (new.loc["xgboost", "threshold"]
                                            + factor * (1 - new.loc["xgboost", "threshold"]))),
               "mean_corrected_p": float(p_corr.mean()), "test_fire_rate": float(y_all.mean())}, f, indent=2)
fig, ax = plt.subplots(figsize=(4, 3.6))
ax.plot([0, cal.mean_corrected_p.max()], [0, cal.mean_corrected_p.max()], "--", color="#868e96", label="perfect")
ax.plot(cal.mean_corrected_p, cal.observed_rate, "o-", color="#1971c2", label="XGBoost, corrected")
ax.set(xlabel="mean corrected probability (decile)", ylabel="observed fire rate", title="Calibration on test")
ax.legend()
fig.savefig(OUT / "calibration.png", dpi=200, bbox_inches="tight")
log("calibration done")

# ---------------------------------------------------------------------------------------------
# 6. Learning curve: best_params on growing shares of training patches
patches = train["patch"].unique()
rng = np.random.default_rng(0)
order = rng.permutation(patches)
ev_lab = ev[ev["label"] != -1]
rows = []
for share in (0.05, 0.1, 0.25, 0.5, 1.0):
    keep = set(order[: int(round(share * len(order)))])
    sub = train[train["patch"].isin(keep)]
    m = xgb.XGBClassifier(**best["xgboost"]).fit(sub[ALL_FEATURES], sub["label"])
    p_tr = m.predict_proba(sub[ALL_FEATURES])[:, 1]
    p_ev = m.predict_proba(ev_lab[ALL_FEATURES])[:, 1]
    rows.append({"share": share, "train_patches": len(keep), "train_rows": len(sub),
                 "train_roc_auc": roc_auc_score(sub["label"], p_tr), "eval_roc_auc": roc_auc_score(ev_lab["label"], p_ev),
                 "eval_pr_auc": average_precision_score(ev_lab["label"], p_ev)})
    log(f"learning curve share {share}: eval PR AUC {rows[-1]['eval_pr_auc']:.4f}")
lc = pd.DataFrame(rows)
lc.to_csv(OUT / "learning_curve.csv", index=False, float_format="%.4f")

# ---------------------------------------------------------------------------------------------
# 7. Trees curve: train vs eval as trees are added (best_params, up to 1000 trees, no early stopping)
es = ev_lab.sample(min(1_000_000, len(ev_lab)), random_state=0)
tr_s = train.sample(min(1_000_000, len(train)), random_state=0)
params = {**best["xgboost"], "n_estimators": 1000, "eval_metric": ["auc", "aucpr"]}
m = xgb.XGBClassifier(**params).fit(train[ALL_FEATURES], train["label"],
                                     eval_set=[(tr_s[ALL_FEATURES], tr_s["label"]), (es[ALL_FEATURES], es["label"])],
                                     verbose=False)
hist = m.evals_result()
tc = pd.DataFrame({"trees": np.arange(1, 1001), "train_auc": hist["validation_0"]["auc"],
                   "eval_auc": hist["validation_1"]["auc"], "train_aucpr_1to5": hist["validation_0"]["aucpr"],
                   "eval_aucpr": hist["validation_1"]["aucpr"]})
tc.to_csv(OUT / "trees_curve.csv", index=False, float_format="%.5f")
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
axes[0].plot(lc.train_rows / 1e6, lc.train_roc_auc, "o-", label="train (ROC AUC)")
axes[0].plot(lc.train_rows / 1e6, lc.eval_roc_auc, "o-", label="eval (ROC AUC)")
axes[0].set(xlabel="training rows (millions)", ylabel="ROC AUC", title="Learning curve")
axes[0].legend()
axes[1].plot(tc.trees, tc.train_auc, label="train (ROC AUC)")
axes[1].plot(tc.trees, tc.eval_auc, label="eval (ROC AUC)")
axes[1].axvline(best["xgboost"]["n_estimators"], ls="--", color="#868e96")
axes[1].set(xlabel="trees", title="Train vs eval as trees are added")
axes[1].legend()
fig.savefig(OUT / "learning_curves.png", dpi=200, bbox_inches="tight")
log("learning and trees curves done")

# ---------------------------------------------------------------------------------------------
# 8. Larger example maps for 4 patches named in the Results text (same plotting rules as notebook 02)
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

THR = new.loc["xgboost", "threshold"]
p_xgb = probs["xgboost"]
pred = p_xgb >= THR
GREY, FIRE, FP_C, FN_C = "#c8c8c8", "#e34948", "#eda100", "#2a78d6"
MASK = ListedColormap([GREY, "white", FIRE])
OUTC = ListedColormap([GREY, "white", FIRE, FP_C, FN_C])
pick = [i for i in (290, 515, 1071, 69) if i < n] or list(range(min(4, n)))
desc = {290: "fire creeps outward", 515: "false-positive ring around a dying fire",
        1071: "new fire away from day-t fire is missed", 69: "clouds hide most of the patch"}
fig, axes = plt.subplots(len(pick), 4, figsize=(9, 2.4 * len(pick)), layout="constrained", squeeze=False)
for rr, i in enumerate(pick):
    o = np.ones(y_map[i].shape, int)
    o[(y_map[i] == 1) & pred[i]] = 2
    o[(y_map[i] == 0) & pred[i]] = 3
    o[(y_map[i] == 1) & ~pred[i]] = 4
    o[y_map[i] == -1] = 0
    a = axes[rr]
    a[0].imshow(prev_map[i] + 1, cmap=MASK, vmin=0, vmax=2, interpolation="nearest", origin="lower")
    a[1].imshow(y_map[i] + 1, cmap=MASK, vmin=0, vmax=2, interpolation="nearest", origin="lower")
    im = a[2].imshow(p_xgb[i], cmap="Reds", vmin=0, vmax=1, interpolation="nearest", origin="lower")
    th = np.deg2rad(th_map[i])
    mt = np.arctan2(np.sin(th).mean(), np.cos(th).mean())
    dx, dy = -np.sin(mt), -np.cos(mt)
    a[2].arrow(32 - 9 * dx, 32 - 9 * dy, 18 * dx, 18 * dy, width=1.2, color="black", length_includes_head=True)
    a[3].imshow(o, cmap=OUTC, vmin=0, vmax=4, interpolation="nearest", origin="lower")
    for k, (ax, h) in enumerate(zip(a, ["Fire day t", "True fire day t+1", "XGBoost probability", "Thresholded"])):
        ax.set_xticks([]); ax.set_yticks([])
        if rr == 0:
            ax.set_title(h, fontsize=10)
    a[0].set_ylabel(f"patch {i}\n{desc.get(i, '')}", fontsize=8.5)
handles = [Patch(color=FIRE, label="fire / true positive"), Patch(color=FP_C, label="false positive"),
           Patch(color=FN_C, label="false negative"), Patch(color=GREY, label="unknown (-1)")]
fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9, frameon=False)
fig.colorbar(im, ax=axes, shrink=0.5, pad=0.01, label="P(fire on day t+1)")
fig.savefig(OUT / "prob_maps_selected.png", dpi=200)
log("selected maps done")

log("ALL DONE")
