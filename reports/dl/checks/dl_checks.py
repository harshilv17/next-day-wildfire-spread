"""Phase 2 checks on saved prediction maps (N, 64, 64): the loss sweep and the technical validation.

Run from the repo root:
  python reports/dl/checks/dl_checks.py selfcheck   # fast bootstrap AP == sklearn, on random data
  python reports/dl/checks/dl_checks.py sweep       # loss sweep table, paired bootstrap, figure
  python reports/dl/checks/dl_checks.py validate    # seeds, bootstrap CIs, calibration, overlap, reruns,
                                                    # learning curve (each check runs if its inputs exist)
  python reports/dl/checks/dl_checks.py plots       # redraw the figures from the CSVs and logs (no data)
Inputs:
  reports/dl/checks/runs.csv     test metrics of every run (results_dl.csv as written by train_dl on Kaggle);
                                 falls back to reports/dl/results_dl.csv
  reports/dl/checks/logs/        train_dl output per run ({name}.log), for the per-epoch eval curves
  reports/dl/preds/{name}.npz    test predictions from train_dl (not in git; shared by zip)
  reports/preds/xgboost.npz      Phase 1 XGBoost test predictions (not in git; shared by zip)
  data/processed/{train,test}.npz   python -m src.data
Outputs: reports/dl/checks/*.csv, *.json and reports/dl/figures/*.png.

Run names follow {model}_{loss}[_s{seed}], same-seed reruns end in _rerun, learning-curve runs in _frac{pct}.
Phase 1 conventions are kept: test is scored once, thresholds and choices come from eval, and confidence
intervals resample whole fire events (patches linked by matching elevation blocks, Phase 1 Appendix B).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
DL = ROOT / "reports" / "dl"
CHECKS, FIGS, PREDS = DL / "checks", DL / "figures", DL / "preds"
XGB_PREDS = ROOT / "reports" / "preds" / "xgboost.npz"
LOSSES = ("bce", "focal", "bce_dice")
METRICS = ["pr_auc", "roc_auc", "f1", "recall_at_1fpr", "recall_at_5fpr"]
N_BOOT = 1000
SEED = 0
PRIOR_FACTOR_XGB = 90.0  # Phase 1: 1:5 subsampling and scale_pos_weight 5 (Theory III-B)
COLORS = {"unet": "#0072B2", "cnn": "#009E73", "xgboost": "#E69F00", "persistence": "#999999",
          "bce": "#0072B2", "focal": "#D55E00", "bce_dice": "#009E73"}


# ---------------------------------------------------------------------------------------------
# Inputs
def load_pred(name):
    path = XGB_PREDS if name == "xgboost" else PREDS / f"{name}.npz"
    with np.load(path) as d:
        return d["y_true"], d["y_prob"]


def results():
    path = CHECKS / "runs.csv" if (CHECKS / "runs.csv").exists() else DL / "results_dl.csv"
    return pd.read_csv(path).set_index("model")


def read_log(path):
    """Per-epoch eval PR AUC from a train_dl log, or None if the log is missing."""
    if not Path(path).exists():
        return None
    scores = [float(m.group(1)) for m in re.finditer(r"^epoch\s+\d+\s+train loss \S+\s+eval PR AUC ([0-9.]+)",
                                                         Path(path).read_text(), re.M)]
    return np.array(scores) if scores else None


# ---------------------------------------------------------------------------------------------
# Fire events and fast bootstrap
def fire_events(elev):
    """Link test patches whose 16x16 elevation blocks match exactly; events are connected components.
    Identical to notebooks/02_evaluation.ipynb and reports/checks/phase1_checks.py."""
    n, k, starts = len(elev), 16, (0, 24, 48)
    blocks = {}
    for a in range(n):
        for r0 in starts:
            for c0 in starts:
                block = elev[a, r0:r0 + k, c0:c0 + k]
                if block.std() > 0:
                    blocks.setdefault(block.tobytes(), set()).add(a)
    links = {i: set() for i in range(n)}
    for b in range(n):
        for w in sliding_window_view(elev[b], (k, k)).reshape(-1, k, k):
            for a in blocks.get(w.tobytes(), ()):
                if a != b:
                    links[a].add(b)
                    links[b].add(a)
    event_of = np.full(n, -1)
    for start in range(n):
        if event_of[start] != -1:
            continue
        event_of[start], stack = start, [start]
        while stack:
            for j in links[stack.pop()]:
                if event_of[j] == -1:
                    event_of[j] = start
                    stack.append(j)
    return event_of


class PooledAP:
    """Average precision of the pooled labelled pixels when each patch carries a weight.

    With integer weights (how often a bootstrap resample drew the patch) this equals sklearn's
    average_precision_score on the concatenated resample. Recall only changes at score groups that hold
    fire, so precision is needed only there; at each such cut the weighted counts are
    (pixels of each patch up to the cut) x (patch weights), one matrix product for many resamples.
    """

    def __init__(self, y_map, p_map, patches=None):
        keep = y_map != -1
        patch = np.broadcast_to(np.arange(len(y_map))[:, None, None], y_map.shape)[keep]
        y, s = y_map[keep] == 1, p_map[keep].astype(np.float64)
        if patches is not None:  # restrict to a subset of patches (e.g. new locations)
            sel = np.isin(patch, patches)
            y, s, patch = y[sel], s[sel], patch[sel]
        order = np.argsort(s, kind="mergesort")[::-1]  # same tie handling as sklearn
        s, y, patch = s[order], y[order], patch[order]
        ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # last position of each tie group
        group = np.repeat(np.arange(len(ends)), np.diff(np.r_[-1, ends]))
        cuts = ends[np.unique(group[y])]  # tie groups that contain fire
        self.n_patches = len(y_map)
        self.all = np.zeros((len(cuts), self.n_patches))
        self.fire = np.zeros((len(cuts), self.n_patches))
        by_patch = np.argsort(patch, kind="stable")  # positions of each patch's pixels, in rank order
        bounds = np.r_[0, np.cumsum(np.bincount(patch, minlength=self.n_patches))]
        for q in range(self.n_patches):
            pos = by_patch[bounds[q]:bounds[q + 1]]
            if len(pos):
                self.all[:, q] = np.searchsorted(pos, cuts, side="right")
                self.fire[:, q] = np.searchsorted(pos[y[pos]], cuts, side="right")

    def many(self, weights, chunk=100):
        """AP for each row of weights (n_resamples, n_patches); NaN if the pixels hold no fire."""
        if len(self.fire) == 0:
            return np.full(len(weights), np.nan)
        out = []
        for i in range(0, len(weights), chunk):
            w = np.asarray(weights[i:i + chunk], np.float64).T
            tp, n = self.fire @ w, self.all @ w
            precision = np.divide(tp, n, out=np.zeros_like(tp), where=n > 0)
            recall = tp / tp[-1]
            out.append(np.sum(np.diff(recall, axis=0, prepend=0) * precision, axis=0))
        return np.concatenate(out)

    def __call__(self, patch_weight):
        return float(self.many(np.asarray(patch_weight)[None])[0])


def resample_weights(n_patches, groups=None, include=None, n_boot=N_BOOT, seed=SEED):
    """(n_boot, n_patches) multiplicities: patches drawn with replacement, or whole groups (fire events).
    include (bool mask) limits the draw to some patches; the others always get weight 0."""
    rng = np.random.default_rng(seed)
    pool = np.arange(n_patches) if include is None else np.flatnonzero(include)
    if groups is None:
        return np.stack([np.bincount(rng.choice(pool, len(pool)), minlength=n_patches)
                         for _ in range(n_boot)]).astype(np.float64)
    ids = np.unique(groups[pool])
    members = {g: pool[groups[pool] == g] for g in ids}
    return np.stack([np.bincount(np.concatenate([members[g] for g in rng.choice(ids, len(ids))]),
                                 minlength=n_patches) for _ in range(n_boot)]).astype(np.float64)


def bootstrap(maps, weights, patches=None):
    """{name: (point, array of n_boot APs)} with the same resamples for every model (paired)."""
    out = {}
    for name, (y, p) in maps.items():
        ap = PooledAP(y, p, patches)
        point = np.ones(len(y)) if patches is None else np.isin(np.arange(len(y)), patches).astype(float)
        out[name] = (ap(point), ap.many(weights))
    return out


def interval_table(boot):
    return pd.DataFrame([{"model": m, "pr_auc": pt, "lo95": np.percentile(b, 2.5), "hi95": np.percentile(b, 97.5)}
                         for m, (pt, b) in boot.items()])


def diff_table(boot, pairs):
    rows = []
    for a, b in pairs:
        d = boot[a][1] - boot[b][1]
        rows.append({"a": a, "b": b, "diff": boot[a][0] - boot[b][0], "lo95": np.percentile(d, 2.5),
                     "hi95": np.percentile(d, 97.5), "share_a_better": (d > 0).mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Plot style (same as reports/checks/plot_checks.py)
def plt_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": "#868E96", "axes.labelcolor": "#212529", "xtick.color": "#868E96",
        "ytick.color": "#868E96", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#E9ECEF", "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300,
    })
    return plt


def seed_names(base):
    return [base, f"{base}_s1", f"{base}_s2"]


# ---------------------------------------------------------------------------------------------
# Loss sweep
def sweep_table(logs):
    """Loss sweep table from runs.csv and the training logs (no data needed)."""
    res = results()
    rows = []
    for loss in LOSSES:
        name = f"unet_{loss}"
        curves = {s: read_log(Path(logs) / f"{s}.log") for s in seed_names(name)}
        evals = [c.max() for c in curves.values() if c is not None]
        tests = [res.loc[s, "pr_auc"] for s in seed_names(name) if s in res.index]
        c0 = curves[name]
        rows.append({"loss": loss, **{m: res.loc[name, m] for m in METRICS},
                     "eval_pr_auc": c0.max() if c0 is not None else np.nan,
                     "best_epoch": int(c0.argmax() + 1) if c0 is not None else np.nan,
                     "epochs_run": len(c0) if c0 is not None else np.nan,
                     "eval_pr_auc_mean": np.mean(evals) if evals else np.nan,
                     "eval_pr_auc_std": np.std(evals, ddof=1) if len(evals) > 1 else np.nan,
                     "test_pr_auc_mean": np.mean(tests),
                     "test_pr_auc_std": np.std(tests, ddof=1) if len(tests) > 1 else np.nan, "n_seeds": len(tests)})
    table = pd.DataFrame(rows)
    table.to_csv(CHECKS / "loss_sweep.csv", index=False, float_format="%.4f")
    return table


def sweep_bootstrap():
    """Event-level CIs of the seed-0 runs and their paired differences (needs predictions and test data)."""
    from src.data import CHANNELS, load_split

    X, _ = load_split("test")
    events = fire_events(X[:, CHANNELS.index("elevation")])
    boot = bootstrap({loss: load_pred(f"unet_{loss}") for loss in LOSSES}, resample_weights(len(events), events))
    interval_table(boot).to_csv(CHECKS / "loss_sweep_ci_events.csv", index=False, float_format="%.4f")
    diff_table(boot, [("focal", "bce"), ("bce_dice", "bce"), ("focal", "bce_dice")]).to_csv(
        CHECKS / "loss_sweep_diffs.csv", index=False, float_format="%.4f")


def plot_sweep(logs):
    plt = plt_setup()
    res = results()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.5, 1]})
    for loss in LOSSES:
        c = read_log(Path(logs) / f"unet_{loss}.log")
        if c is not None:
            a.plot(np.arange(1, len(c) + 1), c, "-o", ms=2.5, lw=1.4, color=COLORS[loss], label=loss)
            a.plot(c.argmax() + 1, c.max(), "o", ms=6, mfc="none", color=COLORS[loss])
    a.set(xlabel="epoch", ylabel="eval PR AUC", title="(a) eval curves, seed 0 (circle: kept epoch)")
    a.legend(frameon=False)
    ci = pd.read_csv(CHECKS / "loss_sweep_ci_events.csv").set_index("model")
    for i, loss in enumerate(LOSSES):
        seeds = [res.loc[s, "pr_auc"] for s in seed_names(f"unet_{loss}") if s in res.index]
        b.errorbar(i, ci.loc[loss, "pr_auc"], yerr=[[ci.loc[loss, "pr_auc"] - ci.loc[loss, "lo95"]],
                   [ci.loc[loss, "hi95"] - ci.loc[loss, "pr_auc"]]], fmt="o", color=COLORS[loss], capsize=3, lw=1.2)
        b.scatter(np.full(len(seeds), i + 0.18), seeds, s=10, color=COLORS[loss], alpha=0.6, marker="D")
    b.set_xticks(range(len(LOSSES)), LOSSES)
    b.set(ylabel="test PR AUC", title="(b) test, seed 0 \u00b1 95% CI; \u25c6 seeds")
    fig.savefig(FIGS / "loss_sweep.png", bbox_inches="tight")
    plt.close(fig)


def sweep(logs):
    CHECKS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    print(sweep_table(logs).round(4).to_string(index=False))
    sweep_bootstrap()
    plot_sweep(logs)


# ---------------------------------------------------------------------------------------------
# Technical validation
def seeds_table(families):
    res = results()
    rows = []
    for fam in families:
        names = [s for s in seed_names(fam) if s in res.index]
        if len(names) < 2:
            continue
        sub = res.loc[names, METRICS]
        rows.append({"model": fam, "n_seeds": len(names),
                     **{f"{m}_mean": sub[m].mean() for m in METRICS}, **{f"{m}_std": sub[m].std() for m in METRICS}})
    return pd.DataFrame(rows)


def calibration(maps, factors, pi_train):
    """Reliability by deciles of the prior-corrected probability, and Brier scores (raw and corrected)."""
    curves, summary = [], []
    for name, (y_map, p_map) in maps.items():
        keep = y_map != -1
        y, p = y_map[keep].astype(np.float64), p_map[keep].astype(np.float64)
        f = factors[name]
        pc = p / (p + f * (1 - p))
        ref = np.mean((pi_train - y) ** 2)  # always predicting the training fire rate
        edges = np.quantile(pc, np.linspace(0, 1, 11))
        which = np.clip(np.digitize(pc, edges[1:-1]), 0, 9)
        bins = pd.DataFrame({"y": y, "p": pc, "b": which}).groupby("b").agg(mean_p=("p", "mean"), observed=("y", "mean"),
                                                                              pixels=("y", "size"))
        ece = float(np.sum(bins.pixels / len(y) * (bins.mean_p - bins.observed).abs()))
        curves.append(bins.assign(model=name).reset_index())
        summary.append({"model": name, "prior_factor": f, "brier_raw": np.mean((p - y) ** 2),
                        "brier_corrected": np.mean((pc - y) ** 2), "brier_climatology": ref,
                        "bss_corrected": 1 - np.mean((pc - y) ** 2) / ref, "ece_corrected": ece,
                        "mean_corrected_p": pc.mean(), "observed_rate": y.mean()})
    return pd.concat(curves), pd.DataFrame(summary)


def validate(final, logs):
    from src.data import CHANNELS, load_split

    CHECKS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    res = results()
    out = {}

    # 1. seeds (XGBoost refits with three seeds, if xgb_seeds.csv was written next to this script)
    st = seeds_table(final)
    if (CHECKS / "xgb_seeds.csv").exists():
        x = pd.read_csv(CHECKS / "xgb_seeds.csv")
        st = pd.concat([st, pd.DataFrame([{"model": "xgboost", "n_seeds": len(x), **{f"{m}_mean": x[m].mean() for m in METRICS},
                                           **{f"{m}_std": x[m].std() for m in METRICS}}])], ignore_index=True)
    st.to_csv(CHECKS / "seeds.csv", index=False, float_format="%.4f")
    out["seeds"] = st

    # 2. bootstrap CIs (patches and events), DL vs XGBoost
    Xte, yte = load_split("test")
    elev_te = Xte[:, CHANNELS.index("elevation")]
    prev = Xte[:, CHANNELS.index("PrevFireMask")]
    events = fire_events(elev_te)
    maps = {m: load_pred(m) for m in final}
    if XGB_PREDS.exists():
        maps["xgboost"] = load_pred("xgboost")
    maps["persistence"] = (yte, np.clip(prev, 0, None))
    pairs = [(a, b) for a in final for b in ("xgboost", "persistence") if b in maps] + \
            [(a, b) for i, a in enumerate(final) for b in final[i + 1:]]
    for label, groups in (("patches", None), ("events", events)):
        boot = bootstrap(maps, resample_weights(len(events), groups))
        interval_table(boot).to_csv(CHECKS / f"ci_{label}.csv", index=False, float_format="%.4f")
        diff_table(boot, pairs).to_csv(CHECKS / f"ci_diffs_{label}.csv", index=False, float_format="%.4f")
        out[label] = boot

    # paired DL - XGBoost for every seed, so the gap is not a property of one seed
    if "xgboost" in maps:
        w = resample_weights(len(events), events)
        rows = []
        for fam in final:
            for s in seed_names(fam):
                if (PREDS / f"{s}.npz").exists():
                    b = bootstrap({s: load_pred(s), "xgboost": maps["xgboost"]}, w)
                    rows.append(diff_table(b, [(s, "xgboost")]).iloc[0])
        pd.DataFrame(rows).to_csv(CHECKS / "ci_diffs_seeds_vs_xgboost.csv", index=False, float_format="%.4f")

    # 3. calibration
    ytr = load_split("train")[1]
    pi_train = (ytr == 1).sum() / (ytr != -1).sum()
    pos_weight = (ytr == 0).sum() / (ytr == 1).sum()  # train_dl weights fire by this, no subsampling (r = 1)
    factors = {m: (PRIOR_FACTOR_XGB if m == "xgboost" else 1.0 if m == "persistence" else pos_weight) for m in maps}
    cal_maps = {m: v for m, v in maps.items() if m != "persistence"}
    curves, summary = calibration(cal_maps, factors, pi_train)
    curves.to_csv(CHECKS / "calibration_bins.csv", index=False, float_format="%.6f")
    summary.to_csv(CHECKS / "calibration.csv", index=False, float_format="%.6f")
    out["calibration"] = summary

    # 4. location overlap: test patches whose elevation grid equals a training patch
    Xtr = load_split("train")[0]
    tr_elev = {x.tobytes() for x in Xtr[:, CHANNELS.index("elevation")]}
    seen = np.array([e.tobytes() in tr_elev for e in elev_te])
    del Xtr
    rows = []
    for label, patches in (("all", None), ("new", np.flatnonzero(~seen)), ("seen", np.flatnonzero(seen))):
        for m, (y, p) in maps.items():
            w = np.ones(len(y)) if patches is None else np.isin(np.arange(len(y)), patches).astype(float)
            rows.append({"subset": label, "model": m, "patches": int(w.sum()), "pr_auc": PooledAP(y, p, patches)(w)})
    ov = pd.DataFrame(rows).pivot(index="model", columns="subset", values="pr_auc")
    ov.to_csv(CHECKS / "overlap.csv", float_format="%.4f")
    w_new = resample_weights(len(events), events, include=~seen)
    boot_new = bootstrap(maps, w_new, patches=np.flatnonzero(~seen))
    diff_table(boot_new, pairs).to_csv(CHECKS / "overlap_diffs_new_locations.csv", index=False, float_format="%.4f")
    json.dump({"test_patches": int(len(seen)), "seen_locations": int(seen.sum())},
              open(CHECKS / "overlap_counts.json", "w"), indent=2)
    out["overlap"] = ov

    # 5. same-seed reruns
    rows = []
    for name in res.index:
        if not name.endswith("_rerun") or name[:-6] not in res.index:
            continue
        base = name[:-6]
        a, b = load_pred(base)[1], load_pred(name)[1]
        ca, cb = (read_log(Path(logs) / f"{n}.log") if logs else None for n in (base, name))
        rows.append({"run": base, "identical_predictions": bool(np.array_equal(a, b)),
                     "max_abs_prob_diff": float(np.abs(a - b).max()), "mean_abs_prob_diff": float(np.abs(a - b).mean()),
                     **{f"d_{m}": res.loc[name, m] - res.loc[base, m] for m in METRICS},
                     "best_epoch": int(ca.argmax() + 1) if ca is not None else np.nan,
                     "best_epoch_rerun": int(cb.argmax() + 1) if cb is not None else np.nan})
    pd.DataFrame(rows).to_csv(CHECKS / "reruns.csv", index=False, float_format="%.6f")

    # 6. learning curve (runs named {final[0]}_frac{pct}; 100% is the seed-0 run)
    base = final[0]
    rows = []
    for name in sorted(res.index):
        m = re.fullmatch(rf"{re.escape(base)}_frac(\d+)", name)
        if m:
            c = read_log(Path(logs) / f"{name}.log") if logs else None
            rows.append({"share": int(m.group(1)) / 100, "run": name, "test_pr_auc": res.loc[name, "pr_auc"],
                         "eval_pr_auc": c.max() if c is not None else np.nan})
    if rows:
        c = read_log(Path(logs) / f"{base}.log") if logs else None
        rows.append({"share": 1.0, "run": base, "test_pr_auc": res.loc[base, "pr_auc"],
                     "eval_pr_auc": c.max() if c is not None else np.nan})
        pd.DataFrame(rows).sort_values("share").to_csv(CHECKS / "learning_curve.csv", index=False, float_format="%.4f")
    return out


# ---------------------------------------------------------------------------------------------
def family(name):
    return "unet" if name.startswith("unet") else "cnn" if name.startswith("cnn") else name


def plot_validation(final):
    """Figures from the CSVs written by validate(); skipped until those exist."""
    if not (CHECKS / "ci_events.csv").exists():
        return
    plt = plt_setup()
    ci = pd.read_csv(CHECKS / "ci_events.csv").set_index("model")
    res = results()
    order = [m for m in final + ["xgboost", "persistence"] if m in ci.index]
    fig, ax = plt.subplots(figsize=(3.5, 0.45 * len(order) + 0.7))
    for i, m in enumerate(order[::-1]):
        r = ci.loc[m]
        ax.errorbar(r.pr_auc, i, xerr=[[r.pr_auc - r.lo95], [r.hi95 - r.pr_auc]], fmt="o", capsize=3, lw=1.3,
                    color=COLORS[family(m)])
        seeds = [res.loc[s, "pr_auc"] for s in seed_names(m) if s in res.index] if m in final else []
        ax.scatter(seeds, np.full(len(seeds), i + 0.22), marker="D", s=9, color=COLORS[family(m)], alpha=0.6)
    ax.set_yticks(range(len(order)), order[::-1])
    ax.set(xlabel="test PR AUC", title="95% CI over fire events; \u25c6 = each seed")
    fig.savefig(FIGS / "dl_ci_events.png", bbox_inches="tight")
    plt.close(fig)

    bins = pd.read_csv(CHECKS / "calibration_bins.csv")
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    lims = (1e-4, 0.5)
    ax.plot(lims, lims, "--", lw=0.9, color="#868E96")
    for m, g in bins.groupby("model"):
        ax.plot(g.mean_p, g.observed, "-o", ms=3, lw=1.3, color=COLORS[family(m)], label=m)
    ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims, aspect="equal", xlabel="mean corrected probability",
           ylabel="observed fire rate", title="Reliability on test, deciles")
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=6.5)
    fig.savefig(FIGS / "dl_calibration.png", bbox_inches="tight")
    plt.close(fig)

    if (CHECKS / "learning_curve.csv").exists():
        lc = pd.read_csv(CHECKS / "learning_curve.csv")
        fig, ax = plt.subplots(figsize=(3.2, 2.3))
        ax.plot(lc.share, lc.eval_pr_auc, "-o", ms=3.5, lw=1.5, color=COLORS["unet"], label=final[0])
        p1 = ROOT / "reports" / "checks" / "learning_curve.csv"
        if p1.exists():
            x = pd.read_csv(p1)
            ax.plot(x.share, x.eval_pr_auc, "-o", ms=3.5, lw=1.5, color=COLORS["xgboost"], label="xgboost (Phase 1)")
        ax.set(xscale="log", xlabel="share of training patches", ylabel="eval PR AUC", title="Learning curve")
        ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1], ["5%", "10%", "25%", "50%", "100%"])
        ax.minorticks_off()
        ax.legend(frameon=False, fontsize=6.5)
        fig.savefig(FIGS / "dl_learning_curve.png", bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------------------------
def selfcheck():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    y = (rng.random((40, 8, 8)) < 0.1).astype(np.float32)
    y[rng.random(y.shape) < 0.05] = -1
    p = np.round(rng.random(y.shape), 2).astype(np.float32)  # rounding creates ties
    ap = PooledAP(y, p)
    keep = y != -1
    assert np.isclose(ap(np.ones(40)), average_precision_score(y[keep], p[keep]), rtol=0, atol=1e-12)
    for w in resample_weights(40, n_boot=20):
        idx = np.repeat(np.arange(40), w.astype(int))
        yy, pp = y[idx], p[idx]
        assert np.isclose(ap(w), average_precision_score(yy[yy != -1], pp[yy != -1]), rtol=0, atol=1e-12)
    groups = np.repeat(np.arange(10), 4)
    for w in resample_weights(40, groups, n_boot=5):  # an event is drawn whole: equal weights inside it
        assert all(len(set(w[groups == g])) == 1 for g in range(10)) and w.sum() == 40
    inc = np.arange(40) < 20
    assert all(w[~inc].sum() == 0 and w.sum() == 20 for w in resample_weights(40, include=inc, n_boot=5))
    print("dl_checks self-check ok: pooled bootstrap AP equals sklearn on resampled patches")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("check", choices=["selfcheck", "sweep", "validate", "plots"])
    p.add_argument("--logs", default=str(CHECKS / "logs"), help="folder with train_dl logs ({name}.log)")
    p.add_argument("--final", nargs="+", default=["unet_focal", "cnn_bce"], help="seed-0 runs of the final models")
    a = p.parse_args()
    if a.check == "selfcheck":
        selfcheck()
    elif a.check == "sweep":
        sweep(a.logs)
    elif a.check == "plots":  # figures and log-based tables only, from the CSVs in reports/dl/checks
        if (CHECKS / "loss_sweep_ci_events.csv").exists():
            sweep_table(a.logs)
            plot_sweep(a.logs)
        plot_validation(a.final)
    else:
        validate(a.final, a.logs)
        plot_validation(a.final)
