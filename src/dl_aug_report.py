"""Augmentation ablation for the focal U-Net: table, paired event bootstrap and figure.

Run after the training runs (python -m src.train_dl ... --aug {none,flip,rot,flip_rot} --seed {0,1,2}):
  python -m src.dl_aug_report            # table + figure from results_dl.csv and the logs (no data needed)
  python -m src.dl_aug_report --boot     # also paired bootstrap over fire events (needs reports/dl/preds/*.npz)
Runs are named unet_focal_aug_{mode}_s{seed}; unet_focal_aug_nowind is flip_rot with th left unturned.
All runs, the no-augmentation baseline included, are trained on the same machine, because a same-seed rerun on
other hardware moves test PR AUC by up to 0.02 (reports/dl/checks/reruns.csv).

Rule, fixed before the runs (as for the loss sweep): the mode with the highest mean eval PR AUC (best epoch) over
seeds 0-2 is used for the final U-Net, and `none` stays unless that mode beats it by more than one seed standard
deviation (the larger of the two). Test results are reported but never used to choose.

Outputs: reports/dl/checks/aug_ablation.csv, aug_ablation_diffs.csv (--boot), reports/dl/figures/aug_ablation.png.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DL = ROOT / "reports" / "dl"
CHECKS, FIGS, LOGS = DL / "checks", DL / "figures", DL / "checks" / "logs"

_spec = importlib.util.spec_from_file_location("dl_checks", CHECKS / "dl_checks.py")
dlc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dlc)  # reuse the log reader, plot style and bootstrap of the loss sweep

MODES = ["none", "flip", "rot", "flip_rot"]
NOWIND = "unet_focal_aug_nowind"
SEEDS = (0, 1, 2)
METRICS = ["pr_auc", "recall_at_1fpr", "recall_at_5fpr", "roc_auc", "f1"]
COLORS = {"none": "#868E96", "flip": "#009E73", "rot": "#0072B2", "flip_rot": "#D55E00", "nowind": "#CC79A7"}


def run_name(mode, seed):
    return f"unet_focal_aug_{mode}_s{seed}"


def run_row(res, name):
    curve = dlc.read_log(LOGS / f"{name}.log")
    return {"run": name, **{m: res.loc[name, m] for m in METRICS},
            "eval_pr_auc": curve.max(), "best_epoch": int(curve.argmax() + 1), "epochs_run": len(curve)}


def table():
    """One row per mode (mean and std over seeds) plus the nowind run; also returns the per-run rows."""
    res = pd.read_csv(DL / "results_dl.csv").drop_duplicates("model", keep="last").set_index("model")
    runs = [dict(mode=mode, seed=s, **run_row(res, run_name(mode, s)))
            for mode in MODES for s in SEEDS if run_name(mode, s) in res.index]
    if NOWIND in res.index:
        runs.append(dict(mode="nowind", seed=0, **run_row(res, NOWIND)))
    runs = pd.DataFrame(runs)
    cols = ["eval_pr_auc", *METRICS, "best_epoch"]
    g = runs.groupby("mode", sort=False)[cols]
    summary = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).join(g.size().rename("n_seeds"))
    summary.to_csv(CHECKS / "aug_ablation.csv", float_format="%.4f")
    return summary, runs


def decide(summary):
    s = summary.loc[[m for m in MODES if m in summary.index]]
    best = s["eval_pr_auc_mean"].idxmax()
    margin = s.loc[best, "eval_pr_auc_mean"] - s.loc["none", "eval_pr_auc_mean"]
    noise = max(s.loc[best, "eval_pr_auc_std"], s.loc["none", "eval_pr_auc_std"])
    return (best if best != "none" and margin > noise else "none"), best, margin, noise


def markdown(summary):
    def ms(mode, m):
        r = summary.loc[mode]
        sd = r[f"{m}_std"]
        return f"{r[f'{m}_mean']:.4f}" + ("" if np.isnan(sd) else f" ± {sd:.4f}")

    label = {"nowind": "flip_rot, th not turned (seed 0)"}
    lines = ["| `--aug` | Seeds | Eval PR AUC | Test PR AUC | Recall @ 1% FPR | Recall @ 5% FPR |",
             "|---|---:|---:|---:|---:|---:|"]
    for mode in summary.index:
        lines.append(f"| {label.get(mode, mode)} | {int(summary.loc[mode, 'n_seeds'])} | "
                     + " | ".join(ms(mode, m) for m in ["eval_pr_auc", "pr_auc", "recall_at_1fpr", "recall_at_5fpr"])
                     + " |")
    return "\n".join(lines)


def paired_bootstrap(runs):
    """Test PR AUC differences over the same fire-event resamples: each mode vs none per seed, nowind vs flip_rot."""
    from src.data import CHANNELS, load_split

    X, _ = load_split("test")
    events = dlc.fire_events(X[:, CHANNELS.index("elevation")])
    del X
    w = dlc.resample_weights(len(events), events)
    rows = []
    pairs = [(run_name(m, s), run_name("none", s)) for m in MODES[1:] for s in SEEDS]
    pairs.append((run_name("flip_rot", 0), NOWIND))
    for a, b in pairs:
        if not all((DL / "preds" / f"{n}.npz").exists() for n in (a, b)):
            continue
        boot = dlc.bootstrap({a: dlc.load_pred(a), b: dlc.load_pred(b)}, w)
        rows.append(dlc.diff_table(boot, [(a, b)]).iloc[0])
    out = pd.DataFrame(rows)
    out.to_csv(CHECKS / "aug_ablation_diffs.csv", index=False, float_format="%.4f")
    return out


def plot(summary, runs):
    plt = dlc.plt_setup()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.3})
    for mode in MODES:
        c = dlc.read_log(LOGS / f"{run_name(mode, 0)}.log")
        if c is not None:
            a.plot(np.arange(1, len(c) + 1), c, "-o", ms=2.5, lw=1.4, color=COLORS[mode], label=mode)
            a.plot(c.argmax() + 1, c.max(), "o", ms=6, mfc="none", color=COLORS[mode])
    c = dlc.read_log(LOGS / f"{NOWIND}.log")
    if c is not None:
        a.plot(np.arange(1, len(c) + 1), c, "--", lw=1.2, color=COLORS["nowind"], label="flip_rot, th not turned")
    a.set(xlabel="epoch", ylabel="eval PR AUC", title="(a) Eval PR AUC per epoch, seed 0")
    a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    a.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

    order = [m for m in [*MODES, "nowind"] if m in summary.index]
    ticks = {"nowind": "flip_rot\nth fixed", "flip_rot": "flip_rot"}
    for i, mode in enumerate(order):
        r, pts = summary.loc[mode], runs.loc[runs["mode"] == mode, "pr_auc"]
        if r["n_seeds"] > 1:
            b.errorbar(i, r["pr_auc_mean"], yerr=r["pr_auc_std"], fmt="o", ms=5, color=COLORS[mode], capsize=3, lw=1.2)
        b.scatter(np.full(len(pts), i + 0.2), pts, s=12, marker="D", color=COLORS[mode], alpha=0.7,
                  facecolors="none" if mode == "nowind" else COLORS[mode])
    b.axhline(summary.loc["none", "pr_auc_mean"], ls=":", lw=0.9, color=COLORS["none"])
    b.set_xticks(range(len(order)), [ticks.get(m, m) for m in order])
    b.set(ylabel="test PR AUC", title="(b) Test PR AUC: mean ± std, seeds as diamonds")
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "aug_ablation.png", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boot", action="store_true", help="paired bootstrap over fire events (needs predictions)")
    a = p.parse_args()
    summary, runs = table()
    print(runs.round(4).to_string(index=False), "\n")
    print(markdown(summary), "\n")
    chosen, best, margin, noise = decide(summary)
    print(f"highest mean eval PR AUC: {best} ({margin:+.4f} vs none, seed std {noise:.4f}) -> use --aug {chosen}")
    if a.boot:
        print(paired_bootstrap(runs).round(4).to_string(index=False))
    plot(summary, runs)


if __name__ == "__main__":
    main()
