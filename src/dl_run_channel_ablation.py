"""Run U-Net channel-group ablation experiments sequentially.

Experiments:
  1. unet_base: all 12 channels (baseline)
  2. unet_no_weather: drop weather (4 channels)
  3. unet_no_vegetation: drop vegetation (11 channels)
  4. unet_no_terrain: drop terrain (11 channels)
  5. unet_no_fire: drop fire (11 channels)
  6. unet_no_human: drop human (11 channels)

Outputs:
  reports/dl/results_dl.csv
  reports/dl/channel_ablation.csv
  reports/dl/channel_ablation.png
  reports/dl/channel_ablation_summary.md
"""
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "dl"

EXPERIMENTS = [
    {"name": "unet_base", "group": None, "channels": 12, "desc": "All channels (baseline)"},
    {"name": "unet_no_weather", "group": "weather", "channels": 4, "desc": "No weather (th, vs, temp, precip, etc.)"},
    {"name": "unet_no_vegetation", "group": "vegetation", "channels": 11, "desc": "No vegetation (NDVI)"},
    {"name": "unet_no_terrain", "group": "terrain", "channels": 11, "desc": "No terrain (elevation)"},
    {"name": "unet_no_fire", "group": "fire", "channels": 11, "desc": "No fire (PrevFireMask)"},
    {"name": "unet_no_human", "group": "human", "channels": 11, "desc": "No human (population)"},
]


def run_experiment(exp):
    cmd = [
        sys.executable, "-m", "src.train_dl",
        "--model", "unet",
        "--name", exp["name"],
        "--seed", "0",
        "--epochs", "30",
        "--patience", "5",
        "--lr", "1e-3",
        "--batch-size", "32",
    ]
    if exp["group"]:
        cmd.extend(["--drop-group", exp["group"]])

    print(f"\n{'='*70}")
    print(f"Starting {exp['name']} ({exp['desc']}, {exp['channels']} channels)...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    t0 = time.time()
    res = subprocess.run(cmd, cwd=ROOT, check=True)
    elapsed = time.time() - t0
    print(f"Completed {exp['name']} in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    return elapsed


def plot_ablation(df, out_path):
    plot_df = df.iloc[::-1].copy()  # Baseline on top
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=200)

    bars = ax.barh(plot_df["run"], plot_df["pr_auc"], height=0.6, color="#2a78d6")

    base_pr = df.loc[df["run"] == "unet_base", "pr_auc"].item()
    ax.axvline(base_pr, color="#52514e", lw=1, ls="--", label=f"Baseline PR AUC ({base_pr:.4f})")

    for y_idx, (_, row) in enumerate(plot_df.iterrows()):
        val = row["pr_auc"]
        drop = row["pr_auc_drop"]
        if abs(drop) < 1e-4:
            label = f"{val:.4f} (baseline)"
        else:
            label = f"{val:.4f} (drop: {drop:+.4f})"
        ax.text(val + 0.005, y_idx, label, va="center", fontsize=7.5, color="#0b0b0b",
                bbox={"fc": "white", "ec": "none", "pad": 1}, zorder=3)

    ax.set_xlabel("Test PR AUC (U-Net, seed=0)", fontsize=8, color="#52514e")
    ax.set_title("U-Net Channel-Group Ablation Performance", fontsize=9, fontweight="bold", pad=10)
    ax.set_xlim(0, max(plot_df["pr_auc"].max() * 1.35, 0.35))
    ax.tick_params(labelsize=7.5, colors="#52514e", length=0)
    ax.grid(axis="x", color="#e5e4e0", lw=0.6)
    ax.set_axisbelow(True)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved ablation comparison figure to {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runtimes = {}

    for exp in EXPERIMENTS:
        elapsed = run_experiment(exp)
        runtimes[exp["name"]] = elapsed

    # Collect and compile results
    results_path = OUT_DIR / "results_dl.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} not found after training runs.")

    all_res = pd.read_csv(results_path)

    # Filter to our ablation runs
    names = [e["name"] for e in EXPERIMENTS]
    ablation_res = all_res[all_res["model"].isin(names)].copy()

    # Reorder according to EXPERIMENTS
    ablation_res["order"] = ablation_res["model"].map({e["name"]: i for i, e in enumerate(EXPERIMENTS)})
    ablation_res = ablation_res.sort_values("order").drop(columns=["order"])

    base_pr = ablation_res.loc[ablation_res["model"] == "unet_base", "pr_auc"].item()
    ablation_res["pr_auc_drop"] = base_pr - ablation_res["pr_auc"]
    ablation_res["n_channels"] = ablation_res["model"].map({e["name"]: e["channels"] for e in EXPERIMENTS})
    ablation_res["runtime_sec"] = ablation_res["model"].map(runtimes).round(1)
    ablation_res = ablation_res.rename(columns={"model": "run"})

    cols = ["run", "n_channels", "pr_auc", "roc_auc", "precision", "recall", "f1",
            "recall_at_1fpr", "recall_at_5fpr", "best_f1", "best_f1_threshold",
            "pr_auc_drop", "runtime_sec"]
    summary_df = ablation_res[[c for c in cols if c in ablation_res.columns]]

    csv_path = OUT_DIR / "channel_ablation.csv"
    summary_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved channel ablation table to {csv_path}")

    # Plot comparison figure
    fig_path = OUT_DIR / "channel_ablation.png"
    plot_ablation(summary_df, fig_path)

    print("\n" + "="*70)
    print("CHANNEL ABLATION RESULTS SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

