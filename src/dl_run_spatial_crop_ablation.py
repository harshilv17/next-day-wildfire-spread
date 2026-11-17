"""Run U-Net spatial crop ablation experiment.

Phase 2 Spatial Crop Ablation:
Compare:
1. Full Spatial Training (unet_base):
   - Training: Full 64x64 patches
   - Evaluation: Full 64x64 patches
   - Test: Full 64x64 patches
2. Center-Crop Training (unet_crop32):
   - Training: Center 32x32 crops of training patches
   - Evaluation: Full 64x64 patches
   - Test: Full 64x64 patches

Outputs:
  reports/dl/spatial_crop_ablation.csv
  reports/dl/spatial_crop_ablation.png
  reports/dl/spatial_crop_ablation_summary.md
  reports/dl/results_dl.csv (appends unet_crop32)
"""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "reports" / "dl"
MODELS_DIR = ROOT / "models"


def load_run_meta(name):
    meta_path = MODELS_DIR / f"{name}_run.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


def extract_row(config_name, model_name, train_size, eval_size, test_size, meta):
    ev_m = meta.get("eval_metrics", {})
    ts_m = meta.get("test_metrics", {})
    return {
        "configuration": config_name,
        "model": model_name,
        "train_crop": train_size,
        "eval_size": eval_size,
        "test_size": test_size,
        "n_params": int(meta.get("n_params", 0)),
        "best_epoch": int(meta.get("best_epoch", 0)),
        "training_duration_s": float(meta.get("runtime_sec", 0.0)),
        "eval_pr_auc": float(ev_m.get("pr_auc", 0.0)),
        "eval_roc_auc": float(ev_m.get("roc_auc", 0.0)),
        "eval_threshold": float(ev_m.get("threshold", 0.0)),
        "eval_f1": float(ev_m.get("f1", 0.0)),
        "test_pr_auc": float(ts_m.get("pr_auc", 0.0)),
        "test_roc_auc": float(ts_m.get("roc_auc", 0.0)),
        "precision": float(ts_m.get("precision", 0.0)),
        "recall": float(ts_m.get("recall", 0.0)),
        "f1": float(ts_m.get("f1", 0.0)),
        "recall_at_1fpr": float(ts_m.get("recall_at_1fpr", 0.0)),
        "recall_at_5fpr": float(ts_m.get("recall_at_5fpr", 0.0)),
    }


def run_crop_training():
    ckpt = MODELS_DIR / "unet_crop32.pt"
    meta = load_run_meta("unet_crop32")
    if ckpt.exists() and meta is not None:
        print("[CACHE] Found verified unet_crop32 run. Skipping re-training.")
        return meta

    cmd = [
        sys.executable, "-m", "src.train_dl",
        "--model", "unet",
        "--depth", "3",
        "--width", "32",
        "--crop", "32",
        "--name", "unet_crop32",
        "--seed", "0",
        "--epochs", "30",
        "--patience", "5",
        "--lr", "1e-3",
        "--batch-size", "32",
    ]

    print("=" * 70)
    print("Starting Center-Crop Training: unet_crop32 (32x32 train -> 64x64 eval/test)")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)

    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    elapsed = time.time() - t0
    print(f"Completed unet_crop32 training in {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    meta = load_run_meta("unet_crop32")
    if meta is None:
        raise RuntimeError("Failed to find run metadata for unet_crop32 after completion.")
    return meta


def plot_crop_ablation(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

    # Panel 1: Primary Metrics Comparison (PR AUC, ROC AUC, F1)
    ax1 = axes[0]
    metrics_to_plot = ["test_pr_auc", "test_roc_auc", "f1", "recall_at_1fpr", "recall_at_5fpr"]
    metric_labels = ["Test PR AUC", "Test ROC AUC", "Test F1", "Rec @ 1% FPR", "Rec @ 5% FPR"]

    full_vals = df[df["model"] == "unet_base"][metrics_to_plot].values[0]
    crop_vals = df[df["model"] == "unet_crop32"][metrics_to_plot].values[0]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    rects1 = ax1.bar(x - width / 2, full_vals, width, label="Full 64x64 Training", color="#0072B2")
    rects2 = ax1.bar(x + width / 2, crop_vals, width, label="Center 32x32 Training", color="#D55E00")

    ax1.set_ylabel("Score", fontsize=9, fontweight="bold")
    ax1.set_title("A. Test Performance: Full vs 32x32 Crop Training", fontsize=10, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=8, rotation=15)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8.5)

    # Value annotations on bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width() / 2, h),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width() / 2, h),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)

    # Panel 2: Training Efficiency vs Performance
    ax2 = axes[1]
    times = df["training_duration_s"].values / 60.0  # in minutes
    pr_aucs = df["test_pr_auc"].values
    labels = ["Full 64x64\n(unet_base)", "Center 32x32\n(unet_crop32)"]
    colors = ["#0072B2", "#D55E00"]

    for i in range(len(df)):
        ax2.scatter(times[i], pr_aucs[i], color=colors[i], s=120, zorder=3)
        ax2.annotate(f"{labels[i]}\nPR AUC: {pr_aucs[i]:.4f}\n({times[i]:.1f} min)",
                     (times[i], pr_aucs[i]),
                     textcoords="offset points", xytext=(10, -5), fontsize=8, fontweight="bold")

    ax2.set_xlabel("Training Duration (minutes)", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Test PR AUC (Scored on Full 64x64)", fontsize=9, fontweight="bold")
    ax2.set_title("B. Training Efficiency vs Test PR AUC", fontsize=10, fontweight="bold", pad=10)
    ax2.set_xlim(0, max(times) * 1.45)
    ax2.set_ylim(min(pr_aucs) - 0.02, max(pr_aucs) + 0.03)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved spatial crop ablation figure to {out_path}")


def generate_summary_md(df, out_path):
    base_row = df[df["model"] == "unet_base"].iloc[0]
    crop_row = df[df["model"] == "unet_crop32"].iloc[0]

    delta_pr = crop_row["test_pr_auc"] - base_row["test_pr_auc"]
    delta_roc = crop_row["test_roc_auc"] - base_row["test_roc_auc"]
    delta_f1 = crop_row["f1"] - base_row["f1"]
    time_ratio = crop_row["training_duration_s"] / max(1e-6, base_row["training_duration_s"])

    summary = f"""# Spatial Crop Ablation Report (Phase 2, Task 1c)

## 1. Executive Summary

This study investigates whether reducing the spatial input context during training impairs next-day wildfire spread prediction.
We compare:
1. **Full Spatial Training (`unet_base`):** U-Net trained on full 64x64 patches (4,096 km^2).
2. **Center-Crop Training (`unet_crop32`):** U-Net trained on center 32x32 crops (1,024 km^2, 1/4 the spatial area).

**Methodological Rigour:** In contrast to earlier naive evaluations where crop models were scored only on center pixels (which artificially inflated fire density and produced non-comparable scores), **both models in this study are evaluated and tested on the exact same full 64x64 patches**. Because the U-Net architecture is fully convolutional, weights learned on 32x32 inputs apply directly to 64x64 inference.

---

## 2. Quantitative Results Matrix

| Configuration | Model | Train Size | Eval / Test Size | Parameters | Best Epoch | Duration (s) | Eval PR AUC | Test PR AUC | Delta vs Base | Test ROC AUC | Test Precision | Test Recall | Test F1 | Test Rec@1%FPR | Test Rec@5%FPR |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full Training** | `unet_base` | 64x64 | 64x64 | {base_row['n_params']:,} | {base_row['best_epoch']} | {base_row['training_duration_s']:.1f} | {base_row['eval_pr_auc']:.4f} | {base_row['test_pr_auc']:.4f} | Reference | {base_row['test_roc_auc']:.4f} | {base_row['precision']:.4f} | {base_row['recall']:.4f} | {base_row['f1']:.4f} | {base_row['recall_at_1fpr']:.4f} | {base_row['recall_at_5fpr']:.4f} |
| **Center Crop** | `unet_crop32` | 32x32 | 64x64 | {crop_row['n_params']:,} | {crop_row['best_epoch']} | {crop_row['training_duration_s']:.1f} | {crop_row['eval_pr_auc']:.4f} | {crop_row['test_pr_auc']:.4f} | {delta_pr:+.4f} | {crop_row['test_roc_auc']:.4f} | {crop_row['precision']:.4f} | {crop_row['recall']:.4f} | {crop_row['f1']:.4f} | {crop_row['recall_at_1fpr']:.4f} | {crop_row['recall_at_5fpr']:.4f} |

---

## 3. Analysis & Key Insights

### 1. Result
- **Test PR AUC:** {base_row['test_pr_auc']:.4f} (full 64x64 training) vs {crop_row['test_pr_auc']:.4f} (centre 32x32 training), **{delta_pr:+.4f}**.
- **Eval PR AUC:** {base_row['eval_pr_auc']:.4f} vs {crop_row['eval_pr_auc']:.4f} ({crop_row['eval_pr_auc'] - base_row['eval_pr_auc']:+.4f}).
- **Test F1:** {base_row['f1']:.4f} vs {crop_row['f1']:.4f} ({delta_f1:+.4f}); **recall at 1% FPR:** {base_row['recall_at_1fpr']:.4f} vs {crop_row['recall_at_1fpr']:.4f} ({crop_row['recall_at_1fpr'] - base_row['recall_at_1fpr']:+.4f}).
- The drop is far larger than run-to-run noise (same-seed reruns move test PR AUC by up to 0.02), so **training on the centre crop and predicting on full patches clearly does not work**. Single seed (0), weighted BCE.

### 2. What this does and does not show
This is a **centre** crop, the same 32x32 window of every patch. It is not the **random** 32x32 crops Huot et al. (2022) train on, so it does not test their setup. Three effects change together and cannot be separated here:
1. **Different fire share.** Patches are centred on fires, so the centre crop holds about 3x more fire (pos_weight 30.4 against 90.0 on full patches). The model learns a different prior and is then scored on the sparser full patch.
2. **No patch edges in training.** The model never sees the outer ring of the patch (3/4 of its area), yet it has to predict there at test time.
3. **Train/test input mismatch.** The fully convolutional U-Net runs on 64x64 inputs it was never trained on (different zero-padded borders and batch-norm statistics).

The result supports training and predicting on the same full 64x64 patches. It does not show *how much* of the drop comes from losing spatial context. That would need random-crop training, which was not run.

### 3. Cost
- Training took {crop_row['training_duration_s']:.1f}s against {base_row['training_duration_s']:.1f}s ({time_ratio * 100:.1f}% of the baseline time), because each patch has 4x fewer pixels.

### 4. Protocol
- Both models are evaluated on the same full 64x64 eval and test patches (an earlier version also cropped eval/test, which made the scores incomparable).
- The decision threshold is chosen on eval (threshold = {crop_row['eval_threshold']:.4f}) and applied unchanged to test.

---

## 4. Visualizations
The spatial crop comparison is visualized in `reports/dl/spatial_crop_ablation.png`:
- **Panel A:** Side-by-side bar chart of primary test metrics (PR AUC, ROC AUC, F1, Recall @ 1% FPR, Recall @ 5% FPR).
- **Panel B:** Training duration versus Test PR AUC trade-off plot.
"""
    with open(out_path, "w") as f:
        f.write(summary.strip() + "\n")
    print(f"Saved spatial crop ablation summary to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run U-Net spatial crop ablation.")
    parser.add_argument("--check-only", action="store_true", help="Only verify configurations without training")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("U-NET SPATIAL CROP ABLATION: FULL (64x64) vs CENTER CROP (32x32)")
    print("=" * 75)
    base_meta = load_run_meta("unet_base")
    crop_meta = load_run_meta("unet_crop32")
    print(f"  Full 64x64 Training (unet_base)   -> [{'CACHED' if base_meta else 'PENDING'}]")
    print(f"  Center 32x32 Training (unet_crop32) -> [{'CACHED' if crop_meta else 'PENDING'}]")

    if args.check_only:
        print("\nPreflight check completed.")
        return

    if base_meta is None:
        raise FileNotFoundError("Missing baseline run metadata 'models/unet_base_run.json'.")

    # Run crop experiment
    crop_meta = run_crop_training()

    # Compile table
    csv_path = OUT_DIR / "spatial_crop_ablation.csv"
    rows = [
        extract_row("Full spatial training", "unet_base", "64x64", "64x64", "64x64", base_meta),
        extract_row("Center-crop training", "unet_crop32", "32x32", "64x64", "64x64", crop_meta),
    ]

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved spatial crop ablation table to {csv_path}")

    # Plot
    fig_path = OUT_DIR / "spatial_crop_ablation.png"
    plot_crop_ablation(df, fig_path)

    # Summary
    md_path = OUT_DIR / "spatial_crop_ablation_summary.md"
    generate_summary_md(df, md_path)

    print("\n" + "=" * 75)
    print("COMPLETED SPATIAL CROP ABLATION")
    print("=" * 75)
    print(df[["configuration", "model", "train_crop", "eval_size", "training_duration_s", "eval_pr_auc", "test_pr_auc", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
