"""Run Hybrid U-Net (13-Channel) and Probability Ensemble experiments.

Phase 2, Task 2 & Task 3:
1. Hybrid U-Net Models (13 channels = 12 raw + GBM probability map):
   - unet_hybrid_base (Depth 3, Width 32, 13 channels)
   - unet_hybrid_d4 (Depth 4, Width 32, 13 channels)
   * Training uses leakage-free fire-event GroupKFold OOF predictions.
   * Eval and Test use final full GBM model predictions.
   * Probability channel normalized with train-only statistics.

2. Probability Ensembles:
   - Simple Average (alpha = 0.5):
     * unet_base + xgboost
     * unet_d4_w32 + xgboost
     * unet_hybrid_base + xgboost
     * unet_hybrid_d4 + xgboost
   - Weighted Average (alpha * U-Net + (1 - alpha) * XGBoost):
     * alpha tuned exclusively on evaluation split (search grid [0.0, 1.0], step 0.01)
     * Decision threshold selected on evaluation split at optimal alpha
     * Scored unchanged on held-out test split.

Outputs:
  reports/dl/hybrid_ensemble_results.csv
  reports/dl/hybrid_ensemble.png
  reports/dl/hybrid_ensemble_summary.md
  reports/dl/results_dl.csv (appends hybrid and top ensemble models)
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
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_split
from src.dl_data import make_loaders
from src.dl_models import build_model
from src.evaluate import metrics
from src.train_dl import device, predict

OUT_DIR = ROOT / "reports" / "dl"
MODELS_DIR = ROOT / "models"
PREDS_DIR = OUT_DIR / "preds"
NOISE = 0.02  # largest same-seed rerun shift in test PR AUC (reports/dl/checks/reruns.csv)


def load_run_meta(name):
    meta_path = MODELS_DIR / f"{name}_run.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


def run_hybrid_training(name, depth, width=32):
    ckpt = MODELS_DIR / f"{name}.pt"
    meta = load_run_meta(name)
    if ckpt.exists() and meta is not None:
        print(f"[CACHE] Found verified run for {name}. Skipping re-training.")
        return meta

    cmd = [
        sys.executable, "-m", "src.train_dl",
        "--model", "unet",
        "--depth", str(depth),
        "--width", str(width),
        "--hybrid",
        "--name", name,
        "--seed", "0",
        "--epochs", "30",
        "--patience", "5",
        "--lr", "1e-3",
        "--batch-size", "32",
    ]

    print("=" * 70)
    print(f"Starting Training: {name} (Depth {depth}, Width {width}, 13 Channels)")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)

    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    elapsed = time.time() - t0
    print(f"Completed {name} training in {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    meta = load_run_meta(name)
    if meta is None:
        raise RuntimeError(f"Failed to find run metadata for {name} after completion.")
    return meta


def get_model_predictions(name, depth, width=32, hybrid=False):
    """Load or generate (eval_prob, test_prob) for a U-Net model."""
    dev = device()
    _, eval_dl, test_dl = make_loaders(batch_size=32, hybrid=hybrid)

    ckpt = MODELS_DIR / f"{name}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt}")

    model = build_model("unet", depth=depth, width=width, in_ch=13 if hybrid else 12).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    p_eval = predict(model, eval_dl, dev)
    p_test = predict(model, test_dl, dev)
    return p_eval, p_test


def load_xgboost_predictions():
    """Load final XGBoost eval and test probability maps."""
    eval_npz = ROOT / "data" / "processed" / "gbm_oof_eval.npz"
    test_npz = ROOT / "data" / "processed" / "gbm_oof_test.npz"
    if not eval_npz.exists():
        eval_npz = OUT_DIR / "gbm_oof_eval.npz"
    if not test_npz.exists():
        test_npz = OUT_DIR / "gbm_oof_test.npz"

    with np.load(eval_npz) as d:
        p_eval = d["probabilities"].astype(np.float32)
    with np.load(test_npz) as d:
        p_test = d["probabilities"].astype(np.float32)
    return p_eval, p_test


def optimize_alpha(y_eval, p1_eval, p2_eval, grid=np.linspace(0.0, 1.0, 101)):
    """Find alpha in [0, 1] maximizing evaluation PR AUC."""
    y_flat = y_eval.ravel()
    keep = y_flat != -1
    y_valid = y_flat[keep].astype(int)
    p1_flat = p1_eval.ravel()[keep]
    p2_flat = p2_eval.ravel()[keep]

    best_alpha, best_score = 0.5, -1.0
    for a in grid:
        p_ens = a * p1_flat + (1.0 - a) * p2_flat
        score = average_precision_score(y_valid, p_ens)
        if score > best_score:
            best_score, best_alpha = score, float(a)

    return best_alpha, best_score


def evaluate_ensemble(name, category, y_eval, y_test, p1_eval, p1_test, p2_eval, p2_test, alpha,
                      m1_name, m2_name, n_params=0):
    """Compute ensemble metrics on eval and test splits at a given alpha."""
    p_eval_ens = alpha * p1_eval + (1.0 - alpha) * p2_eval
    p_test_ens = alpha * p1_test + (1.0 - alpha) * p2_test

    eval_m = metrics(y_eval, p_eval_ens)
    eval_threshold = eval_m["best_f1_threshold"]

    test_m = metrics(y_test, p_test_ens, threshold=eval_threshold)

    return {
        "model": name,
        "category": category,
        "m1": m1_name,
        "m2": m2_name,
        "alpha": round(alpha, 3),
        "channels": "Ensemble",
        "n_params": n_params,
        "eval_pr_auc": float(eval_m["pr_auc"]),
        "eval_roc_auc": float(eval_m["roc_auc"]),
        "eval_threshold": float(eval_threshold),
        "eval_f1": float(eval_m["best_f1"]),
        "test_pr_auc": float(test_m["pr_auc"]),
        "test_roc_auc": float(test_m["roc_auc"]),
        "precision": float(test_m["precision"]),
        "recall": float(test_m["recall"]),
        "f1": float(test_m["f1"]),
        "recall_at_1fpr": float(test_m["recall_at_1fpr"]),
        "recall_at_5fpr": float(test_m["recall_at_5fpr"]),
        "p_test": p_test_ens,
    }


def plot_hybrid_ensemble(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=200)

    # Panel 1: Test PR AUC across Parent, Hybrid, and Ensemble Models
    ax1 = axes[0]
    sub = df.sort_values("test_pr_auc", ascending=True).reset_index(drop=True)

    colors = {
        "Parent Model": "#4b5563",
        "Hybrid Model": "#0072B2",
        "Simple Ensemble": "#E69F00",
        "Weighted Ensemble": "#009E73",
    }
    bar_colors = [colors.get(cat, "#3b82f6") for cat in sub["category"]]

    y_pos = np.arange(len(sub))
    bars = ax1.barh(y_pos, sub["test_pr_auc"], color=bar_colors, height=0.65, edgecolor="black", linewidth=0.5)

    ax1.set_yticks(y_pos)
    labels = [f"{r['model']} ({r['category']})" for _, r in sub.iterrows()]
    ax1.set_yticklabels(labels, fontsize=7.5)
    ax1.set_xlabel("Test PR AUC", fontsize=9, fontweight="bold")
    ax1.set_title("A. Test PR AUC Comparison", fontsize=10, fontweight="bold", pad=10)
    ax1.set_xlim(0.20, max(sub["test_pr_auc"]) * 1.08)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        ax1.annotate(f"{w:.4f}", (w, bar.get_y() + bar.get_height() / 2),
                     textcoords="offset points", xytext=(4, 0), va="center", fontsize=7.5, fontweight="bold")

    # Legend for Panel 1
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[cat]) for cat in colors]
    ax1.legend(handles, colors.keys(), loc="lower right", fontsize=7.5)

    # Panel 2: PR AUC vs F1 Score Trade-off
    ax2 = axes[1]
    for cat in colors:
        c_sub = sub[sub["category"] == cat]
        if len(c_sub) > 0:
            ax2.scatter(c_sub["test_pr_auc"], c_sub["f1"], color=colors[cat], label=cat, s=90, zorder=3)
            for _, r in c_sub.iterrows():
                ax2.annotate(r["model"], (r["test_pr_auc"], r["f1"]),
                             textcoords="offset points", xytext=(5, 3), fontsize=7)

    ax2.set_xlabel("Test PR AUC", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Test F1 Score (at Eval Threshold)", fontsize=9, fontweight="bold")
    ax2.set_title("B. Test PR AUC vs Test F1 Trade-off", fontsize=10, fontweight="bold", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved hybrid ensemble figure to {out_path}")


def generate_summary_md(df, out_path):
    rows = {r["model"]: r for _, r in df.iterrows()}

    xgb = rows.get("xgboost", {})
    u_base = rows.get("unet_base", {})
    u_d4 = rows.get("unet_d4_w32", {})
    h_base = rows.get("unet_hybrid_base", {})
    h_d4 = rows.get("unet_hybrid_d4", {})

    ens_base_avg = rows.get("ens_base_xgb_avg", {})
    ens_base_w = rows.get("ens_base_xgb_weighted", {})
    ens_d4_avg = rows.get("ens_d4_xgb_avg", {})
    ens_d4_w = rows.get("ens_d4_xgb_weighted", {})
    ens_h_base_w = rows.get("ens_hybrid_base_xgb_weighted", {})
    ens_h_d4_w = rows.get("ens_hybrid_d4_xgb_weighted", {})

    delta_h_base = h_base.get("test_pr_auc", 0) - u_base.get("test_pr_auc", 0)
    delta_h_d4 = h_d4.get("test_pr_auc", 0) - u_d4.get("test_pr_auc", 0)

    summary = f"""# Hybrid U-Net and Probability Ensemble Report (Phase 2, Tasks 2 & 3)

## 1. Executive Summary

This study completes Phase 2 Tasks 2 and 3:
1. **Hybrid U-Net (13-Channel Model):** Ingests the event-grouped XGBoost probability map as an explicit 13th input channel alongside the 12 original environmental variables.
   - Training uses strictly leakage-free Out-of-Fold (OOF) GBM predictions generated via fire-event `GroupKFold(5)`.
   - Evaluation and testing use final GBM predictions generated by the model fitted on all 14,979 training patches.
   - The 13th channel is normalized via training-only z-scoring ($\\\\mu=0.1588, \\\\sigma=0.2137$).
2. **Probability Ensembles:**
   - **Simple Average:** $0.5 \\\\cdot P_{{\\\\text{{unet}}}} + 0.5 \\\\cdot P_{{\\\\text{{xgb}}}}$
   - **Weighted Average:** $\\\\alpha \\\\cdot P_{{\\\\text{{unet}}}} + (1 - \\\\alpha) \\\\cdot P_{{\\\\text{{xgb}}}}$, where $\\\\alpha \\\\in [0, 1]$ is tuned **exclusively on the evaluation split** to maximize Eval PR AUC, and applied unchanged to the test split.

---

## 2. Complete Model & Ensemble Comparison Table

| Category | Model Name | Architecture | Channels | Parameters | Best Alpha | Eval PR AUC | Test PR AUC | Test ROC AUC | Test Precision | Test Recall | Test F1 | Test Rec@1%FPR | Test Rec@5%FPR |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Parent Model** | `xgboost` | Gradient Boosted Trees | 35 tabular | — | — | {xgb.get('eval_pr_auc', 0):.4f} | **{xgb.get('test_pr_auc', 0):.4f}** | {xgb.get('test_roc_auc', 0):.4f} | {xgb.get('precision', 0):.4f} | {xgb.get('recall', 0):.4f} | {xgb.get('f1', 0):.4f} | {xgb.get('recall_at_1fpr', 0):.4f} | {xgb.get('recall_at_5fpr', 0):.4f} |
| **Parent Model** | `unet_base` | U-Net (Depth 3, Width 32) | 12 | {u_base.get('n_params', 0):,} | — | {u_base.get('eval_pr_auc', 0):.4f} | **{u_base.get('test_pr_auc', 0):.4f}** | {u_base.get('test_roc_auc', 0):.4f} | {u_base.get('precision', 0):.4f} | {u_base.get('recall', 0):.4f} | {u_base.get('f1', 0):.4f} | {u_base.get('recall_at_1fpr', 0):.4f} | {u_base.get('recall_at_5fpr', 0):.4f} |
| **Parent Model** | `unet_d4_w32` | U-Net (Depth 4, Width 32) | 12 | {u_d4.get('n_params', 0):,} | — | {u_d4.get('eval_pr_auc', 0):.4f} | **{u_d4.get('test_pr_auc', 0):.4f}** | {u_d4.get('test_roc_auc', 0):.4f} | {u_d4.get('precision', 0):.4f} | {u_d4.get('recall', 0):.4f} | {u_d4.get('f1', 0):.4f} | {u_d4.get('recall_at_1fpr', 0):.4f} | {u_d4.get('recall_at_5fpr', 0):.4f} |
| **Hybrid Model** | `unet_hybrid_base` | U-Net (Depth 3, Width 32) | 13 (12 + GBM) | {h_base.get('n_params', 0):,} | — | {h_base.get('eval_pr_auc', 0):.4f} | **{h_base.get('test_pr_auc', 0):.4f}** | {h_base.get('test_roc_auc', 0):.4f} | {h_base.get('precision', 0):.4f} | {h_base.get('recall', 0):.4f} | {h_base.get('f1', 0):.4f} | {h_base.get('recall_at_1fpr', 0):.4f} | {h_base.get('recall_at_5fpr', 0):.4f} |
| **Hybrid Model** | `unet_hybrid_d4` | U-Net (Depth 4, Width 32) | 13 (12 + GBM) | {h_d4.get('n_params', 0):,} | — | {h_d4.get('eval_pr_auc', 0):.4f} | **{h_d4.get('test_pr_auc', 0):.4f}** | {h_d4.get('test_roc_auc', 0):.4f} | {h_d4.get('precision', 0):.4f} | {h_d4.get('recall', 0):.4f} | {h_d4.get('f1', 0):.4f} | {h_d4.get('recall_at_1fpr', 0):.4f} | {h_d4.get('recall_at_5fpr', 0):.4f} |
| **Simple Ensemble** | `ens_base_xgb_avg` | `unet_base` + `xgboost` | Ensemble | — | 0.500 | {ens_base_avg.get('eval_pr_auc', 0):.4f} | **{ens_base_avg.get('test_pr_auc', 0):.4f}** | {ens_base_avg.get('test_roc_auc', 0):.4f} | {ens_base_avg.get('precision', 0):.4f} | {ens_base_avg.get('recall', 0):.4f} | {ens_base_avg.get('f1', 0):.4f} | {ens_base_avg.get('recall_at_1fpr', 0):.4f} | {ens_base_avg.get('recall_at_5fpr', 0):.4f} |
| **Weighted Ensemble** | `ens_base_xgb_weighted` | `unet_base` + `xgboost` | Ensemble | — | {ens_base_w.get('alpha', 0.5):.3f} | {ens_base_w.get('eval_pr_auc', 0):.4f} | **{ens_base_w.get('test_pr_auc', 0):.4f}** | {ens_base_w.get('test_roc_auc', 0):.4f} | {ens_base_w.get('precision', 0):.4f} | {ens_base_w.get('recall', 0):.4f} | {ens_base_w.get('f1', 0):.4f} | {ens_base_w.get('recall_at_1fpr', 0):.4f} | {ens_base_w.get('recall_at_5fpr', 0):.4f} |
| **Simple Ensemble** | `ens_d4_xgb_avg` | `unet_d4_w32` + `xgboost` | Ensemble | — | 0.500 | {ens_d4_avg.get('eval_pr_auc', 0):.4f} | **{ens_d4_avg.get('test_pr_auc', 0):.4f}** | {ens_d4_avg.get('test_roc_auc', 0):.4f} | {ens_d4_avg.get('precision', 0):.4f} | {ens_d4_avg.get('recall', 0):.4f} | {ens_d4_avg.get('f1', 0):.4f} | {ens_d4_avg.get('recall_at_1fpr', 0):.4f} | {ens_d4_avg.get('recall_at_5fpr', 0):.4f} |
| **Weighted Ensemble** | `ens_d4_xgb_weighted` | `unet_d4_w32` + `xgboost` | Ensemble | — | {ens_d4_w.get('alpha', 0.5):.3f} | {ens_d4_w.get('eval_pr_auc', 0):.4f} | **{ens_d4_w.get('test_pr_auc', 0):.4f}** | {ens_d4_w.get('test_roc_auc', 0):.4f} | {ens_d4_w.get('precision', 0):.4f} | {ens_d4_w.get('recall', 0):.4f} | {ens_d4_w.get('f1', 0):.4f} | {ens_d4_w.get('recall_at_1fpr', 0):.4f} | {ens_d4_w.get('recall_at_5fpr', 0):.4f} |
| **Weighted Ensemble** | `ens_hybrid_base_xgb_weighted` | `unet_hybrid_base` + `xgboost` | Ensemble | — | {ens_h_base_w.get('alpha', 0.5):.3f} | {ens_h_base_w.get('eval_pr_auc', 0):.4f} | **{ens_h_base_w.get('test_pr_auc', 0):.4f}** | {ens_h_base_w.get('test_roc_auc', 0):.4f} | {ens_h_base_w.get('precision', 0):.4f} | {ens_h_base_w.get('recall', 0):.4f} | {ens_h_base_w.get('f1', 0):.4f} | {ens_h_base_w.get('recall_at_1fpr', 0):.4f} | {ens_h_base_w.get('recall_at_5fpr', 0):.4f} |
| **Weighted Ensemble** | `ens_hybrid_d4_xgb_weighted` | `unet_hybrid_d4` + `xgboost` | Ensemble | — | {ens_h_d4_w.get('alpha', 0.5):.3f} | {ens_h_d4_w.get('eval_pr_auc', 0):.4f} | **{ens_h_d4_w.get('test_pr_auc', 0):.4f}** | {ens_h_d4_w.get('test_roc_auc', 0):.4f} | {ens_h_d4_w.get('precision', 0):.4f} | {ens_h_d4_w.get('recall', 0):.4f} | {ens_h_d4_w.get('f1', 0):.4f} | {ens_h_d4_w.get('recall_at_1fpr', 0):.4f} | {ens_h_d4_w.get('recall_at_5fpr', 0):.4f} |

---

## 3. Detailed Scientific Analysis

Noise reference: a same-seed rerun moves test PR AUC by up to {NOISE} (`reports/dl/checks/reruns.csv`), so single-seed differences smaller than that are not evidence either way.

### 1. Does the GBM probability channel help the U-Net?
- **Depth 3:** `unet_base` ({u_base.get('test_pr_auc', 0):.4f}) → `unet_hybrid_base` ({h_base.get('test_pr_auc', 0):.4f}), **{delta_h_base:+.4f}**.
- **Depth 4:** `unet_d4_w32` ({u_d4.get('test_pr_auc', 0):.4f}) → `unet_hybrid_d4` ({h_d4.get('test_pr_auc', 0):.4f}), **{delta_h_d4:+.4f}**.
- {"**No measurable effect.** Both changes are well inside run-to-run noise: the 13th channel adds nothing the U-Net does not already learn from the raw patch." if max(abs(delta_h_base), abs(delta_h_d4)) < NOISE else "At least one change exceeds run-to-run noise; confirm it with 3 seeds before drawing a conclusion."}
- One possible reason: on train, 55% of patches belong to one linked fire event, so their GBM channel comes from a fold model fitted on only 6,741 patches, weaker than the GBM that produces the eval/test channel (`gbm_oof_metadata.json`). The U-Net may learn to trust the channel less than it deserves.

### 2. Does averaging with XGBoost help?
- **`unet_base` + `xgboost`:** simple average {ens_base_avg.get('test_pr_auc', 0):.4f}, eval-tuned weighted average {ens_base_w.get('test_pr_auc', 0):.4f} (alpha = {ens_base_w.get('alpha', 0.5):.2f} on the U-Net), against {u_base.get('test_pr_auc', 0):.4f} for the U-Net alone: **{ens_base_w.get('test_pr_auc', 0) - u_base.get('test_pr_auc', 0):+.4f}**, within noise.
- **`unet_d4_w32` + `xgboost`:** eval-tuned alpha = **{ens_d4_w.get('alpha', 0.5):.2f}**. {"With alpha = 1 XGBoost gets zero weight, so the weighted ensemble *is* the U-Net (identical scores) and is not reported as a separate model." if ens_d4_w.get('alpha', 0.5) >= 0.995 else ""} The forced 50/50 average scores {ens_d4_avg.get('test_pr_auc', 0):.4f}, {ens_d4_avg.get('test_pr_auc', 0) - u_d4.get('test_pr_auc', 0):+.4f} against the U-Net alone.
- **`unet_hybrid_d4` + `xgboost`:** eval-tuned alpha = **{ens_h_d4_w.get('alpha', 0.5):.2f}**; same reading.

### 3. Reading
- XGBoost adds at most a within-noise gain to the weaker depth-3 U-Net, and nothing to the depth-4 U-Net: tuning on eval gives it zero weight, and forcing it in lowers PR AUC. Once the U-Net sees enough spatial context, the 35 engineered features are **redundant with what it learns, not complementary**.
- Neither the hybrid channel nor the ensemble is used for the final model.

### 4. Limitations
- Single seed (0) and weighted BCE for every model here; the final model uses focal loss.
- The depth-3 vs depth-4 gap is a single-seed result; see `architecture_ablation_summary.md` and the 3-seed final runs.

---

## 4. Visualizations
The results are visualized in `reports/dl/hybrid_ensemble.png`:
- **Panel A:** Horizontal bar chart comparing Test PR AUC across parent models, hybrid architectures, and ensembles.
- **Panel B:** Scatter plot of Test PR AUC versus Test F1 showing the Pareto performance frontier.
"""
    with open(out_path, "w") as f:
        f.write(summary.strip() + "\n")
    print(f"Saved hybrid ensemble summary to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Hybrid U-Net and Probability Ensemble pipeline.")
    parser.add_argument("--check-only", action="store_true", help="Run sanity checks without training")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("HYBRID U-NET & ENSEMBLE PIPELINE (PHASE 2, TASKS 2 & 3)")
    print("=" * 75)

    # 1. Verify OOF and Parent predictions
    p_xgb_eval, p_xgb_test = load_xgboost_predictions()
    print(f"XGBoost predictions: eval={p_xgb_eval.shape}, test={p_xgb_test.shape}")

    # Load ground truth labels
    _, y_eval = load_split("eval")
    _, y_test = load_split("test")

    xgb_eval_m = metrics(y_eval, p_xgb_eval)
    xgb_test_m = metrics(y_test, p_xgb_test, threshold=xgb_eval_m["best_f1_threshold"])
    print(f"Verified parent XGBoost: Eval PR={xgb_eval_m['pr_auc']:.4f}, Test PR={xgb_test_m['pr_auc']:.4f}")

    if args.check_only:
        print("\nPreflight check completed successfully.")
        return

    # 2. Train Hybrid U-Net models
    print("\n>>> Step 1: Training Hybrid U-Net Models (13 channels)...")
    meta_h_base = run_hybrid_training("unet_hybrid_base", depth=3, width=32)
    meta_h_d4 = run_hybrid_training("unet_hybrid_d4", depth=4, width=32)

    # 3. Load or compute probability maps for all models
    print("\n>>> Step 2: Generating probability maps for ensemble evaluation...")
    p_ubase_eval, p_ubase_test = get_model_predictions("unet_base", depth=3, width=32, hybrid=False)
    p_ud4_eval, p_ud4_test = get_model_predictions("unet_d4_w32", depth=4, width=32, hybrid=False)
    p_hbase_eval, p_hbase_test = get_model_predictions("unet_hybrid_base", depth=3, width=32, hybrid=True)
    p_hd4_eval, p_hd4_test = get_model_predictions("unet_hybrid_d4", depth=4, width=32, hybrid=True)

    # 4. Compile parent and hybrid rows
    rows = []

    # Parent XGBoost
    rows.append({
        "model": "xgboost",
        "category": "Parent Model",
        "m1": "xgboost", "m2": "none",
        "alpha": 0.0, "channels": "35 tabular", "n_params": 0,
        "eval_pr_auc": float(xgb_eval_m["pr_auc"]), "eval_roc_auc": float(xgb_eval_m["roc_auc"]),
        "eval_threshold": float(xgb_eval_m["best_f1_threshold"]), "eval_f1": float(xgb_eval_m["best_f1"]),
        "test_pr_auc": float(xgb_test_m["pr_auc"]), "test_roc_auc": float(xgb_test_m["roc_auc"]),
        "precision": float(xgb_test_m["precision"]), "recall": float(xgb_test_m["recall"]),
        "f1": float(xgb_test_m["f1"]), "recall_at_1fpr": float(xgb_test_m["recall_at_1fpr"]),
        "recall_at_5fpr": float(xgb_test_m["recall_at_5fpr"]),
        "p_test": p_xgb_test,
    })

    # Parent U-Net Base
    u_base_eval_m = metrics(y_eval, p_ubase_eval)
    u_base_thresh = u_base_eval_m["best_f1_threshold"]
    u_base_test_m = metrics(y_test, p_ubase_test, threshold=u_base_thresh)
    rows.append({
        "model": "unet_base",
        "category": "Parent Model",
        "m1": "unet_base", "m2": "none",
        "alpha": 1.0, "channels": "12", "n_params": 469761,
        "eval_pr_auc": float(u_base_eval_m["pr_auc"]), "eval_roc_auc": float(u_base_eval_m["roc_auc"]),
        "eval_threshold": float(u_base_thresh), "eval_f1": float(u_base_eval_m["best_f1"]),
        "test_pr_auc": float(u_base_test_m["pr_auc"]), "test_roc_auc": float(u_base_test_m["roc_auc"]),
        "precision": float(u_base_test_m["precision"]), "recall": float(u_base_test_m["recall"]),
        "f1": float(u_base_test_m["f1"]), "recall_at_1fpr": float(u_base_test_m["recall_at_1fpr"]),
        "recall_at_5fpr": float(u_base_test_m["recall_at_5fpr"]),
        "p_test": p_ubase_test,
    })

    # Parent U-Net D4
    u_d4_eval_m = metrics(y_eval, p_ud4_eval)
    u_d4_thresh = u_d4_eval_m["best_f1_threshold"]
    u_d4_test_m = metrics(y_test, p_ud4_test, threshold=u_d4_thresh)
    rows.append({
        "model": "unet_d4_w32",
        "category": "Parent Model",
        "m1": "unet_d4_w32", "m2": "none",
        "alpha": 1.0, "channels": "12", "n_params": 1929601,
        "eval_pr_auc": float(u_d4_eval_m["pr_auc"]), "eval_roc_auc": float(u_d4_eval_m["roc_auc"]),
        "eval_threshold": float(u_d4_thresh), "eval_f1": float(u_d4_eval_m["best_f1"]),
        "test_pr_auc": float(u_d4_test_m["pr_auc"]), "test_roc_auc": float(u_d4_test_m["roc_auc"]),
        "precision": float(u_d4_test_m["precision"]), "recall": float(u_d4_test_m["recall"]),
        "f1": float(u_d4_test_m["f1"]), "recall_at_1fpr": float(u_d4_test_m["recall_at_1fpr"]),
        "recall_at_5fpr": float(u_d4_test_m["recall_at_5fpr"]),
        "p_test": p_ud4_test,
    })

    # Hybrid U-Net Base
    h_base_eval_m = metrics(y_eval, p_hbase_eval)
    h_base_thresh = h_base_eval_m["best_f1_threshold"]
    h_base_test_m = metrics(y_test, p_hbase_test, threshold=h_base_thresh)
    rows.append({
        "model": "unet_hybrid_base",
        "category": "Hybrid Model",
        "m1": "unet_hybrid_base", "m2": "none",
        "alpha": 1.0, "channels": "13", "n_params": 470049,
        "eval_pr_auc": float(h_base_eval_m["pr_auc"]), "eval_roc_auc": float(h_base_eval_m["roc_auc"]),
        "eval_threshold": float(h_base_thresh), "eval_f1": float(h_base_eval_m["best_f1"]),
        "test_pr_auc": float(h_base_test_m["pr_auc"]), "test_roc_auc": float(h_base_test_m["roc_auc"]),
        "precision": float(h_base_test_m["precision"]), "recall": float(h_base_test_m["recall"]),
        "f1": float(h_base_test_m["f1"]), "recall_at_1fpr": float(h_base_test_m["recall_at_1fpr"]),
        "recall_at_5fpr": float(h_base_test_m["recall_at_5fpr"]),
        "p_test": p_hbase_test,
    })

    # Hybrid U-Net D4
    h_d4_eval_m = metrics(y_eval, p_hd4_eval)
    h_d4_thresh = h_d4_eval_m["best_f1_threshold"]
    h_d4_test_m = metrics(y_test, p_hd4_test, threshold=h_d4_thresh)
    rows.append({
        "model": "unet_hybrid_d4",
        "category": "Hybrid Model",
        "m1": "unet_hybrid_d4", "m2": "none",
        "alpha": 1.0, "channels": "13", "n_params": 1929889,
        "eval_pr_auc": float(h_d4_eval_m["pr_auc"]), "eval_roc_auc": float(h_d4_eval_m["roc_auc"]),
        "eval_threshold": float(h_d4_thresh), "eval_f1": float(h_d4_eval_m["best_f1"]),
        "test_pr_auc": float(h_d4_test_m["pr_auc"]), "test_roc_auc": float(h_d4_test_m["roc_auc"]),
        "precision": float(h_d4_test_m["precision"]), "recall": float(h_d4_test_m["recall"]),
        "f1": float(h_d4_test_m["f1"]), "recall_at_1fpr": float(h_d4_test_m["recall_at_1fpr"]),
        "recall_at_5fpr": float(h_d4_test_m["recall_at_5fpr"]),
        "p_test": p_hd4_test,
    })

    # 5. Evaluate Probability Ensembles
    print("\n>>> Step 3: Evaluating Probability Ensembles (Simple & Weighted)...")
    ensemble_pairs = [
        ("ens_base_xgb", "unet_base", "xgboost", p_ubase_eval, p_ubase_test, p_xgb_eval, p_xgb_test, 469761),
        ("ens_d4_xgb", "unet_d4_w32", "xgboost", p_ud4_eval, p_ud4_test, p_xgb_eval, p_xgb_test, 1929601),
        ("ens_hybrid_base_xgb", "unet_hybrid_base", "xgboost", p_hbase_eval, p_hbase_test, p_xgb_eval, p_xgb_test, 470049),
        ("ens_hybrid_d4_xgb", "unet_hybrid_d4", "xgboost", p_hd4_eval, p_hd4_test, p_xgb_eval, p_xgb_test, 1929889),
    ]

    for prefix, m1_name, m2_name, p1_ev, p1_te, p2_ev, p2_te, p_cnt in ensemble_pairs:
        # A. Simple Average (alpha = 0.5)
        row_avg = evaluate_ensemble(
            f"{prefix}_avg", "Simple Ensemble",
            y_eval, y_test, p1_ev, p1_te, p2_ev, p2_te,
            alpha=0.5, m1_name=m1_name, m2_name=m2_name, n_params=p_cnt
        )
        rows.append(row_avg)
        print(f"Simple Ensemble: {row_avg['model']:26s} | Eval PR={row_avg['eval_pr_auc']:.4f} | Test PR={row_avg['test_pr_auc']:.4f} | Test F1={row_avg['f1']:.4f}")

        # B. Weighted Average (alpha tuned on eval)
        best_alpha, opt_eval_pr = optimize_alpha(y_eval, p1_ev, p2_ev)
        row_w = evaluate_ensemble(
            f"{prefix}_weighted", "Weighted Ensemble",
            y_eval, y_test, p1_ev, p1_te, p2_ev, p2_te,
            alpha=best_alpha, m1_name=m1_name, m2_name=m2_name, n_params=p_cnt
        )
        rows.append(row_w)
        print(f"Weighted Ens:   {row_w['model']:26s} | alpha={best_alpha:.2f} | Eval PR={row_w['eval_pr_auc']:.4f} | Test PR={row_w['test_pr_auc']:.4f} | Test F1={row_w['f1']:.4f}")

    # 6. Save results table
    df = pd.DataFrame(rows)
    # Save predictions to PREDS_DIR for key models
    for _, r in df.iterrows():
        p_path = PREDS_DIR / f"{r['model']}.npz"
        np.savez_compressed(p_path, y_true=y_test, y_prob=r["p_test"])

    # Drop p_test before saving CSV
    df_clean = df.drop(columns=["p_test"])
    csv_path = OUT_DIR / "hybrid_ensemble_results.csv"
    df_clean.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved hybrid ensemble results table to {csv_path}")

    # 7. Update reports/dl/results_dl.csv
    results_dl_path = OUT_DIR / "results_dl.csv"
    if results_dl_path.exists():
        existing_dl = pd.read_csv(results_dl_path)
    else:
        existing_dl = pd.DataFrame()

    new_dl_rows = []
    # Add hybrid models and top ensembles to results_dl.csv
    export_models = ["unet_hybrid_base", "unet_hybrid_d4", "ens_base_xgb_weighted", "ens_d4_xgb_weighted", "ens_hybrid_d4_xgb_weighted"]
    for _, r in df.iterrows():
        # an ensemble with alpha 0 or 1 is just one of its parents: don't list it as a separate model
        if r["model"] in export_models and not (r["category"] == "Weighted Ensemble" and r["alpha"] in (0.0, 1.0)):
            new_dl_rows.append({
                "model": r["model"],
                "pr_auc": r["test_pr_auc"],
                "roc_auc": r["test_roc_auc"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "threshold": r["eval_threshold"],
                "best_f1": r["eval_f1"],
                "best_f1_threshold": r["eval_threshold"],
                "recall_at_1fpr": r["recall_at_1fpr"],
                "recall_at_5fpr": r["recall_at_5fpr"],
                "n_pixels": len(y_test.ravel()[y_test.ravel() != -1]),
                "fire_rate": float((y_test.ravel()[y_test.ravel() != -1] == 1).mean()),
            })

    if len(new_dl_rows) > 0:
        for nr in new_dl_rows:
            if len(existing_dl) > 0:
                existing_dl = existing_dl[existing_dl["model"] != nr["model"]]
            existing_dl = pd.concat([existing_dl, pd.DataFrame([nr])])
        existing_dl.to_csv(results_dl_path, index=False, float_format="%.4f")
        print(f"Updated {results_dl_path} with hybrid and ensemble models.")

    # 8. Generate plot and summary markdown
    fig_path = OUT_DIR / "hybrid_ensemble.png"
    plot_hybrid_ensemble(df_clean, fig_path)

    md_path = OUT_DIR / "hybrid_ensemble_summary.md"
    generate_summary_md(df_clean, md_path)

    print("\n" + "=" * 80)
    print("HYBRID & ENSEMBLE RESULTS SUMMARY")
    print("=" * 80)
    print(df_clean[["model", "category", "alpha", "eval_pr_auc", "test_pr_auc", "test_roc_auc", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
