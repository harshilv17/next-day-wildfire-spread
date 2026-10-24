"""Shared metrics for every model.

Use: from src.evaluate import evaluate, metrics
     threshold = metrics(y_eval, p_eval)["best_f1_threshold"]   # chosen on eval, never on test
     evaluate(y_test, p_test, "lightgbm", threshold=threshold)
y_true / y_prob can be any shape (pixel vector or (N, 64, 64) maps); pixels with label -1 are ignored.
Pass maps where possible so reports/preds/ can be turned back into probability maps.
best_f1 is the F1 at the best threshold on the scored split itself (an upper bound);
precision / recall / f1 are at the eval-chosen threshold, the honest operating point.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve,
)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
RESULTS_CSV = REPORTS_DIR / "results.csv"
PREDS_DIR = REPORTS_DIR / "preds"


def metrics(y_true, y_prob, threshold=None):
    y_true, y_prob = np.ravel(y_true), np.ravel(y_prob)
    keep = y_true != -1
    y_true, y_prob = y_true[keep].astype(int), y_prob[keep]

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = np.argmax(f1[:-1])  # last PR point has no threshold

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    def recall_at(target):
        return tpr[np.searchsorted(fpr, target, side="right") - 1]

    at_threshold = (np.nan, np.nan, np.nan)
    if threshold is not None:
        at_threshold = precision_recall_fscore_support(
            y_true, y_prob >= threshold, average="binary", zero_division=0
        )[:3]

    return {
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": at_threshold[0],
        "recall": at_threshold[1],
        "f1": at_threshold[2],
        "threshold": np.nan if threshold is None else threshold,
        "best_f1": f1[best],
        "best_f1_threshold": thresholds[best],
        "recall_at_1fpr": recall_at(0.01),
        "recall_at_5fpr": recall_at(0.05),
        "n_pixels": int(keep.sum()),
        "fire_rate": y_true.mean(),
    }


def evaluate(y_true, y_prob, name, threshold=None):
    """Compute metrics, upsert a row in reports/results.csv, save preds to reports/preds/{name}.npz.

    threshold: decision threshold chosen on eval (see module docstring); None leaves precision/recall/f1 empty.
    """
    row = {"model": name, **metrics(y_true, y_prob, threshold)}

    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PREDS_DIR / f"{name}.npz", y_true=y_true, y_prob=y_prob)

    results = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else pd.DataFrame()
    if len(results):
        results = results[results["model"] != name]  # re-running a model replaces its row
    pd.concat([results, pd.DataFrame([row])]).to_csv(RESULTS_CSV, index=False, float_format="%.4f")

    print(", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    return row


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = (rng.random(10_000) < 0.05).astype(float)
    y[:500] = -1
    perfect = metrics(y, np.where(y == 1, 0.9, 0.1), threshold=0.5)
    assert perfect["pr_auc"] == perfect["roc_auc"] == perfect["best_f1"] == perfect["recall_at_1fpr"] == 1
    assert perfect["f1"] == perfect["precision"] == perfect["recall"] == 1
    assert metrics(y, np.where(y == 1, 0.9, 0.1), threshold=0.95)["recall"] == 0
    assert perfect["n_pixels"] == 9_500
    noise = metrics(y, rng.random(10_000))
    assert 0.45 < noise["roc_auc"] < 0.55 and noise["pr_auc"] < 0.1
    print("evaluate self-check ok")
