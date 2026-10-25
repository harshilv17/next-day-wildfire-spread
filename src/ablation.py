"""Feature-group ablation with the tuned LightGBM.

Run: python -m src.ablation   (after src.features and src.train)
Out: reports/ablation.csv, reports/figures/ablation_pr_auc.png

Every run uses the same hyperparameters from reports/best_params.json (fixed number of trees,
no re-tuning), so differences come from the features alone. Each run picks its F1 threshold on
eval and is scored on test with metrics(), not evaluate(), so reports/results.csv keeps only the
main models.

Note: upwind_fire_* and fire_wind_alignment use wind direction but sit in fire_neighbourhood,
so "no_weather" still sees some wind information.
"""
import json

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluate import REPORTS_DIR, metrics
from src.features import ALL_FEATURES, FEATURE_GROUPS, load_pixels
from src.train import BEST_PARAMS

RUNS = {
    "all": ALL_FEATURES,
    **{f"no_{group}": [f for f in ALL_FEATURES if f not in feats] for group, feats in FEATURE_GROUPS.items()},
    "fire_only": FEATURE_GROUPS["fire_neighbourhood"],
}
COLUMNS = ["pr_auc", "roc_auc", "precision", "recall", "f1", "recall_at_5fpr"]


def run(features, params, train, ev, test):
    model = lgb.LGBMClassifier(**params).fit(train[features], train["label"])
    threshold = metrics(ev["label"], model.predict_proba(ev[features])[:, 1])["best_f1_threshold"]
    return metrics(test["label"], model.predict_proba(test[features])[:, 1], threshold)


def plot(df, path):
    df = df.iloc[::-1]  # "all" on top
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=200)
    ax.barh(df["run"], df["pr_auc"], height=0.6, color="#2a78d6")
    ax.axvline(df.loc[df["run"] == "all", "pr_auc"].item(), color="#52514e", lw=1, ls="--")
    for y, (value, drop) in enumerate(zip(df["pr_auc"], df["pr_auc_drop"])):
        label = f"{value:.3f}" if abs(drop) < 5e-4 else f"{value:.3f} ({-drop:+.3f})"
        ax.text(value + 0.003, y, label, va="center", fontsize=7, color="#0b0b0b",
                bbox={"fc": "white", "ec": "none", "pad": 1}, zorder=3)
    ax.set_xlabel("Test PR AUC (LightGBM, best_params)", fontsize=8, color="#52514e")
    ax.set_xlim(0, df["pr_auc"].max() * 1.25)
    ax.tick_params(labelsize=7, colors="#52514e", length=0)
    ax.grid(axis="x", color="#e5e4e0", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    params = json.loads(BEST_PARAMS.read_text())["lightgbm"]
    train, ev, test = (load_pixels(split) for split in ("train", "eval", "test"))
    rows = []
    for name, features in RUNS.items():
        score = run(features, params, train, ev, test)
        rows.append({"run": name, "n_features": len(features), **{k: score[k] for k in COLUMNS}})
        print(f"{name:>24}: {len(features):2d} features, pr_auc={score['pr_auc']:.4f}")
    df = pd.DataFrame(rows)
    df["pr_auc_drop"] = df.loc[0, "pr_auc"] - df["pr_auc"]  # positive = worse than "all"
    df.to_csv(REPORTS_DIR / "ablation.csv", index=False, float_format="%.4f")
    plot(df, REPORTS_DIR / "figures" / "ablation_pr_auc.png")
