"""Baseline models for next-day wildfire spread prediction on raw channels.

Run: python -m src.baselines (after running python -m src.features)

Models evaluated:
1. persistence: naive baseline predicting day t PrevFireMask clipped to 0 (no learning).
2. logreg_raw: StandardScaler (fitted on train only) + LogisticRegression (balanced weights).
3. rf_raw: RandomForestClassifier (100 trees, min_samples_leaf=50, subsampled to 1M rows for speed).
4. xgb_raw: XGBClassifier (hist method, scale_pos_weight matching train class ratio).

All models are trained strictly on the train split and evaluated on the test split
via evaluate() using full 2D probability maps (N, 64, 64).
"""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluate import RESULTS_CSV, evaluate
from src.features import CHANNELS, load_pixels

SEED = 0


def run_persistence(test, y_test_map):
    """Persistence baseline: predict that fire on day t+1 is identical to day t.

    PrevFireMask contains values {-1, 0, 1}. We clip to 0 so uncertain pixels (-1)
    and unburned pixels (0) both predict 0 probability, while active fire (1) predicts 1.0.
    """
    print("\n--- 1. Persistence Baseline ---")
    prob = test["PrevFireMask"].clip(lower=0).to_numpy().reshape(-1, 64, 64)
    return evaluate(y_test_map, prob, "persistence")


def run_logreg(train, test, y_test_map):
    """Logistic Regression on standardized raw channels.

    Standardization is required for logistic regression because L2 regularization
    penalizes coefficients uniformly; unscaled channels (e.g. NDVI ~ 5000 vs sph ~ 0.007)
    would improperly dominate the optimization.
    The scaler is fitted strictly on the training set to prevent data leakage.
    """
    print("\n--- 2. Logistic Regression (Raw Channels) ---")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[CHANNELS])
    X_test = scaler.transform(test[CHANNELS])
    y_train = train["label"].to_numpy()

    t0 = time.time()
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0

    prob = clf.predict_proba(X_test)[:, 1].reshape(-1, 64, 64)
    print(f"Logistic regression fit completed in {fit_time:.2f} s")
    return evaluate(y_test_map, prob, "logreg_raw")


def run_random_forest(train, test, y_test_map):
    """Random Forest Classifier on raw channels.

    Decision trees are invariant to monotonic scaling, so raw clipped channels are used directly.
    The training set contains ~3.95M rows; to keep runtime manageable (~40s vs ~4 minutes)
    without sacrificing statistical representation, we subsample 1,000,000 rows with seed 0.
    """
    print("\n--- 3. Random Forest (Raw Channels) ---")
    rng = np.random.default_rng(SEED)

    # Subsample train to 1M rows (seed 0) for speed, maintaining class ratio
    sample_size = min(1_000_000, len(train))
    sub_idx = rng.choice(len(train), size=sample_size, replace=False)
    X_train_sub = train.loc[sub_idx, CHANNELS]
    y_train_sub = train.loc[sub_idx, "label"].to_numpy()

    clf = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=50,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )

    t0 = time.time()
    clf.fit(X_train_sub, y_train_sub)
    fit_time = time.time() - t0

    X_test = test[CHANNELS]
    prob = clf.predict_proba(X_test)[:, 1].reshape(-1, 64, 64)
    print(f"Random forest fit (1M sample) completed in {fit_time:.2f} s")
    return evaluate(y_test_map, prob, "rf_raw")


def run_xgboost(train, test, y_test_map):
    """XGBoost Classifier on raw channels.

    Uses histogram-based split finding (tree_method='hist') for fast execution on millions of rows.
    Class weighting is applied via scale_pos_weight = (#no-fire / #fire) in the training table,
    matching the weighting convention used across Harshil's tuned models.
    """
    print("\n--- 4. XGBoost (Raw Channels) ---")
    X_train = train[CHANNELS]
    y_train = train["label"].to_numpy()

    # scale_pos_weight = negative_count / positive_count in the training table (~5.0)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    clf = XGBClassifier(
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
        scale_pos_weight=scale_pos_weight,
    )

    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0

    X_test = test[CHANNELS]
    prob = clf.predict_proba(X_test)[:, 1].reshape(-1, 64, 64)
    print(f"XGBoost fit (scale_pos_weight={scale_pos_weight:.2f}) completed in {fit_time:.2f} s")
    return evaluate(y_test_map, prob, "xgb_raw")


def main():
    print("Loading pixel tables (raw mode)...")
    train = load_pixels("train", "raw")
    test = load_pixels("test", "raw")
    print(f"Train table: {train.shape[0]:,} rows | Test table: {test.shape[0]:,} rows")

    # Reshape test ground truth labels into (N, 64, 64) maps for evaluate()
    y_test_map = test["label"].to_numpy().reshape(-1, 64, 64)


    # Run all baselines
    run_persistence(test, y_test_map)
    run_logreg(train, test, y_test_map)
    run_random_forest(train, test, y_test_map)
    run_xgboost(train, test, y_test_map)

    # Print final summary table
    print("\n" + "=" * 95)
    print("FINAL BASELINE RESULTS (reports/results.csv)")
    print("=" * 95)
    results = pd.read_csv(RESULTS_CSV)
    baseline_models = ["persistence", "logreg_raw", "rf_raw", "xgb_raw"]
    summary = results[results["model"].isin(baseline_models)].copy()
    print(summary.to_string(index=False))
    print("=" * 95)


if __name__ == "__main__":
    main()
