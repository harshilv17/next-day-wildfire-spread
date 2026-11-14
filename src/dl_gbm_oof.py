"""Generate leakage-safe patch-grouped Out-of-Fold (OOF) GBM probability maps.

Phase 2, Task 2:
1. 5-fold patch-grouped cross validation on the train split (14,979 patches).
   - Every patch is assigned to exactly one validation fold.
   - No patch from a validation fold is ever used to train that fold's model.
   - All pixels for each validation patch are predicted, producing an OOF map of shape (14979, 64, 64).
2. Final full model trained on all training patches.
   - Predicts eval probability map: (1877, 64, 64).
   - Predicts test probability map: (1689, 64, 64).
3. Saves artifacts to reports/dl/ and data/processed/ for hybrid U-Net ingestion (13th channel).

Outputs:
  reports/dl/gbm_oof_train.npz
  reports/dl/gbm_oof_eval.npz
  reports/dl/gbm_oof_test.npz
  reports/dl/gbm_oof_metadata.json
"""
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold

from src.data import PROCESSED_DIR, SIZE, load_split
from src.evaluate import metrics
from src.features import ALL_FEATURES, feature_maps, load_pixels
from src.train import BEST_PARAMS

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "dl"
N_FOLDS = 5
CHUNK = 500  # Process patches in chunks to stay well within 8 GB RAM


def predict_patches_chunked(model, X, chunk_size=CHUNK):
    """Predict GBM probabilities for patches X (N, 12, 64, 64) in chunks.

    Returns:
        prob: (N, 64, 64) float32 probability array in [0, 1].
    """
    n = len(X)
    prob = np.empty((n, SIZE, SIZE), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        xc = X[start:end]
        maps = feature_maps(xc)
        # Flatten features into 2D tabular matrix: (n_chunk * 4096, 35)
        feat_matrix = np.column_stack([maps[f].ravel() for f in ALL_FEATURES])
        prob[start:end] = model.predict_proba(feat_matrix)[:, 1].reshape(end - start, SIZE, SIZE).astype(np.float32)
    return prob


def generate_oof(n_folds=N_FOLDS, seed=0):
    """Run patch-grouped OOF cross validation and full model inference."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not BEST_PARAMS.exists():
        raise FileNotFoundError(f"Missing {BEST_PARAMS}. Run Phase 1 training first.")
    params = json.loads(BEST_PARAMS.read_text())["xgboost"]

    # 1. Load train split
    print(f"Loading training data (patches and tabular pixels)...")
    X_train, y_train = load_split("train")
    n_train = len(X_train)
    assert n_train == 14979, f"Expected 14979 train patches, got {n_train}"

    train_pixels = load_pixels("train")
    patch_col = train_pixels["patch"].to_numpy()

    # 2. Patch-grouped 5-fold cross validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_train = np.full((n_train, SIZE, SIZE), np.nan, dtype=np.float32)
    fold_ids = np.full(n_train, -1, dtype=np.int32)
    fold_patch_counts = []
    fold_times = []

    print(f"\n--- Starting {n_folds}-Fold Patch-Grouped OOF Cross-Validation ---")
    patch_indices = np.arange(n_train)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(patch_indices)):
        t0 = time.time()
        val_patches_set = set(val_idx)
        trn_patches_set = set(trn_idx)

        # Strict leakage assertion: zero patch overlap
        overlap = trn_patches_set & val_patches_set
        assert len(overlap) == 0, f"Fold {fold}: Data leakage detected! Overlapping patches: {len(overlap)}"

        # Select pixel rows for training fold (only pixels from trn_patches)
        trn_pixel_mask = np.isin(patch_col, trn_idx)
        X_trn_df = train_pixels.loc[trn_pixel_mask, ALL_FEATURES]
        y_trn_df = train_pixels.loc[trn_pixel_mask, "label"]

        print(f"Fold {fold + 1}/{n_folds}: {len(trn_idx)} train patches ({len(X_trn_df):,} px) | "
              f"{len(val_idx)} val patches")

        # Train fold model
        fold_model = xgb.XGBClassifier(**params)
        fold_model.fit(X_trn_df, y_trn_df)

        # Predict OOF probabilities for validation fold patches
        val_prob = predict_patches_chunked(fold_model, X_train[val_idx])
        oof_train[val_idx] = val_prob
        fold_ids[val_idx] = fold

        fold_patch_counts.append(len(val_idx))
        elapsed = time.time() - t0
        fold_times.append(elapsed)
        print(f"Fold {fold + 1} completed in {elapsed:.1f}s")

    # Verify OOF integrity
    assert not np.isnan(oof_train).any(), "OOF predictions contain missing (NaN) values!"
    assert (oof_train >= 0.0).all() and (oof_train <= 1.0).all(), "OOF predictions outside [0, 1]!"
    assert (fold_ids >= 0).all(), "Unassigned fold IDs found!"

    # Score OOF against train ground truth
    oof_metrics = metrics(y_train, oof_train)
    print(f"\n>>> Full OOF Train PR AUC: {oof_metrics['pr_auc']:.4f} | ROC AUC: {oof_metrics['roc_auc']:.4f} | "
          f"F1: {oof_metrics['f1']:.4f}")

    # 3. Train final model on all training pixels
    print(f"\n--- Training Final Full Model on All {n_train} Training Patches ---")
    t_full = time.time()
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(train_pixels[ALL_FEATURES], train_pixels["label"])
    print(f"Final model fitted in {time.time() - t_full:.1f}s")

    # 4. Predict evaluation split
    print("Generating predictions for eval split...")
    X_eval, y_eval = load_split("eval")
    oof_eval = predict_patches_chunked(final_model, X_eval)
    assert not np.isnan(oof_eval).any() and (oof_eval >= 0.0).all() and (oof_eval <= 1.0).all()
    eval_m = metrics(y_eval, oof_eval)
    eval_threshold = eval_m["best_f1_threshold"]
    print(f"Eval PR AUC: {eval_m['pr_auc']:.4f} | ROC AUC: {eval_m['roc_auc']:.4f} | Best-F1: {eval_m['best_f1']:.4f} "
          f"(thresh={eval_threshold:.4f})")

    # 5. Predict held-out test split
    print("Generating predictions for test split...")
    X_test, y_test = load_split("test")
    oof_test = predict_patches_chunked(final_model, X_test)
    assert not np.isnan(oof_test).any() and (oof_test >= 0.0).all() and (oof_test <= 1.0).all()
    test_m = metrics(y_test, oof_test, threshold=eval_threshold)
    print(f"Test PR AUC: {test_m['pr_auc']:.4f} | ROC AUC: {test_m['roc_auc']:.4f} | F1: {test_m['f1']:.4f}")

    # 6. Save artifacts to reports/dl/ and copy to data/processed/
    print("\nSaving output artifacts...")
    artifacts = [
        ("gbm_oof_train.npz", {"probabilities": oof_train, "patch_indices": patch_indices, "fold_ids": fold_ids}),
        ("gbm_oof_eval.npz", {"probabilities": oof_eval, "patch_indices": np.arange(len(X_eval), dtype=np.int32)}),
        ("gbm_oof_test.npz", {"probabilities": oof_test, "patch_indices": np.arange(len(X_test), dtype=np.int32)}),
    ]

    for filename, arrays in artifacts:
        target_report = OUT_DIR / filename
        target_proc = PROCESSED_DIR / filename
        np.savez_compressed(target_report, **arrays)
        np.savez_compressed(target_proc, **arrays)
        print(f"Saved {target_report} and {target_proc} (shape: {arrays['probabilities'].shape})")

    metadata = {
        "model": "xgboost",
        "parameters": params,
        "n_folds": n_folds,
        "seed": seed,
        "train_patches": n_train,
        "eval_patches": len(X_eval),
        "test_patches": len(X_test),
        "fold_patch_counts": fold_patch_counts,
        "fold_runtimes_seconds": [round(t, 2) for t in fold_times],
        "eval_threshold": float(eval_threshold),
        "metrics": {
            "train_oof": {k: float(v) for k, v in oof_metrics.items() if not isinstance(v, (int, np.integer))},
            "eval": {k: float(v) for k, v in eval_m.items() if not isinstance(v, (int, np.integer))},
            "test": {k: float(v) for k, v in test_m.items() if not isinstance(v, (int, np.integer))},
        },
    }

    meta_path = OUT_DIR / "gbm_oof_metadata.json"
    meta_proc = PROCESSED_DIR / "gbm_oof_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(meta_proc, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    return metadata


if __name__ == "__main__":
    meta = generate_oof(n_folds=5, seed=0)
    print("\nGBM OOF generation completed successfully!")

