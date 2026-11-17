"""Generate leakage-safe fire-event-grouped Out-of-Fold (OOF) GBM probability maps.

Phase 2, Task 2:
1. 5-fold fire-event-grouped cross validation on the train split (14,979 patches) using GroupKFold.
   - Fire events are identified by matching 16x16 elevation blocks across patches
     (see fire_events in reports/dl/checks/dl_checks.py).
   - Patches belonging to the same fire event are strictly kept within the same fold,
     guaranteeing zero event leakage between train and validation folds.
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
import argparse
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from reports.dl.checks.dl_checks import fire_events
from src.data import CHANNELS, PROCESSED_DIR, SIZE, load_split
from src.evaluate import metrics
from src.features import ALL_FEATURES, feature_maps, load_pixels
from src.train import BEST_PARAMS

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "dl"
EVENTS_CACHE = PROCESSED_DIR / "train_events.npy"
N_FOLDS = 5
CHUNK = 500  # Process patches in chunks to stay well within 8 GB RAM


def get_train_events(X_train, force_recompute=False):
    """Load or compute fire-event grouping IDs for training patches.

    Events are determined by matching 16x16 elevation blocks across patches
    (fire_events in reports/dl/checks/dl_checks.py).
    """
    train_npz = PROCESSED_DIR / "train.npz"
    cache_valid = False
    if EVENTS_CACHE.exists() and not force_recompute:
        try:
            events = np.load(EVENTS_CACHE)
            if events.shape == (len(X_train),) and np.issubdtype(events.dtype, np.integer):
                if train_npz.exists() and EVENTS_CACHE.stat().st_mtime >= train_npz.stat().st_mtime:
                    cache_valid = True
                elif not train_npz.exists():
                    cache_valid = True
        except Exception as e:
            print(f"Warning: Failed reading {EVENTS_CACHE} ({e}). Recomputing...")
            cache_valid = False

    if cache_valid:
        print(f"Loaded cached train fire events from {EVENTS_CACHE}")
        events = np.load(EVENTS_CACHE)
    else:
        print("Computing train fire events via elevation matching (fire_events)...")
        t0 = time.time()
        elev = X_train[:, CHANNELS.index("elevation")]
        events = fire_events(elev)
        print(f"Computed fire events in {time.time() - t0:.1f}s")
        np.save(EVENTS_CACHE, events)
        print(f"Cached fire events to {EVENTS_CACHE}")

    return events


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


def generate_oof(n_folds=N_FOLDS, seed=0, force_recompute_events=False):
    """Run fire-event-grouped OOF cross validation and full model inference."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not BEST_PARAMS.exists():
        raise FileNotFoundError(f"Missing {BEST_PARAMS}. Run Phase 1 training first.")
    params = json.loads(BEST_PARAMS.read_text())["xgboost"]

    # 1. Load train split
    print("Loading training data (patches and tabular pixels)...")
    X_train, y_train = load_split("train")
    n_train = len(X_train)
    assert n_train == 14979, f"Expected 14979 train patches, got {n_train}"

    train_pixels = load_pixels("train")
    patch_col = train_pixels["patch"].to_numpy()

    # 2. Derive fire-event groups
    train_events = get_train_events(X_train, force_recompute=force_recompute_events)
    assert len(train_events) == n_train, f"Expected {n_train} event IDs, got {len(train_events)}"

    unique_events, event_counts = np.unique(train_events, return_counts=True)
    n_events = len(unique_events)
    largest_event_size = int(event_counts.max())
    min_event_size = int(event_counts.min())
    mean_event_size = float(event_counts.mean())
    median_event_size = float(np.median(event_counts))
    print(f"Fire events summary: {n_events} distinct events across {n_train} patches")
    print(f"  Largest event: {largest_event_size} patches | Min: {min_event_size} | "
          f"Mean: {mean_event_size:.1f} | Median: {median_event_size:.1f}")

    # 3. Fire-event-grouped cross validation via GroupKFold
    gkf = GroupKFold(n_splits=n_folds)
    oof_train = np.full((n_train, SIZE, SIZE), np.nan, dtype=np.float32)
    fold_ids = np.full(n_train, -1, dtype=np.int32)
    fold_patch_counts = []
    fold_event_counts = []
    fold_times = []

    print(f"\n--- Starting {n_folds}-Fold Fire-Event-Grouped OOF Cross-Validation ---")
    patch_indices = np.arange(n_train)

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(patch_indices, groups=train_events)):
        t0 = time.time()
        val_patches_set = set(val_idx)
        trn_patches_set = set(trn_idx)

        # Strict leakage assertions: zero patch overlap AND zero event overlap
        patch_overlap = trn_patches_set & val_patches_set
        assert len(patch_overlap) == 0, f"Fold {fold}: Patch leakage detected! {len(patch_overlap)} overlapping patches"

        trn_events_set = set(train_events[trn_idx])
        val_events_set = set(train_events[val_idx])
        event_overlap = trn_events_set & val_events_set
        assert len(event_overlap) == 0, f"Fold {fold}: Event leakage detected! Overlapping events: {event_overlap}"

        # Select pixel rows for training fold (only pixels from trn_patches)
        trn_pixel_mask = np.isin(patch_col, trn_idx)
        X_trn_df = train_pixels.loc[trn_pixel_mask, ALL_FEATURES]
        y_trn_df = train_pixels.loc[trn_pixel_mask, "label"]

        n_val_events = len(val_events_set)
        print(f"Fold {fold + 1}/{n_folds}: {len(trn_idx)} train patches ({len(trn_events_set)} events, {len(X_trn_df):,} px) | "
              f"{len(val_idx)} val patches ({n_val_events} events)")

        # Train fold model
        fold_model = xgb.XGBClassifier(**params)
        fold_model.fit(X_trn_df, y_trn_df)

        # Predict OOF probabilities for validation fold patches
        val_prob = predict_patches_chunked(fold_model, X_train[val_idx])
        oof_train[val_idx] = val_prob
        fold_ids[val_idx] = fold

        fold_patch_counts.append(len(val_idx))
        fold_event_counts.append(n_val_events)
        elapsed = time.time() - t0
        fold_times.append(elapsed)
        print(f"Fold {fold + 1} completed in {elapsed:.1f}s")

    # Verify OOF integrity
    assert not np.isnan(oof_train).any(), "OOF predictions contain missing (NaN) values!"
    assert (oof_train >= 0.0).all() and (oof_train <= 1.0).all(), "OOF predictions outside [0, 1]!"
    assert (fold_ids >= 0).all(), "Unassigned fold IDs found!"
    assert sum(fold_patch_counts) == n_train, f"Expected {n_train} total patches across folds, got {sum(fold_patch_counts)}"

    # Score OOF against train ground truth
    oof_metrics = metrics(y_train, oof_train)
    print(f"\n>>> Full OOF Train PR AUC: {oof_metrics['pr_auc']:.4f} | ROC AUC: {oof_metrics['roc_auc']:.4f} | "
          f"Best-F1: {oof_metrics['best_f1']:.4f}")

    # 4. Train final model on all training pixels
    print(f"\n--- Training Final Full Model on All {n_train} Training Patches ---")
    t_full = time.time()
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(train_pixels[ALL_FEATURES], train_pixels["label"])
    print(f"Final model fitted in {time.time() - t_full:.1f}s")

    # 5. Predict evaluation split
    print("Generating predictions for eval split...")
    X_eval, y_eval = load_split("eval")
    oof_eval = predict_patches_chunked(final_model, X_eval)
    assert not np.isnan(oof_eval).any() and (oof_eval >= 0.0).all() and (oof_eval <= 1.0).all()
    eval_m = metrics(y_eval, oof_eval)
    eval_threshold = eval_m["best_f1_threshold"]
    print(f"Eval PR AUC: {eval_m['pr_auc']:.4f} | ROC AUC: {eval_m['roc_auc']:.4f} | Best-F1: {eval_m['best_f1']:.4f} "
          f"(thresh={eval_threshold:.4f})")

    # 6. Predict held-out test split
    print("Generating predictions for test split...")
    X_test, y_test = load_split("test")
    oof_test = predict_patches_chunked(final_model, X_test)
    assert not np.isnan(oof_test).any() and (oof_test >= 0.0).all() and (oof_test <= 1.0).all()
    test_m = metrics(y_test, oof_test, threshold=eval_threshold)
    print(f"Test PR AUC: {test_m['pr_auc']:.4f} | ROC AUC: {test_m['roc_auc']:.4f} | F1: {test_m['f1']:.4f}")

    # 7. Save artifacts to reports/dl/ and copy to data/processed/
    print("\nSaving output artifacts...")
    artifacts = [
        ("gbm_oof_train.npz", {"probabilities": oof_train, "patch_indices": patch_indices, "fold_ids": fold_ids, "event_ids": train_events}),
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
        "grouping_method": "GroupKFold",
        "group_definition": "fire_events (16x16 matching elevation blocks)",
        "group_source_function": "reports.dl.checks.dl_checks.fire_events",
        "n_folds": n_folds,
        "seed": seed,
        "train_patches": n_train,
        "eval_patches": len(X_eval),
        "test_patches": len(X_test),
        "n_events": n_events,
        "largest_event_size": largest_event_size,
        "min_event_size": min_event_size,
        "mean_event_size": mean_event_size,
        "median_event_size": median_event_size,
        "fold_patch_counts": fold_patch_counts,
        "fold_event_counts": fold_event_counts,
        "fold_runtimes_seconds": [round(t, 2) for t in fold_times],
        "zero_event_overlap": True,
        "complete_patch_coverage": True,
        "eval_threshold": float(eval_threshold),
        "metrics": {
            "train_oof": {k: (None if np.isnan(float(v)) else float(v)) for k, v in oof_metrics.items() if not isinstance(v, (int, np.integer))},
            "eval": {k: (None if np.isnan(float(v)) else float(v)) for k, v in eval_m.items() if not isinstance(v, (int, np.integer))},
            "test": {k: (None if np.isnan(float(v)) else float(v)) for k, v in test_m.items() if not isinstance(v, (int, np.integer))},
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


def main():
    parser = argparse.ArgumentParser(description="Generate fire-event-grouped OOF GBM probability maps.")
    parser.add_argument("--folds", type=int, default=N_FOLDS, help=f"Number of folds for GroupKFold (default: {N_FOLDS})")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--recompute-events", action="store_true", help="Force recomputation of train fire events")
    args = parser.parse_args()
    generate_oof(n_folds=args.folds, seed=args.seed, force_recompute_events=args.recompute_events)


if __name__ == "__main__":
    main()
