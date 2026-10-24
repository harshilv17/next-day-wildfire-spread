"""XGBoost + LightGBM on mode="full": random search scored on eval, final run on test.

Run: python -m src.train [xgboost|lightgbm]   (both by default, ~1 h on a laptop)
Out: reports/tuning_{model}.csv, reports/best_params.json, test rows in reports/results.csv

Each config trains on the subsampled train table, early-stops on 1M random labelled eval pixels,
and is scored by PR AUC on the full eval split. The best model per family is then scored on test.
best_params.json holds full constructor kwargs: XGBClassifier(**p["xgboost"]) / LGBMClassifier(**p["lightgbm"]).
"""
import json
import sys
import time

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from scipy.stats import loguniform, uniform
from sklearn.model_selection import ParameterSampler

from src.data import SIZE
from src.evaluate import REPORTS_DIR, evaluate, metrics
from src.features import ALL_FEATURES, SEED, load_pixels

N_CONFIGS = 20
MAX_TREES = 1000
EARLY_STOP = 50
ES_ROWS = 1_000_000
BEST_PARAMS = REPORTS_DIR / "best_params.json"

SPACES = {
    "xgboost": {
        "max_depth": [4, 6, 8, 10],
        "learning_rate": loguniform(0.02, 0.3),
        "min_child_weight": [1, 5, 20, 100],
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(0.1, 10),
        "gamma": [0, 0.1, 1],
    },
    "lightgbm": {
        "num_leaves": [15, 31, 63, 127, 255],
        "learning_rate": loguniform(0.02, 0.3),
        "min_child_samples": [20, 100, 500, 2000],
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(0.1, 10),
        "min_split_gain": [0, 0.1, 1],
    },
}
FIXED = {
    "xgboost": {"tree_method": "hist", "n_jobs": -1, "random_state": SEED},
    "lightgbm": {"subsample_freq": 1, "n_jobs": -1, "random_state": SEED, "verbose": -1},
}


def fit(name, kwargs, X, y, X_es, y_es):
    """Fit with early stopping; return model and number of trees kept."""
    if name == "xgboost":
        model = xgb.XGBClassifier(
            **kwargs, n_estimators=MAX_TREES, early_stopping_rounds=EARLY_STOP, eval_metric="aucpr"
        )
        model.fit(X, y, eval_set=[(X_es, y_es)], verbose=False)
        return model, model.best_iteration + 1
    # metric set here, not via eval_metric: otherwise binary_logloss is tracked too and early stopping
    # fires on it after 1 tree (train is 1:5, eval ~1:70, so eval logloss rises from the start)
    model = lgb.LGBMClassifier(**kwargs, n_estimators=MAX_TREES, metric="average_precision")
    model.fit(X, y, eval_X=(X_es,), eval_y=(y_es,), callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return model, model.best_iteration_


def tune(name, train, es, ev, scale_pos_weight):
    log_path = REPORTS_DIR / f"tuning_{name}.csv"
    rows, best = [], None
    for i, params in enumerate(ParameterSampler(SPACES[name], N_CONFIGS, random_state=SEED)):
        kwargs = {**params, **FIXED[name], "scale_pos_weight": scale_pos_weight}
        start = time.time()
        model, n_trees = fit(name, kwargs, train[ALL_FEATURES], train["label"], es[ALL_FEATURES], es["label"])
        score = metrics(ev["label"], model.predict_proba(ev[ALL_FEATURES])[:, 1])
        rows.append({"config": i, **params, "n_estimators": n_trees, **score, "seconds": round(time.time() - start)})
        pd.DataFrame(rows).to_csv(log_path, index=False, float_format="%.4f")  # rewritten each config
        print(f"{name} {i + 1}/{N_CONFIGS}: pr_auc={score['pr_auc']:.4f} trees={n_trees} {rows[-1]['seconds']}s")
        if best is None or score["pr_auc"] > best[0]:
            best = (score["pr_auc"], model, {**kwargs, "n_estimators": n_trees})
    return best[1], best[2]


if __name__ == "__main__":
    names = sys.argv[1:] or list(SPACES)
    train, ev, test = load_pixels("train"), load_pixels("eval"), load_pixels("test")
    es = ev[ev["label"] != -1].sample(ES_ROWS, random_state=SEED)
    scale_pos_weight = (train["label"] == 0).sum() / (train["label"] == 1).sum()  # train only
    best_params = json.loads(BEST_PARAMS.read_text()) if BEST_PARAMS.exists() else {}

    for name in names:
        model, params = tune(name, train, es, ev, scale_pos_weight)
        best_params[name] = params
        BEST_PARAMS.write_text(json.dumps(best_params, indent=2))
        prob = model.predict_proba(test[ALL_FEATURES])[:, 1]
        evaluate(test["label"].to_numpy().reshape(-1, SIZE, SIZE), prob.reshape(-1, SIZE, SIZE), name)
