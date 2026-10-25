# Phase 1 results (test split, 1,689 patches, 6.73M labelled pixels, 1.25% fire)

Precision / recall / F1 at the decision threshold chosen on the eval split. Bold = best per column.

| Model | PR AUC | ROC AUC | Precision | Recall | F1 | Recall @ 1% FPR | Recall @ 5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0.106 | 0.633 | 0.357 | 0.273 | 0.309 | 0.273 | 0.273 |
| Logistic reg. (raw) | 0.146 | 0.798 | **0.378** | 0.266 | 0.312 | 0.281 | 0.319 |
| Random forest (raw) | 0.157 | 0.816 | 0.371 | 0.270 | 0.312 | 0.279 | 0.362 |
| XGBoost (raw) | 0.145 | 0.795 | 0.337 | 0.257 | 0.292 | 0.267 | 0.375 |
| XGBoost (tuned, all features) | **0.257** | **0.906** | 0.242 | **0.526** | **0.332** | **0.388** | **0.675** |
| LightGBM (tuned, all features) | 0.248 | 0.903 | 0.242 | 0.520 | 0.331 | 0.384 | 0.671 |

## Reference: Huot et al. (2022), PR AUC on the same test split

| Model | PR AUC |
|---|---:|
| Persistence | 0.115 |
| Logistic regression (+ 8 neighbours) | 0.198 |
| Random forest (+ 8 neighbours) | 0.225 |
| CNN autoencoder | 0.284 |

Paper LR / RF also use the 8 neighbouring pixels of every channel; numbers are from their pipeline, so they are a reference point, not a like-for-like comparison.

## Event-level evaluation

Test patches grouped into 257 events (crops of the same place, matched on elevation); 240 events have both fire and no-fire pixels and are scored. Each event counts once. Small event = at most 36 next-day fire pixels. Recall at the eval-chosen threshold. Bold = best per column.

| Model | PR AUC (pixels pooled) | PR AUC (mean over events) | PR AUC (median over events) | PR AUC (small events) | PR AUC (large events) | Recall (mean over events) | Events with any fire found |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0.106 | 0.074 | 0.021 | 0.074 | 0.074 | 0.154 | **0.608** |
| Logistic reg. (raw) | 0.146 | 0.097 | 0.033 | 0.102 | 0.092 | 0.093 | 0.408 |
| Random forest (raw) | 0.157 | 0.101 | 0.035 | 0.102 | 0.101 | 0.110 | 0.479 |
| XGBoost (raw) | 0.145 | 0.091 | 0.027 | 0.089 | 0.093 | 0.115 | 0.487 |
| XGBoost (tuned, all features) | **0.257** | **0.187** | **0.120** | 0.157 | **0.219** | 0.246 | 0.479 |
| LightGBM (tuned, all features) | 0.248 | 0.184 | 0.116 | **0.160** | 0.210 | **0.250** | 0.512 |
