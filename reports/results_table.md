# Phase 1 results (test split, 1,689 patches, 6.73M labelled pixels, 1.25% fire)

Precision / recall / F1 at the decision threshold chosen on the eval split. Bold = best per column.

| Model | PR AUC | ROC AUC | Precision | Recall | F1 | Recall @ 1% FPR | Recall @ 5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0.106 | 0.633 | 0.357 | 0.273 | 0.309 | 0.273 | 0.273 |
| Logistic reg. (raw) | 0.146 | 0.798 | **0.378** | 0.266 | 0.312 | 0.281 | 0.319 |
| Random forest (raw) | 0.157 | 0.816 | 0.371 | 0.270 | 0.312 | 0.279 | 0.362 |
| XGBoost (raw) | 0.145 | 0.795 | 0.337 | 0.257 | 0.292 | 0.267 | 0.375 |
| XGBoost (tuned, all features) | **0.256** | **0.903** | 0.257 | 0.504 | **0.341** | **0.390** | **0.675** |
| LightGBM (tuned, all features) | 0.252 | 0.896 | 0.253 | **0.514** | 0.339 | 0.389 | 0.670 |

## Reference: Huot et al. (2022), PR AUC on the same test split

| Model | PR AUC |
|---|---:|
| Persistence | 0.115 |
| Logistic regression (+ 8 neighbours) | 0.198 |
| Random forest (+ 8 neighbours) | 0.225 |
| CNN autoencoder | 0.284 |

Paper LR / RF also use the 8 neighbouring pixels of every channel; numbers are from their pipeline, so they are a reference point, not a like-for-like comparison.
