# Phase 1 vs Phase 2 results (test split, 1,689 patches, 6.73M labelled pixels, 1.25% fire)

Mean ± standard deviation (sample std) over seeds 0-2 where several runs exist (`reports/dl/checks/seeds.csv`; depth-3 U-Net from its three runs); other rows are single runs. F1 at the threshold chosen on the eval split. XGBoost over 3 seeds is Priyanshu's refit with the committed hyperparameters.

| Model | Phase | Runs | PR AUC | ROC AUC | F1 | Recall @ 1% FPR | Recall @ 5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 1 | 1 | 0.106 | 0.633 | 0.309 | 0.273 | 0.273 |
| Logistic reg. (raw) | 1 | 1 | 0.146 | 0.798 | 0.312 | 0.281 | 0.319 |
| Random forest (raw) | 1 | 1 | 0.157 | 0.816 | 0.312 | 0.279 | 0.362 |
| XGBoost (raw) | 1 | 1 | 0.145 | 0.795 | 0.292 | 0.267 | 0.375 |
| XGBoost (tuned) | 1 | 1 | 0.257 | 0.906 | 0.332 | 0.388 | 0.675 |
| LightGBM (tuned) | 1 | 1 | 0.248 | 0.903 | 0.331 | 0.384 | 0.671 |
| XGBoost (tuned) | 1 | 3 seeds | 0.255 ± 0.002 | 0.905 ± 0.001 | 0.330 ± 0.000 | 0.388 ± 0.001 | 0.673 ± 0.001 |
| U-Net final (depth 4, focal) | 2 | 3 seeds | 0.344 ± 0.007 | 0.934 ± 0.009 | 0.399 ± 0.003 | 0.459 ± 0.003 | 0.783 ± 0.006 |
| CNN (BCE) | 2 | 3 seeds | 0.249 ± 0.010 | 0.878 ± 0.011 | 0.335 ± 0.008 | 0.378 ± 0.009 | 0.653 ± 0.006 |
| U-Net depth 3 (focal) | 2 | 3 seeds | 0.275 ± 0.008 | 0.923 ± 0.003 | 0.349 ± 0.006 | 0.397 ± 0.004 | 0.711 ± 0.005 |
| CNN (BCE, reference run) | 2 | 1 (seed 0) | 0.257 | 0.882 | 0.329 | 0.379 | 0.652 |
| U-Net (BCE, reference run) | 2 | 1 (seed 0) | 0.269 | 0.929 | 0.353 | 0.393 | 0.703 |
| U-Net (BCE) | 2 | 1 (seed 0) | 0.258 | 0.922 | 0.354 | 0.384 | 0.700 |
| U-Net (BCE + Dice) | 2 | 1 (seed 0) | 0.269 | 0.924 | 0.355 | 0.395 | 0.704 |
| U-Net depth 4 (BCE) | 2 | 1 (seed 0) | 0.337 | 0.934 | 0.391 | 0.446 | 0.784 |
| Hybrid U-Net depth 4 (+ XGBoost channel) | 2 | 1 (seed 0) | 0.335 | 0.942 | 0.388 | 0.445 | 0.770 |
| Ensemble U-Net depth 3 + XGBoost (weighted) | 2 | 1 (seed 0) | 0.274 | 0.931 | 0.355 | 0.398 | 0.700 |

## Event-level evaluation

Test patches grouped into 257 events (crops of the same place, matched on elevation, as in Phase 1); 240 events have both fire and no-fire pixels and are scored. Each event counts once. Small event = at most 36 next-day fire pixels. Recall at the eval-chosen threshold. Seed-0 runs. Bold = best per column.

| Model | PR AUC (pixels pooled) | PR AUC (mean over events) | PR AUC (median over events) | PR AUC (small events) | PR AUC (large events) | Recall (mean over events) | Events with any fire found |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0.106 | 0.074 | 0.021 | 0.074 | 0.074 | 0.154 | 0.608 |
| XGBoost (tuned) | 0.257 | 0.187 | 0.120 | 0.157 | 0.219 | 0.246 | 0.479 |
| LightGBM (tuned) | 0.248 | 0.184 | 0.116 | 0.160 | 0.210 | 0.250 | 0.512 |
| CNN (BCE) | 0.259 | 0.180 | 0.108 | 0.161 | 0.201 | 0.238 | 0.525 |
| U-Net depth 3 (focal) | 0.269 | 0.236 | 0.152 | 0.225 | 0.247 | 0.319 | 0.600 |
| U-Net final (depth 4, focal) | **0.338** | **0.506** | **0.494** | **0.587** | **0.419** | **0.635** | **0.988** |

95% intervals over fire events (Priyanshu, `reports/dl/checks/ci_events.csv`, 1,000 resamples; his XGBoost row is a refit, 0.2576):

| Model | PR AUC | 2.5% | 97.5% |
|---|---:|---:|---:|
| unet_final_s0 | 0.3376 | 0.2928 | 0.3854 |
| cnn_bce | 0.2591 | 0.1801 | 0.3334 |
| xgboost | 0.2576 | 0.1900 | 0.3220 |
| persistence | 0.1065 | 0.0591 | 0.1611 |

Paired difference against XGBoost on the same resamples (Priyanshu, `ci_diffs_seeds_vs_xgboost.csv`):

| Model | b | diff | lo95 | hi95 | share_a_better |
|---|---:|---:|---:|---:|---:|
| unet_final_s0 | xgboost | 0.0800 | 0.0577 | 0.1131 | 1.0000 |
| unet_final_s1 | xgboost | 0.0853 | 0.0586 | 0.1221 | 1.0000 |
| unet_final_s2 | xgboost | 0.0938 | 0.0777 | 0.1176 | 1.0000 |
| cnn_bce | xgboost | 0.0014 | -0.0113 | 0.0133 | 0.5720 |
| cnn_bce_s1 | xgboost | -0.0093 | -0.0222 | 0.0050 | 0.1150 |
| cnn_bce_s2 | xgboost | -0.0183 | -0.0294 | -0.0080 | 0.0010 |

Mean PR AUC over events (each event once), events resampled:

| Model | mean over events | lo95 | hi95 | diff vs xgboost | diff lo95 | diff hi95 | events better |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgboost | 0.187 | 0.161 | 0.215 | 0.000 | 0.000 | 0.000 | 0.000 |
| unet_final_s0 | 0.506 | 0.480 | 0.535 | 0.318 | 0.288 | 0.352 | 0.921 |
| unet_final_s1 | 0.510 | 0.483 | 0.537 | 0.322 | 0.292 | 0.354 | 0.929 |
| unet_final_s2 | 0.493 | 0.465 | 0.522 | 0.305 | 0.274 | 0.338 | 0.929 |
