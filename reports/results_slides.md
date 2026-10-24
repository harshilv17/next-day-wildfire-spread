# Results slides (Sachin) — content for 3 slides

Figures are in `reports/figures/`. Speaker notes under each slide.

---

## Slide 1 — Engineered features nearly double PR AUC

**Visual:** `eval_pr_curves.png` (left), `eval_pr_auc_bars.png` (right, optional)

| Model | PR AUC | ROC AUC | F1 | Recall @ 1% / 5% FPR |
|---|---:|---:|---:|---:|
| Persistence | 0.106 | 0.633 | 0.309 | 0.27 / 0.27 |
| Best raw-channel baseline (RF) | 0.157 | 0.816 | 0.312 | 0.28 / 0.36 |
| **XGBoost, tuned, 35 features** | **0.256** | **0.903** | **0.341** | **0.39 / 0.67** |
| LightGBM, tuned, 35 features | 0.252 | 0.896 | 0.339 | 0.39 / 0.67 |
| *Huot et al. CNN (reference)* | *0.284* | | | |

- Only 1.25% of pixels burn → PR AUC is the main metric (random = 0.0125; ROC AUC looks good for every model)
- Raw-channel models stop at persistence's recall (0.27): they only learn "fire stays where it is"
- Tuned models find **50% of next-day fire** at the eval-chosen threshold, twice the baselines
- XGBoost ≈ LightGBM → the gain comes from the features, not the library

*Speaker notes:* Threshold chosen on eval, applied once to test. Paper numbers come from a different pipeline
(their LR/RF also see 8 neighbouring pixels), so they are a reference point. Our persistence (0.106) matches theirs (0.115).

---

## Slide 2 — The model predicts a 2 km halo around today's fire

**Visual:** 3-4 rows cropped from `eval_prob_maps.png` + `eval_shap_summary.png` (or `eval_shap_groups.png`)

- Fire-neighbourhood features dominate: `dist_to_fire` 34% of gain, `prev_fire_n9` 22%, `prev_fire_n5` 14%
- SHAP by group: fire 2.30 log-odds vs weather 0.43, vegetation 0.24, terrain 0.21, human 0.14
- Physically sensible: closer to fire, hotter, drier air (high VPD, low humidity), more wind, more vegetation → higher risk
- Not causal: high ERC and high specific humidity act as region/season proxies
- **Wind:** the model does *not* push risk downwind (downwind ≈ upwind probability);
  in the data, strong winds (≥ 8 m/s) show 2.7× more fire on the "upwind" side → convention or confounder to check

*Speaker notes:* Refitted model reproduces the saved predictions exactly. The SHAP sample is 10k fire + 10k non-fire
test pixels (seed 0). The wind result is shown in `eval_downwind_check.png` if asked.

---

## Slide 3 — Errors: dying fires and long jumps

**Visual:** one row each from `eval_worst_fp.png` and `eval_worst_fn.png` + `eval_recall_by_distance.png`

- **False positives (123k px):** large, fragmented fires that shrink next day;
  93% of false alarms are within 2 km of today's fire. Only 35% of burning pixels still burn tomorrow
- **False negatives (42k px):** fire that jumps 10-40 km or explodes (8 → 260 px in one day)
- Recall by distance from today's fire: **0.98** burning, **0.95** at 1 km, **0.48** at 2 km, **< 0.02** from 3 km on.
  Yet 43% of next-day fire lies ≥ 3 km away
- Data caveat: 40 duplicate test patches + overlapping crops of the same fire → large fires counted more than once
- **Phase 2:** a CNN / U-Net can see fire-front shape, long-range spread and growth vs decay

*Speaker notes:* The worst patches were picked one per location (an elevation match), because the first pass
returned five crops of the same fire.
