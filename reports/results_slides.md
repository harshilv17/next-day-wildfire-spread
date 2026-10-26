# Results slides (Sachin)

Figures are in `reports/figures/`. All numbers are on the test split.

---

## Slide 1: Engineered features beat every baseline

**Figure:** `reports/figures/eval_pr_curves.png`

- Tuned XGBoost: **PR AUC 0.257**, vs 0.145-0.157 for raw-channel models and 0.106 for persistence (LightGBM 0.248)
- Finds 39% of next-day fire at 1% FPR and 68% at 5% FPR; best F1 0.332
- Between Huot et al.'s neighbour RF (0.225) and their CNN (0.284)

---

## Slide 2: What the model learned

**Figure:** `reports/figures/eval_prob_maps.png` (2-3 rows) or `reports/figures/eval_downwind_check.png`

- Predicts a halo around today's fire; fire-neighbourhood features dominate (SHAP 2.29 vs 0.40 for weather)
- Physically sensible signs: near fire, dry air, strong wind, fire upwind → higher risk (ERC is the exception)
- Pushes risk downwind; train data agrees: downwind/upwind ratio grows from 1.135 (calm) to 3.100 (≥ 8 m/s)

---

## Slide 3: Where it fails → Phase 2

**Figure:** `reports/figures/eval_recall_by_distance.png`

- False alarms around dying fires: 88.8% within 2 km of today's fire; only 35.3% of burning pixels still burn tomorrow
- Misses long-range spread: recall 0.988 at the fire, 0.648 at 2 km, 0 beyond 5 km
- Fixed-radius halo, no fire-front shape → Phase 2: CNN / U-Net that sees the whole patch
