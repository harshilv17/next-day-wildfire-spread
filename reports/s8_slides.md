# S8 slides: Dataset & EDA (3 slides)

## Slide 1: The dataset and why accuracy is useless
- 18,545 patches of US wildfires (2012–2020), 64×64 km at 1 km, 12 channels on day t → fire mask on day t+1
- Split by calendar week 8:1:1 with a 1-day buffer: 14,979 / 1,877 / 1,689 patches
- Next-day fire = 1.1% of labelled pixels (90:1); an all-zero model scores >98.7% accuracy → we report PR AUC
- Figure: `figures/eda_patch_fire_growth.png` (58% of fires grow, 39% shrink)

## Slide 2: What the raw data looks like
- Export artefacts: wind direction from −505,870° to 37,736°, temperatures from −445 K to 1,230 K, negative rain in 9.8% of pixels → clipped or wrapped
- ERC (75 vs 54) and elevation (1,414 vs 709 m) separate fire from no-fire; wind speed and humidity barely differ
- Day-t fire has |ρ| ≤ 0.09 with every raw channel → the signal is spatial, not in a pixel's own weather
- Figure: `figures/eda_fire_vs_nofire_features.png`

## Slide 3: From EDA to features, and what each group is worth
- 40% of new fire is more than 5 km from day-t fire → added `upwind_fire_k8` (ranks 5th of 35 by XGBoost gain)
- Ablation: dropping fire-neighbourhood features drops PR AUC 0.248 → 0.048; vegetation is the most useful environmental group (−0.009)
- Fire features alone: 0.218; environment adds +0.031 on top
- Figure: `figures/ablation_pr_auc.png`
