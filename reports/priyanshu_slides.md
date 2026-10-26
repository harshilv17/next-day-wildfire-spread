# Priyanshu slides: Related work & where we stand (2 slides)

Content for the team deck: a title, 3 bullets and a figure per slide.

- **Finished 16:9 slides:** `reports/figures/slide1_full.png` and `reports/figures/slide2_full.png`.
- **Figure-only versions:** listed under each slide below.

## Slide 1: Related work at a glance

**Figure:** `reports/figures/slide1_related_work_table.png`. Huot et al.'s rows are highlighted.

- Other spread models use different data and metrics, so their scores are context only (white rows).
- Huot et al. 2022 built our dataset: their pixel models reach 0.225 PR AUC and the whole-patch CNN 0.284.
- FireCast has 91% recall but only 3.6% precision. This is why we report PR AUC, not accuracy.

## Slide 2: Where we stand

**Figure:** `reports/figures/slide2_where_we_stand.png`. Our bar is highlighted.

- Our XGBoost on 35 features (12 raw + 23 engineered): 0.257 PR AUC on the same test split (95% interval over fire events: 0.19-0.33).
- Between Huot's random forest (0.225) and CNN (0.284), both inside our interval; recent U-Nets reach about 0.35 (Esparza et al. 2026 preprint).
- Tuning moves PR AUC by about 0.01, spatial features by about 0.20. Phase 2: a CNN / U-Net on the whole patch.
