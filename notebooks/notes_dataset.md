# Dataset Notes: Huot et al. (2022)

**Paper:** *"Next Day Wildfire Spread: A Machine Learning Data Set to Predict Wildfire Spreading from Remote-Sensing Data"*  
**Authors:** Fantine Huot, R. Lily Hu, Nita Goyal, Tharun Sankar, Matthias Ihme, Yi-Fan Chen (Google Research & Stanford University)  
**Reference:** arXiv:2112.02447 [cs.CV] (2022) / IEEE TGRS.

---

### Key Bullets & Dataset Summary

* **Dataset Purpose & Scope:**
  * Curated benchmark dataset of historical wildfires across the contiguous United States (CONUS) spanning 2012 to 2020.
  * Formulated as a daily fire spread prediction task: given observations on day $t$, predict the binary wildfire spread mask on day $t+1$.
  * Total of **18,545 events** sampled as $64 \times 64\text{ km}$ patches aligned to a uniform **1 km spatial resolution**.
  * Every sample is conditioned on having at least one active fire pixel in the region on day $t$ (`PrevFireMask`).
  * Fire spread dynamics: 58% of events expand in burned area from $t$ to $t+1$, 39% shrink, and ~3% stay approximately constant.

* **The 12 Input Channels (Sources, Units, Native Resolutions):**
  1. `elevation`: Shuttle Radar Topography Mission (SRTM); units: meters ($\text{m}$); native resolution: $30\text{ m}$.
  2. `th` (Wind Direction): GRIDMET (University of Idaho); units: degrees clockwise from north ($0^\circ\text{–}360^\circ$, direction wind blows *from*); native resolution: $4\text{ km}$ (daily mean).
  3. `vs` (Wind Speed): GRIDMET; units: meters per second ($\text{m/s}$); native resolution: $4\text{ km}$ (daily mean).
  4. `tmmn` (Minimum Temperature): GRIDMET; units: Kelvin ($\text{K}$); native resolution: $4\text{ km}$ (daily mean).
  5. `tmmx` (Maximum Temperature): GRIDMET; units: Kelvin ($\text{K}$); native resolution: $4\text{ km}$ (daily mean).
  6. `sph` (Specific Humidity): GRIDMET; units: kilogram per kilogram ($\text{kg/kg}$); native resolution: $4\text{ km}$ (daily mean).
  7. `pr` (Precipitation): GRIDMET; units: millimeters ($\text{mm}$); native resolution: $4\text{ km}$ (daily accumulation).
  8. `pdsi` (Palmer Drought Severity Index): GRIDMET Drought; units: unitless index (typically $-10$ to $+10$); native resolution: $4\text{ km}$ (5-day composite).
  9. `NDVI` (Normalized Difference Vegetation Index): S-NPP NASA VIIRS (VNP13A1); units: unitless vegetation index (stored multiplied by $10{,}000$); native resolution: $500\text{ m}$ (8-day composite).
  10. `population`: Gridded Population of the World v4 (GPWv4, CIESIN); units: persons per square kilometer ($\text{people/km}^2$); native resolution: $1\text{ km}$ (5-year updates; proxy for human ignition and suppression).
  11. `erc` (Energy Release Component): NFDRS via GRIDMET; units: fuel moisture / fire potential index (proportional to potential energy release per unit area, $\text{BTU/ft}^2 / 100$); native resolution: $4\text{ km}$.
  12. `PrevFireMask`: MOD14A1 V6 (Terra/MODIS); units: discrete classification ($1 = \text{fire}, 0 = \text{no-fire}, -1 = \text{uncertain}$); native resolution: $1\text{ km}$ (daily composite).

* **Target Label (`FireMask`) and Uncertain ($-1$) Mask Encoding:**
  * Ground truth next-day label is `FireMask` from MOD14A1 V6 at day $t+1$ at $1\text{ km}$ resolution.
  * Encoded as: **`1` = active fire**, **`0` = no fire**, and **`-1` = uncertain/unknown** (due to cloud cover, heavy smoke, water, or sensor gaps).
  * In both loss functions and evaluation metrics, all pixels with label `-1` are strictly masked out / ignored.
  * Fire pixels are heavily imbalanced: active fire on day $t+1$ accounts for only ~1.1% of labelled pixels (~90:1 negative-to-positive ratio).

* **Train / Evaluation / Test Split Methodology:**
  * Temporal block splitting: all calendar weeks between 2012 and 2020 were partitioned at an **8 : 1 : 1 ratio** (approx. 80% train, 10% eval, 10% test).
  * A **1-day buffer** between weeks was excluded from sampling to prevent temporal autocorrelation and data leakage across splits.
  * Events separated by $>10\text{ km}$ were treated as independent fires.
  * Resulting split patch counts:
    * **Train:** 14,979 patches (15 TFRecords)
    * **Eval:** 1,877 patches (2 TFRecords)
    * **Test:** 1,689 patches (2 TFRecords)
    * **Total:** 18,545 patches.

* **Evaluation Metrics:**
  * **PR AUC (Precision-Recall Area Under Curve / Average Precision)** is the primary metric due to extreme class imbalance (~90:1). Accuracy and ROC AUC are misleadingly optimistic because true negatives dominate.
  * Secondary reported metrics: Positive-class Precision, Positive-class Recall, and F1 at chosen decision thresholds.

* **Baselines & Benchmark Results Reported in Huot et al. (Table II, 1 km pixel-level test set):**
  * **Persistence Baseline (predicts day $t$ mask for day $t+1$):**
    * PR AUC = **11.5%** ($0.115$)
  * **Logistic Regression (pixel-wise with $3 \times 3$ neighborhood context):**
    * PR AUC = **19.8%** ($0.198$) | Precision = 32.5% | Recall = 35.3%
    * Trained on balanced 1:1 fire/no-fire samples with class weighting factor $W=7$ for no-fire.
  * **Random Forest (pixel-wise with $3 \times 3$ neighborhood context):**
    * PR AUC = **22.5%** ($0.225$) | Precision = 26.3% | Recall = 46.9%
    * Hyperparameters: max depth = 15, class weighting factor $W=5$.
  * Both non-deep models see all 12 channels for the pixel **and its 8 neighbours** (108 inputs) and exclude the 1-pixel patch border.
    Our `logreg_raw` / `rf_raw` baselines use the single pixel only, so they score lower by design, not by bug.
  * **Convolutional Autoencoder (Spatial Deep Learning):**
    * PR AUC = **28.4%** ($0.284$) | Precision = 33.6% | Recall = 43.1%
    * Outperformed pixel models by learning 2D spatial context and fire propagation geometry.
  * **Coarser Scale Trade-off:**
    * When aggregating predictions to coarser spatial blocks (e.g. $8 \times 8\text{ km}$), PR AUC rises to **66.3%**, showing high confidence in regional fire presence at the cost of spatial granularity.

* **Feature Importance & Ablation Findings from the Paper:**
  * `PrevFireMask` is overwhelmingly the single most important predictor. Removing it causes the largest degradation in PR AUC.
  * Models without `PrevFireMask` collapsed to near-zero recall.
  * Among environmental variables, `NDVI` (vegetation fuel availability) and `elevation` (topography) provided the highest incremental gain when paired with `PrevFireMask`.
  * `erc` (energy release component) ranked next in tree-based importance.
  * `pr` (precipitation) had the lowest importance in the paper's models (likely because short-term precipitation is sparse and already captured in drought/fuel indices).
