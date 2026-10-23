# Data Quality Audit & Feature Characteristics

**Author:** Sushant  
**Task:** S5 — Data Quality & Distribution Audit  
**Dataset:** Next Day Wildfire Spread (Huot et al. 2022) — Train Split ($N=14{,}979$ patches, $61{,}353{,}728$ pixels)

---

## 1. Channel Quality & Physical Range Audit

The table below summarizes all 12 input channels evaluated across the full training dataset ($61.35\text{M}$ pixels). Expected physical bounds are derived from sensor specifications and geophysical constraints across the contiguous United States (CONUS).

| Channel | Unit | Source | Expected Physical Range | Observed Min | Observed Max | Observed Mean | % Outside Physical Range | Action Taken |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `elevation` | $\text{m}$ | SRTM ($30\text{ m}$) | $[-86.0, 4421.0]$ | $-45.00$ | $4193.00$ | $896.57$ | $0.000\%$ | None needed (negative elevation is real topography below sea level) |
| `th` | $\text{degrees}$ | GRIDMET ($4\text{ km}$) | $[0.0, 360.0]$ | $-505870.06$ | $37735.63$ | $146.65$ | $0.229\%$ | `CLIP['th'] = (0, 360)` in `src/features.py` |
| `vs` | $\text{m/s}$ | GRIDMET ($4\text{ km}$) | $[0.0, 60.0]$ | $-82.65$ | $103.22$ | $3.63$ | $<0.001\%$ | `CLIP['vs'] = (0, None)` in `src/features.py` |
| `tmmn` | $\text{K}$ | GRIDMET ($4\text{ km}$) | $[253.15, 330.0]$ | $-444.69$ | $716.63$ | $281.85$ | $0.364\%$ | `CLIP['tmmn'] = (253.15, 330)` in `src/features.py` |
| `tmmx` | $\text{K}$ | GRIDMET ($4\text{ km}$) | $[253.15, 330.0]$ | $0.00$ | $1229.85$ | $297.72$ | $0.364\%$ | `CLIP['tmmx'] = (253.15, 330)` in `src/features.py` |
| `sph` | $\text{kg/kg}$ | GRIDMET ($4\text{ km}$) | $[0.0, 0.05]$ | $-0.13$ | $0.09$ | $0.007$ | $<0.001\%$ | `CLIP['sph'] = (0, None)` in `src/features.py` |
| `pr` | $\text{mm}$ | GRIDMET ($4\text{ km}$) | $[0.0, 500.0]$ | $-167.45$ | $56.21$ | $0.32$ | $9.777\%$ | `CLIP['pr'] = (0, None)` in `src/features.py` |
| `pdsi` | index | GRIDMET Drought ($4\text{ km}$) | $[-10.0, 10.0]$ | $-125.71$ | $52.27$ | $-0.77$ | $0.001\%$ | None needed (index anomalies are extremely rare, trees split invariantly) |
| `NDVI` | index | VIIRS VNP13A1 ($500\text{ m}$) | $[-2000, 10000]$ | $-9567.00$ | $9966.00$ | $5350.68$ | $0.356\%$ | Scaled: `c['NDVI'] / 10_000` in `src/features.py` |
| `population`| $\text{people/km}^2$| GPWv4 ($1\text{ km}$) | $[0.0, 100000.0]$ | $0.00$ | $27103.61$ | $30.46$ | $0.000\%$ | None needed (valid, zero-inflated) |
| `erc` | index | NFDRS via GRIDMET | $[0.0, 150.0]$ | $-1196.09$ | $2470.88$ | $53.47$ | $0.029\%$ | `CLIP['erc'] = (0, None)` in `src/features.py` |
| `PrevFireMask` | discrete | MOD14A1 V6 ($1\text{ km}$) | $\{-1, 0, 1\}$ | $-1.00$ | $1.00$ | $-0.002$ | $0.000\%$ | None needed ($-1$ masked/ignored in loss and evaluation) |

---

## 2. Notes on Feature Normalization & Transformations

* **Tree-based models (Random Forest, XGBoost, LightGBM):**
  * **No normalization or scaling required.** Decision tree algorithms partition feature spaces along orthogonal axes via threshold splits based on ranking/impurity (Gini, entropy, gradient/hessian gain).
  * Monotonic transformations (e.g. standard scaling or log scaling) leave tree splits invariant.
  * Tree models handle raw clipped physical values directly without degradation.

* **Linear models (Logistic Regression):**
  * **Standardization is strictly mandatory.** Logistic regression fits weights $w_j$ by gradient descent/L-BFGS penalized by an L2/L1 regularizer $\lambda \|w\|_p^p$.
  * If features have wildly different variances (e.g., raw unscaled `NDVI` in thousands vs. `sph` in thousandths), the regularization penalty dominates smaller-scale coefficients and causes optimization divergence.
  * *Implementation rule:* Use `StandardScaler(with_mean=True, with_std=True)` fitted **strictly on the training split**, then apply `.transform()` to eval/test splits to avoid data leakage.

* **Heavy-Tailed Channels & Transformations:**
  * **`population`:** Over $65\%$ of pixels have population density $= 0$, while urban margins spike above $25{,}000\text{ people/km}^2$. Highly right-skewed. A log transform $\log_{10}(\text{pop} + 1)$ is recommended if used in linear models or neural networks.
  * **`pr` (Precipitation):** Severe zero inflation ($>90\%$ of pixels have $0\text{ mm}$ rain during active wildfire events), with small positive values in remaining pixels and negative artifacts in $9.8\%$.
  * **`dist_to_fire` (Engineered):** Heavy right tail bounded at $64\sqrt{2} \approx 90.5\text{ km}$.

---

## 3. Five EDA Findings for Harshil (Feature Engineering & Pipeline Improvements)

1. **Extreme Wind Direction (`th`) Values & Modulo Wrapping:**
   * Raw `th` contains extreme outliers ranging from $-505{,}870^\circ$ to $+37{,}735^\circ$ ($0.23\%$ of pixels).
   * While `src/features.py` currently clips `th` to `(0, 360)`, `th` is a periodic circular variable. Clipping large negative numbers to $0^\circ$ (North) and positive numbers to $360^\circ$ introduces artificial directional distortion.
   * *Recommendation:* Apply modulo wrapping `th = th % 360` before clipping or trigonometric conversion (`wind_u`, `wind_v`).

2. **Severe Temperature Collinearity ($\rho = 0.92$):**
   * Minimum temperature (`tmmn`) and maximum temperature (`tmmx`) have a Spearman correlation of $0.92$.
   * Feeding both raw temperatures into linear models causes collinearity inflation.
   * *Recommendation:* Harshil's engineered `temp_range = tmmx - tmmn` and `vpd` capture the true physical drivers (vapor pressure deficit and diurnal flux) far better than raw `tmmn`.

3. **Precipitation (`pr`) Carries Near-Zero Predictive Signal:**
   * Precipitation has negative values in $9.8\%$ of pixels (GRIDMET numerical interpolation artifact).
   * Over $92\%$ of active fire pixels experience $0\text{ mm}$ rain. Spearman correlation between `pr` and active fire is virtually zero ($\rho \approx 0.008$), and Huot et al. found precipitation had the lowest feature importance across all models.
   * *Recommendation:* Verify in feature ablation whether `pr` can be pruned without performance drop.

4. **Long-Distance Fire Spreading & Spotting (>40% Occur >5 km Away):**
   * Evaluating pixel distances between day-$t$ and day-$(t+1)$ fires reveals that **$40.4\%$ of next-day fire pixels ignite more than $5\text{ km}$ away from the previous perimeter**, and **$26.1\%$ are more than $10\text{ km}$ away** (caused by wind-driven ember spotting and distinct fire complexes in the same patch).
   * Harshil's `upwind_fire` features currently check only $k=2\text{ km}$ and $k=4\text{ km}$ radii (`upwind_fire_k2`, `upwind_fire_k4`).
   * *Recommendation:* Add longer-range upwind/downwind sampling (e.g. $k=8$ or $k=12\text{ km}$) or rely heavily on `dist_to_fire` and `fire_wind_alignment` to capture long-distance spread and spotting.

5. **NDVI Deep Negative Values & Water Shadow Artifacts:**
   * `NDVI` has values down to $-9567$ (representing $-0.95$, typically open water, cloud shadows, or snow anomalies) in $0.36\%$ of pixels.
   * `src/features.py` divides by $10{,}000$ but does not clip the lower bound in `CLIP`.
   * *Recommendation:* Add `NDVI: (-2000, 10000)` or clip unscaled `ndvi` to $[-0.2, 1.0]$ so unburnable water bodies don't distort local rolling means (`ndvi_n5_mean`, `ndvi_anomaly`).
