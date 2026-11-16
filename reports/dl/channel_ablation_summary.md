# Deep Learning Channel-Group Ablation Summary

This document reports the empirical results of the controlled channel-group ablation experiments on the U-Net architecture (Phase 2, Task 3).

---

## 1. Experimental Methodology & Evaluation Protocol

All experiments used identical training and evaluation configurations:
* **Architecture:** 3-level U-Net (`depth=3`, `width=32`, 469k–489k parameters depending on input channels).
* **Hardware:** Apple Silicon M3 GPU (`mps`).
* **Optimization:** Adam optimizer (`lr=1e-3`, `batch_size=32`, `seed=0`).
* **Loss Function:** Masked Binary Cross-Entropy with `pos_weight=90.0` (ratio of non-fire to fire pixels on train). Pixels with label $-1$ masked out.
* **Early Stopping:** Evaluated PR AUC on the complete evaluation split ($1,877$ patches) after each epoch (`patience=5`).
* **Strict Evaluation Isolation:**
  * The decision threshold for precision, recall, and F1 was chosen strictly on the evaluation split by identifying the threshold maximizing F1 (`best_f1_threshold`).
  * Test split labels were never used for model training, tuning, threshold selection, or architecture modification.
  * Every configuration was scored exactly once on the held-out test split ($1,689$ patches, $6,727,699$ labelled pixels) at its fixed, eval-selected threshold.

### 1.1 Context on Run-to-Run Variance & Significance Threshold
As established in the technical validation of the Phase 2 models ([PR #14](https://github.com/harshilv17/next-day-wildfire-spread/pull/14), `reports/dl/checks/reruns.csv`), identical-seed reruns on GPU produce test PR AUC shifts up to $\pm 0.02$ (e.g. `unet_bce_dice` rerun $\Delta \text{PR AUC} = -0.0217$), and multi-seed runs exhibit a standard deviation of $\sim 0.008$. Consequently, performance variations within $\pm 0.02$ cannot be statistically distinguished from stochastic run-to-run noise without multi-seed averaging.

---

## 2. Quantitative Results

| Experiment | Dropped Group | Input Channels | Test PR AUC | Test ROC AUC | Precision | Recall | F1 Score | Recall @ 5% FPR | PR AUC Drop ($\Delta$) | Significance vs Baseline | Runtime (min) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **`unet_base`** | **None (Baseline)** | **12** | **0.2688** | **0.9289** | **0.2674** | **0.5198** | **0.3531** | **0.7027** | **0.0000** | Reference | 12.97 |
| **`unet_no_weather`** | Weather (`th`, `vs`, `tmm*`, `pr`, etc.) | 4 | 0.2672 | 0.9293 | 0.2622 | 0.5260 | 0.3500 | 0.6997 | +0.0016 | Within run-to-run noise | 7.27 |
| **`unet_no_vegetation`** | Vegetation (`NDVI`) | 11 | 0.2585 | 0.9199 | 0.2628 | 0.5128 | 0.3475 | 0.6996 | +0.0103 | Within run-to-run noise | 12.43 |
| **`unet_no_terrain`** | Terrain (`elevation`) | 11 | 0.2656 | 0.9215 | 0.2669 | 0.5102 | 0.3505 | 0.6870 | +0.0032 | Within run-to-run noise | 9.44 |
| **`unet_no_fire`** | Fire (`PrevFireMask`) | 11 | 0.1000 | 0.8689 | 0.1244 | 0.3643 | 0.1855 | 0.4526 | **+0.1688** | **Statistically significant** | 20.25 |
| **`unet_no_human`** | Human (`population`) | 11 | 0.2633 | 0.9232 | 0.2570 | 0.5220 | 0.3444 | 0.6984 | +0.0055 | Within run-to-run noise | 9.60 |

*Note: PR AUC Drop $\Delta = \text{PR AUC}_{\text{base}} - \text{PR AUC}_{\text{ablation}}$. A positive drop indicates degraded performance when the channel group is omitted.*

---

## 3. Comparative Analysis with Phase 1 Classical ML

| Feature Group | Phase 1 LightGBM Drop (`reports/ablation.csv`) | Phase 2 U-Net Drop (`channel_ablation.csv`) | Empirical Observations & Discussion |
| :--- | :---: | :---: | :--- |
| **Fire History (`PrevFireMask`)** | **+0.2003** (0.2483 $\rightarrow$ 0.0480) | **+0.1688** (0.2688 $\rightarrow$ 0.1000) | **Dominant contributor across both paradigms:** Excluding day-$t$ fire mask causes the largest performance drop by far. Wildfire progression exhibits strong spatial adjacency; without current fire perimeter data, predictive performance degrades substantially. The drop of $-0.1688$ is unambiguously far outside run-to-run noise ($\pm 0.02$). |
| **Vegetation (`NDVI`)** | **+0.0088** (0.2483 $\rightarrow$ 0.2395) | **+0.0103** (0.2688 $\rightarrow$ 0.2585) | **Within run-to-run noise:** While removing NDVI yields a single-seed drop of $+0.0103$ that is directionally consistent with the Phase 1 LightGBM drop (+0.0088), this difference is within the run-to-run variation observed in same-seed reruns ($\le 0.02$, [#14]). Thus, while live vegetation density is a plausible physical proxy for combustible fuel, this single-seed run cannot statistically isolate its contribution from stochastic noise. |
| **Weather (all 8 variables)** | **+0.0008** (0.2483 $\rightarrow$ 0.2475) | **+0.0016** (0.2688 $\rightarrow$ 0.2672) | **Within run-to-run noise:** In both paradigms, dropping all 8 atmospheric variables causes a negligible PR AUC drop ($< 0.002$), well within run-to-run noise ($\le 0.02$). Over a 24-hour prediction horizon, local fire perimeter geometry dominates, and weather features do not provide an independent signal discernible above baseline noise. |
| **Human (`population`)** | **−0.0023** (0.2483 $\rightarrow$ 0.2506) | **+0.0055** (0.2688 $\rightarrow$ 0.2633) | **Within run-to-run noise:** In Phase 1 LightGBM, omitting population shifted test PR AUC by $+0.0023$ (a slight gain), while in Phase 2 U-Net it shifted PR AUC by $-0.0055$ (a slight drop). Both deviations fall well within the run-to-run noise margin ($\le 0.02$); neither model family demonstrates a statistically meaningful dependence on the static population layer. |
| **Terrain (`elevation`)** | **−0.0028** (0.2483 $\rightarrow$ 0.2511) | **+0.0032** (0.2688 $\rightarrow$ 0.2656) | **Within run-to-run noise:** Omitting elevation produced minor shifts in both Phase 1 (+0.0028) and Phase 2 (+0.0032). Both deltas are strictly within run-to-run noise ($\le 0.02$), confirming that raw elevation alone does not impart a statistically detectable independent signal over 24-hour spread. |

---

## 4. Synthesis of Empirical Findings

Restricting conclusions strictly to what the data supports in light of technical validation variance:

1. **Dominance of Prior Fire Information:** Both classical GBDT and deep convolutional neural networks depend critically on prior fire localization (`PrevFireMask`). Excluding fire history causes a dramatic test PR AUC collapse of $-0.1688$ in U-Net (and $-0.2003$ in LightGBM). This is the only channel ablation whose effect is statistically significant and clearly outside run-to-run noise ($\le 0.02$).
2. **Vegetation, Terrain, Human, and Weather Within Run-to-Run Noise:** The single-seed test PR AUC differences observed when excluding vegetation ($+0.0103$), human population ($+0.0055$), terrain elevation ($+0.0032$), and all 8 weather variables ($+0.0016$) all fall within the run-to-run noise envelope ($\le 0.02$). Without multi-seed replications, none of these individual feature groups can be conclusively separated from baseline stochasticity.
3. **Computational Efficiency of Reduced Weather Representation:** Even though omitting all 8 weather channels leaves test PR AUC essentially unchanged (drop of $0.0016$ is within noise), reducing the network input from 12 channels to 4 channels cuts training time by ~44% (7.27 min vs 12.97 min), offering a compelling computational advantage for resource-constrained training.
