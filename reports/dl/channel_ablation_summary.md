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

---

## 2. Quantitative Results

| Experiment | Dropped Group | Input Channels | Test PR AUC | Test ROC AUC | Precision | Recall | F1 Score | Recall @ 5% FPR | PR AUC Drop ($\Delta$) | Runtime (min) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`unet_base`** | **None (Baseline)** | **12** | **0.2688** | **0.9289** | **0.2674** | **0.5198** | **0.3531** | **0.7027** | **0.0000** | 12.97 |
| **`unet_no_weather`** | Weather (`th`, `vs`, `tmm*`, `pr`, etc.) | 4 | 0.2672 | 0.9293 | 0.2622 | 0.5260 | 0.3500 | 0.6997 | +0.0016 | 7.27 |
| **`unet_no_vegetation`** | Vegetation (`NDVI`) | 11 | 0.2585 | 0.9199 | 0.2628 | 0.5128 | 0.3475 | 0.6996 | +0.0103 | 12.43 |
| **`unet_no_terrain`** | Terrain (`elevation`) | 11 | 0.2656 | 0.9215 | 0.2669 | 0.5102 | 0.3505 | 0.6870 | +0.0032 | 9.44 |
| **`unet_no_fire`** | Fire (`PrevFireMask`) | 11 | 0.1000 | 0.8689 | 0.1244 | 0.3643 | 0.1855 | 0.4526 | **+0.1688** | 20.25 |
| **`unet_no_human`** | Human (`population`) | 11 | 0.2633 | 0.9232 | 0.2570 | 0.5220 | 0.3444 | 0.6984 | +0.0055 | 9.60 |

---

## 3. Comparative Analysis with Phase 1 Classical ML

| Feature Group | Phase 1 LightGBM Drop (`reports/ablation.csv`) | Phase 2 U-Net Drop (`channel_ablation.csv`) | Empirical Observations & Discussion |
| :--- | :---: | :---: | :--- |
| **Fire History (`PrevFireMask`)** | **+0.2003** (0.2483 $\rightarrow$ 0.0480) | **+0.1688** (0.2688 $\rightarrow$ 0.1000) | **Dominant contributor across both paradigms:** Excluding day-$t$ fire mask causes the largest performance drop by far. Wildfire progression exhibits strong spatial adjacency; without current fire perimeter data, predictive performance degrades substantially. |
| **Vegetation (`NDVI`)** | **+0.0088** (0.2483 $\rightarrow$ 0.2395) | **+0.0103** (0.2688 $\rightarrow$ 0.2585) | **Important contributor in both paradigms:** Live vegetation density serves as an indicator of available combustible biomass. Removing NDVI results in the second-largest PR AUC decrease in both models. |
| **Weather (all 8 variables)** | **+0.0008** (0.2483 $\rightarrow$ 0.2475) | **+0.0016** (0.2688 $\rightarrow$ 0.2672) | **Relatively small independent contribution:** In both experiments, dropping all 8 atmospheric variables leads to only minor changes in PR AUC ($< 0.002$). One hypothesis is that over a 24-hour window, local active fire geometry and fuel availability provide strong predictive signals, while weather variations across the patch may have a less pronounced independent effect in this experimental setup. |
| **Human (`population`)** | **−0.0023** (0.2483 $\rightarrow$ 0.2506) | **+0.0055** (0.2688 $\rightarrow$ 0.2633) | **Paradigm divergence:** In Phase 1 LightGBM, excluding population slightly increased test PR AUC (+0.0023 gain), whereas in Phase 2 U-Net, excluding population caused a minor drop of 0.0055. This demonstrates differing model sensitivity to demographic features between the two model families. |
| **Terrain (`elevation`)** | **−0.0028** (0.2483 $\rightarrow$ 0.2511) | **+0.0032** (0.2688 $\rightarrow$ 0.2656) | **Paradigm divergence:** In Phase 1 LightGBM, excluding elevation resulted in a minor test PR AUC gain (+0.0028), whereas in U-Net it caused a minor drop of 0.0032. Neither paradigm shows a large sensitivity to the raw elevation channel alone. |

---

## 4. Synthesis of Empirical Findings

Restricting conclusions strictly to what the data supports:

1. **Dominance of Prior Fire Information:** Both classical GBDT and deep convolutional neural networks depend heavily on prior fire localization (`PrevFireMask`), which accounts for the vast majority of predictive capability.
2. **Consistent Role of Vegetation:** NDVI provides a consistent, measurable positive contribution across both modeling approaches.
3. **Limited Independent Weather Contribution:** In this 24-hour spread benchmark, excluding all 8 weather channels yielded only negligible drops in PR AUC for both GBDT and U-Net. A 4-channel U-Net without weather channels trained in ~56% of the time (7.27 min vs 12.97 min) while retaining PR AUC 0.2672.
4. **Divergent Behavior on Static Demographic & Topographic Channels:** Human population and raw elevation showed small, conflicting effects across paradigms (slight positive contribution in U-Net, slight negative contribution in LightGBM). Their marginal utility depends on the underlying model architecture.
