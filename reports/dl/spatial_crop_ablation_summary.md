# Spatial Crop Ablation Report (Phase 2, Task 1c)

## 1. Executive Summary

This study investigates whether reducing the spatial input context during training impairs next-day wildfire spread prediction.
We compare:
1. **Full Spatial Training (`unet_base`):** U-Net trained on full 64x64 patches (4,096 km^2).
2. **Center-Crop Training (`unet_crop32`):** U-Net trained on center 32x32 crops (1,024 km^2, 1/4 the spatial area).

**Methodological Rigour:** In contrast to earlier naive evaluations where crop models were scored only on center pixels (which artificially inflated fire density and produced non-comparable scores), **both models in this study are evaluated and tested on the exact same full 64x64 patches**. Because the U-Net architecture is fully convolutional, weights learned on 32x32 inputs apply directly to 64x64 inference.

---

## 2. Quantitative Results Matrix

| Configuration | Model | Train Size | Eval / Test Size | Parameters | Best Epoch | Duration (s) | Eval PR AUC | Test PR AUC | Delta vs Base | Test ROC AUC | Test Precision | Test Recall | Test F1 | Test Rec@1%FPR | Test Rec@5%FPR |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full Training** | `unet_base` | 64x64 | 64x64 | 469,761 | 7 | 778.2 | 0.2188 | 0.2688 | Reference | 0.9289 | 0.2674 | 0.5198 | 0.3531 | 0.3926 | 0.7027 |
| **Center Crop** | `unet_crop32` | 32x32 | 64x64 | 469,761 | 5 | 148.2 | 0.1333 | 0.1721 | -0.0967 | 0.9014 | 0.1891 | 0.4777 | 0.2709 | 0.2775 | 0.6177 |

---

## 3. Analysis & Key Insights

### 1. Result
- **Test PR AUC:** 0.2688 (full 64x64 training) vs 0.1721 (centre 32x32 training), **-0.0967**.
- **Eval PR AUC:** 0.2188 vs 0.1333 (-0.0855).
- **Test F1:** 0.3531 vs 0.2709 (-0.0822); **recall at 1% FPR:** 0.3926 vs 0.2775 (-0.1151).
- The drop is far larger than run-to-run noise (same-seed reruns move test PR AUC by up to 0.02), so **training on the centre crop and predicting on full patches clearly does not work**. Single seed (0), weighted BCE.

### 2. What this does and does not show
This is a **centre** crop, the same 32x32 window of every patch. It is not the **random** 32x32 crops Huot et al. (2022) train on, so it does not test their setup. Three effects change together and cannot be separated here:
1. **Different fire share.** Patches are centred on fires, so the centre crop holds about 3x more fire (pos_weight 30.4 against 90.0 on full patches). The model learns a different prior and is then scored on the sparser full patch.
2. **No patch edges in training.** The model never sees the outer ring of the patch (3/4 of its area), yet it has to predict there at test time.
3. **Train/test input mismatch.** The fully convolutional U-Net runs on 64x64 inputs it was never trained on (different zero-padded borders and batch-norm statistics).

The result supports training and predicting on the same full 64x64 patches. It does not show *how much* of the drop comes from losing spatial context. That would need random-crop training, which was not run.

### 3. Cost
- Training took 148.2s against 778.2s (19.0% of the baseline time), because each patch has 4x fewer pixels.

### 4. Protocol
- Both models are evaluated on the same full 64x64 eval and test patches (an earlier version also cropped eval/test, which made the scores incomparable).
- The decision threshold is chosen on eval (threshold = 0.9416) and applied unchanged to test.

---

## 4. Visualizations
The spatial crop comparison is visualized in `reports/dl/spatial_crop_ablation.png`:
- **Panel A:** Side-by-side bar chart of primary test metrics (PR AUC, ROC AUC, F1, Recall @ 1% FPR, Recall @ 5% FPR).
- **Panel B:** Training duration versus Test PR AUC trade-off plot.
