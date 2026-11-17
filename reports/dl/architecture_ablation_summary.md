# U-Net Architecture Ablation Report (Phase 2, Task 1b)

## 1. Executive Summary

This study systematically investigates the impact of architectural depth and base width on next-day wildfire spread prediction. Using the canonical U-Net baseline (`unet_base`: depth 3, base width 32 filters, 469,761 parameters) as the reference, we evaluated:
- **Depth Ablation:** Depths 2, 3, and 4 (fixed base width = 32)
- **Width Ablation:** Base widths 16, 32, and 64 (fixed depth = 3)

All experiments were executed on Apple Silicon MPS with strictly identical training configurations (seed 0, batch size 32, Adam optimizer, initial lr 1e-3, masked BCE loss with positive class weighting, and early stopping on evaluation split PR AUC).

---

## 2. Quantitative Results Matrix

| Experiment | Model | Depth | Width | Parameters | Best Epoch | Duration (s) | Eval PR AUC | Test PR AUC | $\\Delta$ vs Base | Test ROC AUC | Test F1 | Test Rec@1%FPR | Test Rec@5%FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Depth 2** | `unet_d2_w32` | 2 | 32 | 104,385 | 7 | 446.6 | 0.2016 | 0.2585 | -0.0103 | 0.9096 | 0.3390 | 0.3728 | 0.6771 |
| **Baseline** | `unet_base` | 3 | 32 | 469,761 | 7 | 778.2 | 0.2188 | 0.2688 | Reference | 0.9289 | 0.3531 | 0.3926 | 0.7027 |
| **Depth 4** | `unet_d4_w32` | 4 | 32 | 1,929,601 | 11 | 791.9 | 0.3008 | 0.3371 | +0.0683 | 0.9338 | 0.3909 | 0.4459 | 0.7845 |
| **Width 16** | `unet_d3_w16` | 3 | 16 | 118,657 | 7 | 278.0 | 0.2158 | 0.2627 | -0.0062 | 0.9248 | 0.3402 | 0.3865 | 0.6993 |
| **Width 64** | `unet_d3_w64` | 3 | 64 | 1,869,313 | 7 | 1126.7 | 0.2251 | 0.2718 | +0.0030 | 0.9250 | 0.3479 | 0.3888 | 0.6984 |

---

## 3. Analysis & Key Insights

### 1. Does increasing depth improve performance?
- Comparing **Depth 2 (104k params)** $\to$ **Depth 3 (470k params)** $\to$ **Depth 4 (1.93M params)** at Base Width 32:
  - Test PR AUC: Depth 2 (0.2585), Depth 3 (0.2688), Depth 4 (0.3371).
  - Receptive field of one output pixel (measured from input gradients): **18×18 px** at depth 2, **44×44 px** at depth 3, **92×92 px** at depth 4. Only depth 4 lets every output pixel see the whole 64×64 km patch.
  - **Depth 2 (-0.0103 vs base):** within run-to-run noise. Its 18 km reach is about that of the Phase 1 neighbourhood features.
  - **Depth 4 (+0.0683 vs base):** Eval PR AUC 0.2188 $\to$ 0.3008 (+0.0820), test PR AUC 0.2688 $\to$ 0.3371 (+0.0683), far outside run-to-run noise ($\le 0.02$). Confirmed with 3 seeds and focal loss (Section 5).

### 2. Does increasing width improve performance?
- Comparing **Width 16 (119k params)** $\to$ **Width 32 (470k params)** $\to$ **Width 64 (1.87M params)** at Depth 3:
  - Test PR AUC: Width 16 (0.2627), Width 32 (0.2688), Width 64 (0.2718).
  - Widening channels from 16 to 32 yields +0.0061, and from 32 to 64 yields only +0.0030.
  - Crucially, `width_64` has almost the exact same parameter count as `depth_4` (1.87M vs 1.93M params), yet achieves only 0.2718 vs 0.3371 (+0.0030 vs +0.0683).
  - This demonstrates that **depth (receptive field / spatial hierarchy) is far more impactful than channel width (capacity per scale)** for next-day wildfire spread modeling.

### 3. Do larger models provide meaningful gains relative to computational cost?
- **Width scaling does not:** Scaling width to 64 quadrupled parameters (470k $\to$ 1.87M) and doubled training duration (778s $\to$ 1127s) for a negligible gain (+0.0030) entirely within run-to-run noise.
- **Depth scaling does:** Scaling depth to 4 increased parameters to 1.93M, but yielded a decisive improvement in PR AUC (+0.0683), Best-F1 (0.3531 $\to$ 0.3909), and Recall at 1% FPR (0.3926 $\to$ 0.4459), with training duration remaining manageable (792s vs 778s).

### 4. Best Performance / Efficiency Trade-Off
- **`unet_base` (Depth 3, Width 32, 469,761 parameters):** Serves as the robust, lightweight baseline providing solid convergence, fast training (~13 min), and low memory overhead (< 500 MB).
- **`unet_d4_w32` (Depth 4, Width 32, 1,929,601 parameters):** Demonstrates the strongest predictive performance overall, providing the deep receptive field required for regional spread dynamics while keeping channel width compact at 32.

### 5. Seeds and confirmation
- The ablation runs above use one seed (0) and weighted BCE. The width changes (at most 0.006) and the depth-2 drop (0.010) are within run-to-run noise ($\le 0.02$, `reports/dl/checks/reruns.csv`).
- **Depth 4 is confirmed.** It was retrained with focal loss on seeds 0, 1 and 2 (`unet_final_s0..s2`, the final Phase 2 model): test PR AUC **0.344 ± 0.007**, against **0.280 ± 0.002** for depth 3 with the same loss and no augmentation (`unet_focal_aug_none_s0..s2`). The gap is +0.064, and every depth-4 seed lands between 0.338 and 0.351.
- Recall at 1% FPR goes from 0.399 ± 0.002 to 0.459 ± 0.003, while ROC AUC is unchanged (0.934 vs 0.935): the gain is in the rare-fire, low-false-alarm region.
- Caveat: the depth-4 and depth-3 seed runs were trained on different Apple-GPU machines. Hardware alone moves test PR AUC by up to 0.02, a third of the gap.

---

## 4. Visualizations
The ablation comparisons are visualized in `reports/dl/architecture_ablation.png`:
- **Panel A:** Test PR AUC and ROC AUC across U-Net depths (2, 3, 4) at base width 32.
- **Panel B:** Test PR AUC and ROC AUC across U-Net base widths (16, 32, 64) at depth 3.
- **Panel C:** Test PR AUC versus total parameter count (log scale) showing the efficiency frontier.
