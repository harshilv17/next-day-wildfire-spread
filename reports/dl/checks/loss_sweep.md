# Loss sweep: U-Net with weighted BCE, focal loss and BCE + Dice

Only `--loss` changes between the runs. The data, architecture (3 levels, width 32), optimiser, batch size, early stopping and seeds are the same, and every run used one NVIDIA T4 GPU. The loss is chosen on eval PR AUC; test results are reported but not used to choose.

## The three runs (seed 0, the rows in `reports/dl/results_dl.csv`)

| Run | Eval PR AUC | Test PR AUC | ROC AUC | F1 | Recall @ 1% FPR | Recall @ 5% FPR |
|---|---:|---:|---:|---:|---:|---:|
| `unet_bce` | 0.2212 | 0.2579 | 0.9220 | 0.3539 | 0.3842 | 0.6997 |
| `unet_focal` | 0.2225 | 0.2694 | 0.9255 | 0.3543 | 0.3965 | 0.7096 |
| `unet_bce_dice` | 0.2245 | 0.2694 | 0.9241 | 0.3548 | 0.3945 | 0.7043 |

## Over three seeds (0, 1, 2)

| Loss | Eval PR AUC | Test PR AUC |
|---|---:|---:|
| bce | 0.2215 ± 0.0032 | 0.2685 ± 0.0101 |
| **focal** | **0.2256 ± 0.0028** | **0.2752 ± 0.0079** |
| bce_dice | 0.2212 ± 0.0036 | 0.2688 ± 0.0056 |

Paired over the same fire events (seed-0 runs, 1,000 resamples):
- focal − bce: +0.012 (95% interval 0.002 to 0.018);
- bce_dice − bce: +0.012 (0.004 to 0.018);
- focal − bce_dice: 0.000 (−0.008 to 0.007).

## Comment

- **Focal loss wins** under the rule fixed before the runs: it has the highest mean eval PR AUC, ahead of BCE by more than one seed standard deviation. It beats BCE on eval and on test for each of the three seeds.
- **The gain is small:** about +0.004 on eval and +0.007 on test. A same-seed rerun on the same GPU moves a single test score by up to 0.02, so with three seeds the gain is consistent but not significant.
- **One seed alone would have misled us.** The seed-0 BCE run is weak: its same-seed rerun scores 0.2706. That makes both other losses look clearly better on seed 0, even though BCE + Dice is no better than BCE over three seeds.
- **Focal loss changes the scores.** Its F1 threshold chosen on eval is about 0.7 instead of 0.93. Its probabilities also no longer match the closed-form correction for class weighting, so they need recalibration on eval before being read as risk (see the technical validation).
- **Decision:** the final U-Net uses `--loss focal`.

Figure: `reports/dl/figures/loss_sweep.png`. Data: `loss_sweep.csv`, `loss_sweep_ci_events.csv`, `loss_sweep_diffs.csv` and `runs.csv`, with the training logs in `logs/`. To recompute: `python reports/dl/checks/dl_checks.py sweep`.
