# Augmentation ablation: focal U-Net with flips and 90-degree rotations

Only `--aug` changes between the runs (`src/dl_augment.py`). The data, architecture (3 levels, width 32), focal loss, optimiser, batch size, early stopping and seeds (0, 1, 2) are the same as in the loss sweep. Augmentation is applied to training batches only, with the same random transform for all 12 channels and the label of a patch, and the wind direction `th` is turned with the patch. `unet_focal_aug_nowind` is `flip_rot` with `th` deliberately left unturned.

All 13 runs used the same machine (Apple M3 GPU, 8 GB). The loss-sweep runs used a Kaggle T4, and a same-seed rerun alone moves test PR AUC by up to 0.02 (`reruns.csv`), so the no-augmentation baseline was retrained here instead of being taken from `seeds.csv`. On the M3 it scores 0.2803 ± 0.0015 against 0.2752 ± 0.0079 on the T4.

## Mean ± standard deviation over seeds 0, 1, 2

| `--aug` | Seeds | Eval PR AUC | Test PR AUC | Recall @ 1% FPR | Recall @ 5% FPR |
|---|---:|---:|---:|---:|---:|
| **none** | 3 | 0.2219 ± 0.0034 | 0.2803 ± 0.0015 | 0.3993 ± 0.0019 | 0.7238 ± 0.0078 |
| flip | 3 | 0.2240 ± 0.0049 | 0.2828 ± 0.0042 | 0.3975 ± 0.0046 | 0.7278 ± 0.0101 |
| rot | 3 | 0.2261 ± 0.0038 | 0.2738 ± 0.0054 | 0.3912 ± 0.0103 | 0.7179 ± 0.0018 |
| flip_rot | 3 | 0.2205 ± 0.0032 | 0.2735 ± 0.0014 | 0.3949 ± 0.0024 | 0.7127 ± 0.0107 |
| flip_rot, `th` not turned | 1 (seed 0) | 0.2218 | 0.2726 | 0.3950 | 0.7301 |

Eval PR AUC is the best epoch's. Per-run numbers: `reports/dl/results_dl.csv` and `logs/`.

## Paired over the same fire events (test PR AUC, 1,000 resamples)

Each augmented run against the no-augmentation run with the same seed (`aug_ablation_diffs.csv`):

| `--aug` − none | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| flip | +0.005 (−0.007 to 0.013) | −0.001 (−0.011 to 0.010) | +0.003 (−0.008 to 0.012) |
| rot | −0.012 (−0.024 to 0.002) | −0.005 (−0.020 to 0.005) | −0.002 (−0.020 to 0.010) |
| flip_rot | −0.007 (−0.019 to 0.001) | −0.007 (−0.018 to 0.004) | −0.007 (−0.022 to 0.003) |

Turning the wind (seed 0): `flip_rot` − `flip_rot` with `th` fixed = +0.001 (−0.013 to 0.013).

## Comment

- **No augmentation reliably helps.** Every paired 95% interval contains zero. The largest mean change on eval is +0.004 (`rot`) and on test +0.0025 (`flip`), both below the seed and rerun noise.
- **Rotations tend to hurt on test.** `rot` and `flip_rot` are below `none` for all three seeds (mean −0.006 and −0.007), though no single difference is significant. Flips alone are neutral.
- **The pre-set rule and its outcome.** The rule picks the mode with the highest mean eval PR AUC if it beats `none` by more than one seed standard deviation. Taken literally it picks `rot`: +0.0042 against a threshold of 0.0038. That margin of 0.0004 is far inside the noise (a three-seed standard deviation is itself very uncertain), and test does not confirm it. We therefore keep the default.
- **Turning the wind made no measurable difference** in the one run that tests it (+0.001, interval −0.013 to 0.013). The model seems to rely little on `th` in this setting, so the check cannot show whether wrong wind physics hurts. The transform is still correct by construction (self-check in `src/dl_augment.py`).
- **Possible reasons rotations do not help:** the data is not rotation-invariant. Fires in the dataset have prevailing wind regimes, and terrain aspect and the south-up grid make north and south physically different, so rotated patches show combinations the test set rarely contains. With about 15,000 training patches, the U-Net is also not short of data (the learning curve is flat beyond half the training set).
- **Cost:** augmentation slows convergence a little (best epoch 12.3 on average for `flip_rot` against 9.3 for `none`).
- **Decision: the final U-Net uses `--aug none`** (the default). `--aug flip` is a safe alternative but brings no measurable gain.

Figure: `reports/dl/figures/aug_ablation.png`. Data: `aug_ablation.csv` and `aug_ablation_diffs.csv`, with the training logs in `logs/`. To recompute: `python -m src.dl_aug_report --boot`.
