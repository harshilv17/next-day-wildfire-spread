"""Train a Phase 2 model and write its test results to reports/dl/.

Use: python -m src.train_dl --model unet --name unet_base
     python -m src.train_dl --model unet --depth 4 --width 16 --seed 1 --name unet_d4_w16
Loss: --loss bce (default; BCE with pos_weight = train no-fire:fire ratio), focal or bce_dice, all in
src/dl_losses.py; pixels labelled -1 are masked out of every loss.
Augmentation: --aug {none,flip,rot,flip_rot} (default none), training batches only, wind direction turned with the
patch (src/dl_augment.py); --aug-fixed-wind leaves th unturned (ablation).
Early stopping on eval PR AUC. Threshold for precision/recall/F1 is chosen on eval, never on test.
Outputs: models/{name}.pt (best checkpoint), reports/dl/results_dl.csv, reports/dl/preds/{name}.npz
(Phase 1's reports/results.csv is never touched).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src import evaluate as ev
from src.dl_augment import GROUPS, Augment, th_norm_stats
from src.dl_data import FireDataset, make_loaders
from src.dl_losses import LOSSES
from src.dl_models import build_model

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "dl"
MODELS_DIR = ROOT / "models"


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, loader, dev):
    """Sigmoid probabilities as (N, 64, 64), in loader order."""
    model.eval()
    return np.concatenate([torch.sigmoid(model(x.to(dev))).squeeze(1).cpu().numpy() for x, _ in loader])


def eval_pr_auc(model, loader, dev):
    y = loader.dataset.y.numpy().ravel()
    keep = y != -1
    return average_precision_score(y[keep], predict(model, loader, dev).ravel()[keep])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unet", choices=["cnn", "unet"])
    p.add_argument("--loss", default="bce", choices=list(LOSSES), help="training loss (src/dl_losses.py)")
    p.add_argument("--depth", type=int, default=3, choices=[2, 3, 4], help="unet levels (default: 3)")
    p.add_argument("--width", "--base-filters", dest="width", type=int, default=32, choices=[16, 32, 64],
                   help="base channels (default: 32)")
    p.add_argument("--hybrid", action="store_true",
                   help="concatenate 13th channel with GBM probabilities (OOF train, final eval/test)")
    p.add_argument("--drop-group", "--drop-channel-group", dest="drop_group", default=None,
                   choices=["weather", "vegetation", "terrain", "fire", "human"],
                   help="drop one channel group for ablation")
    p.add_argument("--crop", type=int, default=None, help="center-crop spatial dimension for training (e.g. 32)")
    p.add_argument("--crop-eval", action="store_true", help="also crop eval and test sets to crop size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-frac", type=float, default=1.0, help="share of training patches used (learning curve)")
    p.add_argument("--aug", default="none", choices=list(GROUPS), help="training-batch flips/rotations")
    p.add_argument("--aug-fixed-wind", action="store_true", help="with --aug, do not turn th (ablation)")
    p.add_argument("--name", default=None, help="results row / checkpoint name (default: model)")
    a = p.parse_args()
    name = a.name or a.model

    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    dev = device()
    if a.aug != "none" and a.drop_group:
        p.error("--aug assumes all 12 channels (th at a fixed index); do not combine it with --drop-group")
    aug = Augment(a.aug, *th_norm_stats(), seed=a.seed, rotate_wind=not a.aug_fixed_wind) if a.aug != "none" else None
    train_dl, eval_dl, test_dl = make_loaders(
        a.batch_size, seed=a.seed, drop_group=a.drop_group, crop=a.crop, crop_eval=a.crop_eval, hybrid=a.hybrid
    )
    if a.train_frac < 1:  # a random share of the training patches, normalised with the full-split statistics
        ds = train_dl.dataset
        keep = torch.from_numpy(np.random.default_rng(a.seed).permutation(len(ds))[: max(1, round(a.train_frac * len(ds)))])
        train_dl = torch.utils.data.DataLoader(FireDataset(ds.X[keep].numpy(), ds.y[keep].numpy()), a.batch_size,
                                               shuffle=True, generator=torch.Generator().manual_seed(a.seed))

    y = train_dl.dataset.y
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    in_ch = train_dl.dataset.X.shape[1]
    kw = {"in_ch": in_ch, "width": a.width, **({"depth": a.depth} if a.model == "unet" else {})}
    model = build_model(a.model, **kw).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    wind = " (th fixed)" if aug and a.aug_fixed_wind else ""
    print(f"{name}: {a.model} {kw}, loss {a.loss}, aug {a.aug}{wind} on {dev}, pos_weight={pos_weight:.1f}")

    t_start = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    ckpt, best, stale, best_epoch = MODELS_DIR / f"{name}.pt", -1.0, 0, 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, yb in train_dl:
            x, yb = x.to(dev), yb.to(dev)
            if aug:
                x, yb = aug(x, yb)
            opt.zero_grad()
            loss = LOSSES[a.loss](model(x), yb, pos_weight)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        score = eval_pr_auc(model, eval_dl, dev)
        print(f"epoch {epoch:>2}  train loss {total / len(train_dl.dataset):.4f}  eval PR AUC {score:.4f}")
        if score > best:
            best, stale, best_epoch = score, 0, epoch
            torch.save(model.state_dict(), ckpt)
        else:
            stale += 1
            if stale >= a.patience:
                print(f"early stop, best eval PR AUC {best:.4f} at epoch {best_epoch}")
                break

    model.load_state_dict(torch.load(ckpt))
    p_eval, p_test = predict(model, eval_dl, dev), predict(model, test_dl, dev)
    eval_m = ev.metrics(eval_dl.dataset.y.numpy(), p_eval)
    threshold = eval_m["best_f1_threshold"]
    eval_m_at_thresh = ev.metrics(eval_dl.dataset.y.numpy(), p_eval, threshold=threshold)
    ev.RESULTS_CSV, ev.PREDS_DIR = OUT_DIR / "results_dl.csv", OUT_DIR / "preds"  # keep Phase 1 outputs untouched
    test_row = ev.evaluate(test_dl.dataset.y.numpy(), p_test, name, threshold=threshold)

    runtime_sec = time.time() - t_start
    n_params = sum(p.numel() for p in model.parameters())
    run_meta = {
        "name": name,
        "model": a.model,
        "depth": a.depth if a.model == "unet" else None,
        "width": a.width,
        "hybrid": a.hybrid,
        "crop": a.crop,
        "drop_group": a.drop_group,
        "seed": a.seed,
        "in_ch": in_ch,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_eval_pr_auc": float(best),
        "runtime_sec": round(runtime_sec, 2),
        "eval_metrics": {
            "pr_auc": float(eval_m["pr_auc"]),
            "roc_auc": float(eval_m["roc_auc"]),
            "precision": float(eval_m_at_thresh["precision"]) if eval_m_at_thresh["precision"] is not None else None,
            "recall": float(eval_m_at_thresh["recall"]) if eval_m_at_thresh["recall"] is not None else None,
            "f1": float(eval_m_at_thresh["f1"]) if eval_m_at_thresh["f1"] is not None else None,
            "threshold": float(threshold),
            "best_f1": float(eval_m["best_f1"]),
            "best_f1_threshold": float(eval_m["best_f1_threshold"]),
            "recall_at_1fpr": float(eval_m["recall_at_1fpr"]),
            "recall_at_5fpr": float(eval_m["recall_at_5fpr"]),
        },
        "test_metrics": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in test_row.items()}
    }
    with open(MODELS_DIR / f"{name}_run.json", "w") as f:
        json.dump(run_meta, f, indent=2)


if __name__ == "__main__":
    main()
