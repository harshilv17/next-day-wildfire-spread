"""Train a Phase 2 model and write its test results to reports/dl/.

Use: python -m src.train_dl --model unet --name unet_base
     python -m src.train_dl --model unet --depth 4 --width 16 --seed 1 --name unet_d4_w16
Loss: BCE with pos_weight = train no-fire:fire ratio, pixels labelled -1 masked out.
Early stopping on eval PR AUC. Threshold for precision/recall/F1 is chosen on eval, never on test.
Outputs: models/{name}.pt (best checkpoint), reports/dl/results_dl.csv, reports/dl/preds/{name}.npz
(Phase 1's reports/results.csv is never touched).
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src import evaluate as ev
from src.dl_data import make_loaders
from src.dl_models import build_model

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "dl"
MODELS_DIR = ROOT / "models"


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def masked_bce(logits, y, pos_weight):
    """BCE over labelled pixels only (y != -1); fire pixels weighted by pos_weight."""
    keep = y != -1
    return F.binary_cross_entropy_with_logits(
        logits.squeeze(1)[keep], y[keep], pos_weight=torch.tensor(pos_weight, device=y.device)
    )


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
    p.add_argument("--depth", type=int, default=3, help="unet levels")
    p.add_argument("--width", type=int, default=32, help="base channels")
    p.add_argument("--drop-group", "--drop-channel-group", dest="drop_group", default=None,
                   choices=["weather", "vegetation", "terrain", "fire", "human"],
                   help="drop one channel group for ablation")
    p.add_argument("--crop", type=int, default=None, help="center-crop spatial dimension (e.g. 32)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--name", default=None, help="results row / checkpoint name (default: model)")
    a = p.parse_args()
    name = a.name or a.model

    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    dev = device()
    train_dl, eval_dl, test_dl = make_loaders(a.batch_size, seed=a.seed, drop_group=a.drop_group, crop=a.crop)

    y = train_dl.dataset.y
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    in_ch = train_dl.dataset.X.shape[1]
    kw = {"in_ch": in_ch, "width": a.width, **({"depth": a.depth} if a.model == "unet" else {})}
    model = build_model(a.model, **kw).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    print(f"{name}: {a.model} {kw} on {dev}, pos_weight={pos_weight:.1f}")

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt, best, stale = MODELS_DIR / f"{name}.pt", -1.0, 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, yb in train_dl:
            x, yb = x.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = masked_bce(model(x), yb, pos_weight)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        score = eval_pr_auc(model, eval_dl, dev)
        print(f"epoch {epoch:>2}  train loss {total / len(train_dl.dataset):.4f}  eval PR AUC {score:.4f}")
        if score > best:
            best, stale = score, 0
            torch.save(model.state_dict(), ckpt)
        else:
            stale += 1
            if stale >= a.patience:
                print(f"early stop, best eval PR AUC {best:.4f}")
                break

    model.load_state_dict(torch.load(ckpt))
    p_eval, p_test = predict(model, eval_dl, dev), predict(model, test_dl, dev)
    threshold = ev.metrics(eval_dl.dataset.y.numpy(), p_eval)["best_f1_threshold"]
    ev.RESULTS_CSV, ev.PREDS_DIR = OUT_DIR / "results_dl.csv", OUT_DIR / "preds"  # keep Phase 1 outputs untouched
    ev.evaluate(test_dl.dataset.y.numpy(), p_test, name, threshold=threshold)


if __name__ == "__main__":
    main()
