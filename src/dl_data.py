"""PyTorch data for Phase 2 (DL).

Use: from src.dl_data import make_loaders
     train_dl, eval_dl, test_dl = make_loaders(batch_size=32)
     x, y = next(iter(train_dl))   # x (B, 12, 64, 64) normalised, y (B, 64, 64) in {-1, 0, 1}

Normalisation is fit on train only: clip each channel to its train 0.5-99.5 percentile
(the raw data has bad values, e.g. th = -505870), then z-score. PrevFireMask is left as is.
Label -1 (unlabelled) is kept so the loss and metrics can mask it.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import CHANNELS, load_split

SKIP = [CHANNELS.index("PrevFireMask")]  # {-1, 0, 1} mask, not normalised


def fit_norm(X):
    """Per-channel clip bounds and mean/std from train X (N, C, H, W)."""
    flat = X.transpose(1, 0, 2, 3).reshape(X.shape[1], -1)
    lo, hi = np.percentile(flat, [0.5, 99.5], axis=1)
    clipped = np.clip(flat, lo[:, None], hi[:, None])
    mean, std = clipped.mean(1), clipped.std(1) + 1e-6
    for c in SKIP:
        lo[c], hi[c], mean[c], std[c] = -1, 1, 0, 1
    return lo, hi, mean, std


def apply_norm(X, stats):
    lo, hi, mean, std = (s[None, :, None, None] for s in stats)
    return ((np.clip(X, lo, hi) - mean) / std).astype(np.float32)


class FireDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loaders(batch_size=32, num_workers=0, seed=0):
    """Return train (shuffled), eval, test loaders, all normalised with train stats."""
    Xtr, ytr = load_split("train")
    stats = fit_norm(Xtr)
    g = torch.Generator().manual_seed(seed)
    out = [DataLoader(FireDataset(apply_norm(Xtr, stats), ytr), batch_size, shuffle=True,
                      generator=g, num_workers=num_workers)]
    for split in ("eval", "test"):
        X, y = load_split(split)
        out.append(DataLoader(FireDataset(apply_norm(X, stats), y), batch_size, num_workers=num_workers))
    return out


if __name__ == "__main__":
    train_dl, eval_dl, test_dl = make_loaders(batch_size=16)
    x, y = next(iter(train_dl))
    assert x.shape == (16, len(CHANNELS), 64, 64) and y.shape == (16, 64, 64)
    assert torch.isfinite(x).all() and set(y.unique().tolist()) <= {-1.0, 0.0, 1.0}
    keep = [c for c in range(len(CHANNELS)) if c not in SKIP]
    assert x[:, keep].abs().max() < 30, "normalised channels should be small"
    print(len(train_dl.dataset), len(eval_dl.dataset), len(test_dl.dataset), "dl_data self-check ok")
