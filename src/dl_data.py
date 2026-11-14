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

from src.data import CHANNELS, PROCESSED_DIR, load_split

CHANNEL_GROUPS = {
    "weather": ["th", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "erc"],
    "vegetation": ["NDVI"],
    "terrain": ["elevation"],
    "human": ["population"],
    "fire": ["PrevFireMask"],
    "fire_neighbourhood": ["PrevFireMask"],  # alias matching features.py
}

STATS_FILE = PROCESSED_DIR / "norm_stats.npz"
SKIP = [CHANNELS.index("PrevFireMask")]  # {-1, 0, 1} mask, not normalised


def get_active_channels(drop_group=None):
    """Return list of channel names after optionally dropping one group."""
    if drop_group is None:
        return list(CHANNELS)
    if drop_group not in CHANNEL_GROUPS:
        raise ValueError(f"Unknown channel group '{drop_group}', choose from: {', '.join(CHANNEL_GROUPS)}")
    drop = set(CHANNEL_GROUPS[drop_group])
    return [c for c in CHANNELS if c not in drop]


def fit_norm(X, skip=None):
    """Per-channel clip bounds and mean/std from train X (N, C, H, W)."""
    flat = X.transpose(1, 0, 2, 3).reshape(X.shape[1], -1)
    lo, hi = np.percentile(flat, [0.5, 99.5], axis=1)
    clipped = np.clip(flat, lo[:, None], hi[:, None])
    mean, std = clipped.mean(1), clipped.std(1) + 1e-6
    if skip is None:
        skip = SKIP
    for c in skip:
        if c < X.shape[1]:
            lo[c], hi[c], mean[c], std[c] = -1, 1, 0, 1
    return lo, hi, mean, std


def get_train_stats(Xtr, force_recompute=False):
    """Return train normalisation statistics (lo, hi, mean, std) for all 12 channels.

    Caches to data/processed/norm_stats.npz to avoid recomputing 0.5-99.5 percentiles
    over 735M values on every execution. Automatically invalidates if train.npz is newer,
    or if sample count / channels do not match.
    """
    train_npz = PROCESSED_DIR / "train.npz"
    if not force_recompute and STATS_FILE.exists():
        try:
            if train_npz.exists() and STATS_FILE.stat().st_mtime < train_npz.stat().st_mtime:
                raise ValueError("Cache is older than train.npz")
            with np.load(STATS_FILE) as d:
                if len(d["lo"]) == len(CHANNELS) and int(d.get("n_samples", len(Xtr))) == len(Xtr):
                    return d["lo"], d["hi"], d["mean"], d["std"]
        except Exception:
            pass  # Corrupt or stale cache -> recompute below

    stats = fit_norm(Xtr)
    np.savez(STATS_FILE, lo=stats[0], hi=stats[1], mean=stats[2], std=stats[3],
             n_samples=len(Xtr), channels=np.array(CHANNELS))
    return stats


def apply_norm(X, stats):
    lo, hi, mean, std = (s[None, :, None, None] for s in stats)
    return ((np.clip(X, lo, hi) - mean) / std).astype(np.float32)


def apply_crop(X, y, crop):
    """Center-crop spatial dimensions of X (N, C, H, W) and y (N, H, W) to (crop, crop)."""
    if crop is None or crop == X.shape[-1]:
        return X, y
    h, w = X.shape[-2], X.shape[-1]
    if crop > h or crop > w or crop <= 0:
        raise ValueError(f"Crop size {crop} must be positive and <= spatial dimensions ({h}, {w})")
    margin_h = (h - crop) // 2
    margin_w = (w - crop) // 2
    X_cropped = np.ascontiguousarray(X[:, :, margin_h:margin_h + crop, margin_w:margin_w + crop])
    y_cropped = np.ascontiguousarray(y[:, margin_h:margin_h + crop, margin_w:margin_w + crop])
    return X_cropped, y_cropped


class FireDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loaders(batch_size=32, num_workers=0, seed=0, drop_group=None, crop=None):
    """Return train (shuffled), eval, test loaders, all normalised with train stats.

    drop_group: drop one group from CHANNEL_GROUPS ('weather', 'vegetation', 'terrain', 'fire', 'human').
    crop: optional center-crop size (e.g. 32). Default None keeps full 64x64.
    """
    active_channels = get_active_channels(drop_group)
    active_idx = [CHANNELS.index(c) for c in active_channels]

    Xtr, ytr = load_split("train")
    full_stats = get_train_stats(Xtr)
    stats = tuple(s[active_idx] for s in full_stats)

    Xtr = Xtr[:, active_idx]
    Xtr_norm = apply_norm(Xtr, stats)
    Xtr_norm, ytr = apply_crop(Xtr_norm, ytr, crop)

    g = torch.Generator().manual_seed(seed)
    out = [DataLoader(FireDataset(Xtr_norm, ytr), batch_size, shuffle=True,
                      generator=g, num_workers=num_workers)]
    for split in ("eval", "test"):
        X, y = load_split(split)
        X = X[:, active_idx]
        X_norm = apply_norm(X, stats)
        X_norm, y = apply_crop(X_norm, y, crop)
        out.append(DataLoader(FireDataset(X_norm, y), batch_size, num_workers=num_workers))
    return out


if __name__ == "__main__":
    train_dl, eval_dl, test_dl = make_loaders(batch_size=16)
    x, y = next(iter(train_dl))
    assert x.shape == (16, len(CHANNELS), 64, 64) and y.shape == (16, 64, 64)
    assert torch.isfinite(x).all() and set(y.unique().tolist()) <= {-1.0, 0.0, 1.0}
    keep = [c for c in range(len(CHANNELS)) if c not in SKIP]
    assert x[:, keep].abs().max() < 30, "normalised channels should be small"
    print(len(train_dl.dataset), len(eval_dl.dataset), len(test_dl.dataset), "dl_data default self-check ok")

    # Ablation channel-group self-checks
    expected_channels = {
        "weather": 12 - 8,
        "vegetation": 12 - 1,
        "terrain": 12 - 1,
        "human": 12 - 1,
        "fire": 12 - 1,
    }
    for grp, exp_c in expected_channels.items():
        tr, ev, te = make_loaders(batch_size=4, drop_group=grp)
        xb, yb = next(iter(tr))
        assert xb.shape == (4, exp_c, 64, 64), f"drop {grp}: expected {exp_c} ch, got {xb.shape[1]}"
        assert torch.isfinite(xb).all()
        print(f"drop_group '{grp}': {exp_c} channels ok")

    # Spatial crop self-check (32x32)
    tr_crop, _, _ = make_loaders(batch_size=4, crop=32)
    xb_crop, yb_crop = next(iter(tr_crop))
    assert xb_crop.shape == (4, 12, 32, 32) and yb_crop.shape == (4, 32, 32)
    print("crop=32 self-check ok")

    # Combined drop-group and crop self-check
    tr_comb, _, _ = make_loaders(batch_size=4, drop_group="weather", crop=32)
    xb_comb, yb_comb = next(iter(tr_comb))
    assert xb_comb.shape == (4, 4, 32, 32) and yb_comb.shape == (4, 32, 32)
    print("combined drop_group + crop self-check ok")
