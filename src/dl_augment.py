"""Training-batch augmentation for Phase 2 (DL): random flips / 90-degree rotations of a patch.

Use: from src.dl_augment import Augment
     aug = Augment("flip_rot", *th_norm_stats(), seed=0)
     x, y = aug(x, y)            # x (B, 12, 64, 64) normalised, y (B, 64, 64); training batches only
Modes: none, flip (identity, left-right, up-down, both), rot (0/90/180/270 deg), flip_rot (all 8 symmetries).
Each patch gets one random transform, applied to all 12 channels and the label alike.

Wind direction. `th` is the direction the wind blows FROM, degrees clockwise from north. Moving the pixels
moves the wind too, so `th` is changed to match (else the model learns wind that no longer points downwind):
  left-right flip (east <-> west):   th -> -th          (360 - th)
  up-down flip   (north <-> south):  th -> 180 - th
  torch.rot90(k, dims=(-2, -1)):     th -> th + 90 * k * ROW_TO_NORTH
Flips do not depend on the grid orientation; rotation does. Patches are south-up (row index grows northward,
ROW_TO_NORTH = 1 in src/features.py), where rot90 with k = 1 sends a pixel offset (row, col) to (-col, row):
east -> south, a clockwise turn on the map, so the bearing grows by 90. The self-check below tests this
physically (fire placed downwind stays downwind) instead of restating the formula.

The loaders hold z-scored data, so th is un-normalised with the train mean/std (dl_data.fit_norm), turned,
wrapped to [0, 360) and normalised again. It is not re-clipped to the train 0.5-99.5 percentile bounds: those
remove export garbage (th = -505870), but a valid turned angle such as 5 deg can fall outside them.
"""
import numpy as np
import torch

from src.data import CHANNELS, load_split
from src.features import ROW_TO_NORTH

TH = CHANNELS.index("th")

# A transform is (k, lr): flip left-right if lr, then rotate by k * 90 deg with torch.rot90(dims=(-2, -1)).
# up-down flip = (2, True): rot180 after a left-right flip reverses only the rows.
GROUPS = {
    "none": [(0, False)],
    "flip": [(0, False), (0, True), (2, True), (2, False)],
    "rot": [(k, False) for k in range(4)],
    "flip_rot": [(k, lr) for k in range(4) for lr in (False, True)],
}


def turn_th(th_deg, k, lr):
    """Wind-from direction (degrees) after transform (k, lr); see module docstring."""
    th = -th_deg if lr else th_deg
    return torch.remainder(th + 90 * k * ROW_TO_NORTH, 360)


def transform(t, k, lr):
    """Apply (k, lr) to the last two (row, col) dims of t."""
    if lr:
        t = torch.flip(t, dims=(-1,))
    return torch.rot90(t, k, dims=(-2, -1)) if k % 4 else t


def th_norm_stats():
    """(mean, std) that dl_data.fit_norm uses to z-score th: clip to the train 0.5-99.5 percentile, then
    mean / std + 1e-6. Same formula on the th channel only, as the full fit takes minutes on an 8 GB machine."""
    th = load_split("train")[0][:, TH].ravel()
    lo, hi = np.percentile(th, [0.5, 99.5])
    th = np.clip(th, lo, hi)
    return float(th.mean()), float(th.std() + 1e-6)


class Augment:
    """Random symmetry per patch. rotate_wind=False leaves th untouched (ablation: wrong physics)."""

    def __init__(self, mode, th_mean, th_std, seed=0, rotate_wind=True):
        self.ops = GROUPS[mode]
        self.th_mean, self.th_std, self.rotate_wind = th_mean, th_std, rotate_wind
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, x, y):
        if self.ops == [(0, False)]:
            return x, y
        pick = torch.randint(len(self.ops), (len(x),), generator=self.gen).to(x.device)
        x, y = x.clone(), y.clone()
        for i, (k, lr) in enumerate(self.ops):
            idx = (pick == i).nonzero().squeeze(1)
            if (k, lr) == (0, False) or len(idx) == 0:
                continue
            xi = transform(x[idx], k, lr)
            if self.rotate_wind:
                th = xi[:, TH] * self.th_std + self.th_mean
                xi[:, TH] = (turn_th(th, k, lr) - self.th_mean) / self.th_std
            x[idx], y[idx] = xi, transform(y[idx], k, lr)
        return x, y


C = 31.5  # centre of a 64 x 64 grid; flips and rotations map it onto itself


def _downwind(th_deg):
    """Unit (row, col) step pointing downwind for wind FROM th_deg, using ROW_TO_NORTH."""
    th = np.deg2rad(th_deg)
    north, east = -np.cos(th), -np.sin(th)       # wind blows TO th + 180
    return np.array([ROW_TO_NORTH * north, east])


def _fire_dir(y):
    """Unit (row, col) step from the grid centre to the single fire pixel of y (64, 64)."""
    v = torch.nonzero(y)[0].numpy() - C
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    mean, std = 200.0, 90.0                      # made-up stats; the round trip must not depend on them
    for th0 in (0.0, 30.0, 90.0, 200.0, 315.0):
        for k, lr in GROUPS["flip_rot"]:
            x = torch.zeros(1, len(CHANNELS), 64, 64)
            x[:, TH] = (th0 - mean) / std
            x[0, 0] = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)   # position tag
            y = torch.zeros(1, 64, 64)
            r, c = np.round(C + 12 * _downwind(th0)).astype(int)
            y[0, r, c] = 1                           # next-day fire 12 km downwind of the centre
            aug = Augment("flip_rot", mean, std)
            aug.ops = [(k, lr)]                      # force this transform
            xa, ya = aug(x, y)
            th1 = float(xa[0, TH, 0, 0] * std + mean)
            assert torch.allclose(xa[0, TH], xa[0, TH, 0, 0]), "th must stay uniform over the patch"
            assert _fire_dir(ya[0]) @ _downwind(th1) > 0.99, f"fire not downwind after {(k, lr)}, th {th0}->{th1:.1f}"
            assert torch.equal(ya[0], transform(y[0], k, lr)), "label must move with the input"
            assert torch.equal(xa[0, 0], transform(x[0, 0], k, lr)), "all channels get the same transform"
            if (k, lr) == (1, False):
                assert abs(th1 - (th0 + 90 * ROW_TO_NORTH) % 360) < 1e-3, "90 deg turn must add 90 to th"
    # The test has power: with th left unturned, a 90-degree rotation puts the fire crosswind.
    aug = Augment("rot", mean, std, rotate_wind=False)
    aug.ops = [(1, False)]
    xa, ya = aug(x, y)                           # last patch from the loop above (th 315)
    assert abs(_fire_dir(ya[0]) @ _downwind(float(xa[0, TH, 0, 0] * std + mean))) < 0.1, "unturned check lacks power"
    print("dl_augment self-check ok (8 symmetries x 5 wind directions; fire stays downwind, label moves with input)")
