"""Pixel-level tables with engineered features.

Run: python -m src.features   -> data/processed/pixels_{train,eval,test}.parquet
Use: from src.features import make_pixel_table, load_pixels, FEATURE_GROUPS

Train is negatively subsampled (all fire pixels + NEG_RATIO x no-fire, fixed seed, -1 labels dropped).
Eval/test keep every pixel in patch/row/col order, so predictions reshape straight back to (N, 64, 64).

Geometry: 1 km pixels, row index grows NORTHWARD (see ROW_TO_NORTH), `th` is the direction wind blows FROM
(degrees clockwise from north, GRIDMET `wind_from_direction`).
"""
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, uniform_filter

from src.data import CHANNELS, PROCESSED_DIR, SIZE, SPLITS, load_split

NEG_RATIO = 5
SEED = 0
CHUNK = 500
MAX_DIST = SIZE * np.sqrt(2)  # distance to fire when the patch has none

# Patches are stored south-up: the row index grows northward. Evidence (train split): with this orientation,
# next-day fire 5 km downwind of day-t fire is more frequent than 5 km upwind in every wind-speed band, and the
# ratio rises with wind speed (1.13 below 3 m/s -> 2.85 at 8+ m/s); assuming north-up reverses it in strong wind (0.37).
ROW_TO_NORTH = 1  # +1: row grows northward; -1 would mean north-up

# Physical bounds for raw channels; the export has artefacts (tmmx = 0 K, small negative pr).
# Fixed numbers, not fitted on data, so no leakage. Lower 253.15 K follows Huot et al., 330 K ~ world record high.
# th is circular, so it is wrapped with % 360 instead (clipping would turn 365 deg into 360, not 5).
CLIP = {
    "vs": (0, None), "tmmn": (253.15, 330), "tmmx": (253.15, 330),
    "sph": (0, None), "pr": (0, None), "erc": (0, None),
}

FEATURE_GROUPS = {
    "weather": [
        "th", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "erc",
        "wind_u", "wind_v", "temp_range", "vpd", "min_rel_humidity", "erc_x_wind",
    ],
    "vegetation": ["NDVI", "ndvi_n5_mean", "ndvi_n5_std", "ndvi_anomaly"],
    "terrain": ["elevation", "slope", "aspect_sin", "aspect_cos", "elev_rel"],
    # upwind_* and fire_wind_alignment also use wind, but they describe where the fire is
    "fire_neighbourhood": [
        "PrevFireMask", "prev_fire_n3", "prev_fire_n5", "prev_fire_n9", "patch_fire_frac",
        "dist_to_fire", "upwind_fire_k2", "upwind_fire_k4", "upwind_fire_k8", "fire_wind_alignment",
    ],
    "human": ["population", "pop_n9"],
}
ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
ENGINEERED = [f for f in ALL_FEATURES if f not in CHANNELS]
META = ["patch", "row", "col", "label"]
assert sorted(set(ALL_FEATURES) & set(CHANNELS)) == sorted(CHANNELS)


def _local_mean(a, k):
    return uniform_filter(a, size=(1, k, k), mode="nearest")


def feature_maps(X):
    """X (n, 12, 64, 64) -> {feature: (n, 64, 64)} for every raw and engineered feature."""
    c = {name: X[:, i] for i, name in enumerate(CHANNELS)}
    c |= {name: np.clip(c[name], lo, hi) for name, (lo, hi) in CLIP.items()}
    c["th"] = c["th"] % 360
    f = dict(c)
    n = len(X)
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    th = np.deg2rad(c["th"])
    up_n, up_e = np.cos(th), np.sin(th)  # unit vector pointing upwind (north, east)

    # fire neighbourhood (-1 = unknown treated as no fire)
    fire = (c["PrevFireMask"] == 1).astype(np.float32)
    for k in (3, 5, 9):
        f[f"prev_fire_n{k}"] = _local_mean(fire, k)
    f["patch_fire_frac"] = np.broadcast_to(fire.mean(axis=(1, 2), keepdims=True), fire.shape)
    p = np.arange(n)[:, None, None]
    for k in (2, 4, 8):  # k=8: ~40% of next-day fire is >5 km from day-t fire (EDA)
        r = np.rint(rows + ROW_TO_NORTH * k * up_n).astype(int)
        cc = np.rint(cols + k * up_e).astype(int)
        inside = (r >= 0) & (r < SIZE) & (cc >= 0) & (cc < SIZE)
        upwind = f["prev_fire_n3"][p, r.clip(0, SIZE - 1), cc.clip(0, SIZE - 1)]
        f[f"upwind_fire_k{k}"] = np.where(inside, upwind, 0)
    dist = np.full(fire.shape, MAX_DIST, np.float32)
    align = np.zeros(fire.shape, np.float32)
    for i in range(n):
        if fire[i].any():
            d, (fr, fc) = distance_transform_edt(fire[i] == 0, return_indices=True)
            to_fire_n, to_fire_e = ROW_TO_NORTH * (fr - rows), fc - cols
            dist[i] = d
            # +1: nearest fire lies upwind, so the wind pushes it toward this pixel
            align[i] = np.where(d > 0, (to_fire_n * up_n[i] + to_fire_e * up_e[i]) / np.maximum(d, 1e-6), 0)
    f["dist_to_fire"], f["fire_wind_alignment"] = dist, align

    # weather
    f["wind_u"] = -c["vs"] * np.sin(th)
    f["wind_v"] = -c["vs"] * np.cos(th)
    f["temp_range"] = c["tmmx"] - c["tmmn"]
    t = c["tmmx"] - 273.15
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))  # saturation vapour pressure, kPa (FAO-56)
    pressure = 101.3 * ((293 - 0.0065 * c["elevation"].clip(0)) / 293) ** 5.26
    e = c["sph"] * pressure / (0.622 + 0.378 * c["sph"])  # actual vapour pressure from specific humidity
    f["vpd"] = es - e
    f["min_rel_humidity"] = e / es
    f["erc_x_wind"] = c["erc"] * c["vs"]

    # vegetation
    ndvi = c["NDVI"] / 10_000  # stored x10000
    mean = _local_mean(ndvi, 5)
    f["ndvi_n5_mean"] = mean
    f["ndvi_n5_std"] = np.sqrt(np.maximum(_local_mean(ndvi**2, 5) - mean**2, 0))
    f["ndvi_anomaly"] = ndvi - mean

    # terrain
    d_row, d_col = np.gradient(c["elevation"], axis=(1, 2))  # m per 1 km pixel
    g_n, g_e = ROW_TO_NORTH * d_row / 1000, d_col / 1000
    f["slope"] = np.degrees(np.arctan(np.hypot(g_n, g_e)))
    aspect = np.arctan2(-g_e, -g_n)  # downhill direction, clockwise from north
    f["aspect_sin"], f["aspect_cos"] = np.sin(aspect), np.cos(aspect)
    f["elev_rel"] = c["elevation"] - _local_mean(c["elevation"], 9)

    # human
    f["pop_n9"] = _local_mean(c["population"], 9)
    return f


def make_pixel_table(X, y, mode="full", neg_ratio=None, seed=SEED):
    """One row per pixel: patch, row, col, label + features.

    mode="raw": the 12 input channels only; mode="full": raw + engineered.
    neg_ratio: keep all fire pixels and ~neg_ratio x as many no-fire pixels, drop -1. None keeps everything.
    """
    columns = {"raw": CHANNELS, "full": ALL_FEATURES}[mode]
    rng = np.random.default_rng(seed)
    if neg_ratio:
        keep_p = neg_ratio * (y == 1).sum() / (y == 0).sum()
    parts = []
    for start in range(0, len(X), CHUNK):
        yc = y[start:start + CHUNK]
        if neg_ratio:
            keep = (yc == 1) | ((yc == 0) & (rng.random(yc.shape) < keep_p))
        else:
            keep = np.ones(yc.shape, bool)
        maps = feature_maps(X[start:start + CHUNK])
        patch, row, col = np.nonzero(keep)  # same order as boolean indexing below
        part = {"patch": (patch + start).astype(np.int32), "row": row.astype(np.int8), "col": col.astype(np.int8)}
        part |= {name: maps[name][keep].astype(np.float32) for name in columns}
        part["label"] = yc[keep].astype(np.int8)
        parts.append(pd.DataFrame(part))
    return pd.concat(parts, ignore_index=True)


def load_pixels(split, mode="full"):
    """Read data/processed/pixels_{split}.parquet with raw or full feature columns."""
    columns = {"raw": CHANNELS, "full": ALL_FEATURES}[mode]
    return pd.read_parquet(PROCESSED_DIR / f"pixels_{split}.parquet", columns=META + columns)


def _check():
    """Fire at the centre, wind from the north: the pixel south of it is downwind."""
    X = np.zeros((1, len(CHANNELS), SIZE, SIZE), np.float32)
    X[0, CHANNELS.index("PrevFireMask"), 32, 32] = 1
    X[0, CHANNELS.index("vs")] = 5
    X[0, CHANNELS.index("sph")] = 0.005
    X[0, CHANNELS.index("tmmx")] = X[0, CHANNELS.index("tmmn")] = 300
    f = feature_maps(X)

    def south(k):  # row of the pixel k km south of the fire
        return 32 - ROW_TO_NORTH * k

    assert f["upwind_fire_k2"][0, south(2), 32] > 0 and f["upwind_fire_k2"][0, south(-2), 32] == 0
    assert np.isclose(f["fire_wind_alignment"][0, south(8), 32], 1)
    assert np.isclose(f["fire_wind_alignment"][0, south(-8), 32], -1)
    assert f["dist_to_fire"][0, 32, 36] == 4
    assert len(ENGINEERED) >= 15


if __name__ == "__main__":
    _check()
    print(f"{len(ENGINEERED)} engineered + {len(CHANNELS)} raw features")
    for split in SPLITS:
        X, y = load_split(split)
        table = make_pixel_table(X, y, "full", neg_ratio=NEG_RATIO if split == "train" else None)
        table.to_parquet(PROCESSED_DIR / f"pixels_{split}.parquet", index=False)
        labelled = table["label"] != -1
        print(f"{split:>5}: {len(table):>9} rows, fire share {(table['label'][labelled] == 1).mean():.4f}")
