"""Shared evaluation helpers for Phase 1 and Phase 2 (used by notebooks/03_dl_evaluation.ipynb).

The functions are the ones in notebooks/02_evaluation.ipynb, rewritten to take their maps as arguments,
so Phase 1 and Phase 2 numbers are computed in exactly the same way. Notebook 02 is left unchanged.

Use: from src import eval_utils as eu
     maps = eu.test_maps()                       # day-t inputs, labels, distance to fire (test split)
     y, p = eu.load_preds("unet_focal")          # (1689, 64, 64) maps, Phase 1 or Phase 2 name
     thr = eu.load_results().loc["unet_focal", "threshold"]   # chosen on eval, never on test

Conventions (as in Phase 1): pixels labelled -1 are dropped from every metric, thresholds come from eval,
patches are stored south-up (row index grows northward, src/features.py ROW_TO_NORTH), `th` is the
direction the wind blows FROM, degrees clockwise from north.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import average_precision_score

from src.data import CHANNELS, SIZE, load_split
from src.features import MAX_DIST

ROOT = Path(__file__).resolve().parent.parent
PRED_DIRS = [ROOT / "reports" / "dl" / "preds", ROOT / "reports" / "preds"]  # Phase 2 first, then Phase 1
# Phase 2 metrics live in two files: runs.csv holds every Kaggle run (seeds, reruns, cnn_bce), results_dl.csv the
# reference and ablation runs. Rows for the same run are identical; the later file wins.
RESULTS = [("phase1", ROOT / "reports" / "results.csv"),
           ("phase2", ROOT / "reports" / "dl" / "checks" / "runs.csv"),
           ("phase2", ROOT / "reports" / "dl" / "results_dl.csv")]
NORM_STATS = ROOT / "reports" / "dl" / "norm_stats.json"
MODELS_DIR = ROOT / "models"

# Distance bands of Phase 1 (eval_recall_by_distance.png), in km = pixels; 0-0.5 = the pixel burns on day t.
DIST_BINS = [0, 0.5, 1.5, 2.5, 5, 10, 20, 91]
DIST_NAMES = ["0 (burning)", "1", "2", "3-5", "6-10", "11-20", "> 20"]

# Outcome codes for outcome_map / OUT_COLORS: unknown, true negative, true positive, false positive, false negative
GREY, FIRE, FP_C, FN_C = "#c8c8c8", "#e34948", "#eda100", "#2a78d6"
MASK_COLORS = [GREY, "white", FIRE]             # mask values -1, 0, 1 shifted to 0, 1, 2
OUT_COLORS = [GREY, "white", FIRE, FP_C, FN_C]  # codes 0..4


# ---------------------------------------------------------------------------------------------------------------
# Inputs
def pred_path(name):
    """Path of a model's saved test predictions, or None if the file is not on this machine."""
    for d in PRED_DIRS:
        if (d / f"{name}.npz").exists():
            return d / f"{name}.npz"
    return None


def load_preds(name):
    """Return y_true and y_prob maps (1689, 64, 64) for one model (Phase 1 or Phase 2)."""
    path = pred_path(name)
    if path is None:
        raise FileNotFoundError(f"no predictions for {name} in {', '.join(str(d) for d in PRED_DIRS)}")
    with np.load(path) as d:
        return d["y_true"], d["y_prob"]


def load_results():
    """Test metrics of every Phase 1 and Phase 2 run in one table, indexed by model, with a `phase` column."""
    tables = [pd.read_csv(path).assign(phase=phase) for phase, path in RESULTS if path.exists()]
    return pd.concat(tables).drop_duplicates("model", keep="last").set_index("model")


def dist_to_fire(prev):
    """km from every pixel to the nearest day-t fire pixel; MAX_DIST in patches without fire.

    Same computation as `dist_to_fire` in src/features.py (unknown -1 counts as no fire).
    """
    fire = prev == 1
    dist = np.full(prev.shape, MAX_DIST, np.float32)
    for i in np.flatnonzero(fire.any(axis=(1, 2))):
        dist[i] = distance_transform_edt(~fire[i])
    return dist


def test_maps():
    """Day-t maps of the test split needed for the evaluation, as a dict of (1689, 64, 64) arrays."""
    X, y = load_split("test")
    ch = {name: X[:, i] for i, name in enumerate(CHANNELS)}
    prev = ch["PrevFireMask"].copy()
    return {
        "y": y,
        "prev": prev,
        "th": ch["th"] % 360,                    # wind blows FROM this direction
        "vs": np.clip(ch["vs"], 0, None),        # wind speed, m/s
        "elev": ch["elevation"].copy(),          # static: used to find crops of the same place
        "dist": dist_to_fire(prev),
    }


# ---------------------------------------------------------------------------------------------------------------
# Thresholded outcomes and error analysis
def outcome_map(y, pred):
    """Per-pixel outcome codes (see OUT_COLORS) for label map y and boolean prediction map pred (same shape)."""
    out = np.ones(y.shape, int)                  # true negative
    out[(y == 1) & pred] = 2                     # true positive
    out[(y == 0) & pred] = 3                     # false positive
    out[(y == 1) & ~pred] = 4                    # false negative
    out[y == -1] = 0                             # unknown
    return out


def outcome_counts(y, pred):
    """TP, FP, FN pixels per patch (labels -1 never count)."""
    return {"tp": ((y == 1) & pred).sum(axis=(1, 2)),
            "fp": ((y == 0) & pred).sum(axis=(1, 2)),
            "fn": ((y == 1) & ~pred).sum(axis=(1, 2))}


def recall_by_distance(y, pred, dist):
    """Recall of next-day fire pixels in each Phase 1 distance band (pred: boolean map at the eval threshold)."""
    fire = y == 1
    band = pd.cut(dist[fire], DIST_BINS, labels=DIST_NAMES, include_lowest=True)
    g = pd.DataFrame({"band": band, "found": pred[fire]}).groupby("band", observed=False)["found"]
    return pd.DataFrame({"fire pixels": g.size(), "share of all fire": g.size() / fire.sum(), "recall": g.mean()})


def same_place(elev, a, b, k=16):
    """True if any of 9 k x k elevation blocks of patch a appears exactly (shifted) inside patch b."""
    windows = sliding_window_view(elev[b], (k, k))
    for r0 in (0, 24, 48):
        for c0 in (0, 24, 48):
            block = elev[a, r0:r0 + k, c0:c0 + k]
            if block.std() > 0 and (windows == block).all(axis=(2, 3)).any():
                return True
    return False


def distinct_fires(elev, order, k=5):
    """First k patches in `order` that show different places (so different fires)."""
    keep = []
    for i in order:
        if not any(same_place(elev, i, j) or same_place(elev, j, i) for j in keep):
            keep.append(i)
        if len(keep) == k:
            break
    return np.array(keep)


# ---------------------------------------------------------------------------------------------------------------
# Wind
def mean_wind_from(th_patch):
    """Circular mean of a patch's wind direction (degrees, FROM), in radians."""
    th = np.deg2rad(th_patch)
    return np.arctan2(np.sin(th).mean(), np.cos(th).mean())


def wind_arrow(ax, th_patch, color="black"):
    """Arrow from the patch centre toward where the wind blows; for maps drawn with origin='lower'."""
    t = mean_wind_from(th_patch)
    dx, dy = -np.sin(t), -np.cos(t)              # blows TO th + 180; x = east, y = north (south-up storage)
    ax.arrow(32 - 9 * dx, 32 - 9 * dy, 18 * dx, 18 * dy, width=1.2, color=color, length_includes_head=True)


def wind_sides(prev, th, vs, labels, probs=None, wind_bins=(0, 99)):
    """Compare pixels 1-5 px downwind vs upwind of day-t fire (notebook 02, section 4c).

    prev, th, vs, labels: (N, 64, 64) maps; probs: dict name -> (N, 64, 64) probabilities (optional).
    Only target pixels that are NOT burning on day t are used; a pixel downwind of one fire pixel and upwind of
    another is dropped from both sides. One row per (wind band, distance, side).
    """
    fire = prev == 1
    pf, rf, cf = np.nonzero(fire)
    rad = np.deg2rad(th[pf, rf, cf])
    step_row, step_col = -np.cos(rad), -np.sin(rad)  # one pixel downwind (rows grow northward)
    speed = vs[pf, rf, cf]
    rows = []
    for lo, hi in zip(wind_bins[:-1], wind_bins[1:]):
        m = (speed >= lo) & (speed < hi)
        for d in range(1, 6):
            sides = {}
            for side, sign in (("downwind", 1), ("upwind", -1)):
                r = np.rint(rf[m] + sign * d * step_row[m]).astype(int)
                c = np.rint(cf[m] + sign * d * step_col[m]).astype(int)
                inside = (r >= 0) & (r < SIZE) & (c >= 0) & (c < SIZE)
                flat = np.unique(np.ravel_multi_index((pf[m][inside], r[inside], c[inside]), labels.shape))
                sides[side] = flat[~fire.flat[flat]]
            both = np.intersect1d(sides["downwind"], sides["upwind"])
            for side, idx in sides.items():
                idx = np.setdiff1d(idx, both)
                y = labels.flat[idx]
                row = {"wind_m/s": f"{lo}-{hi}", "distance_px": d, "side": side, "pixels": len(idx),
                       "true_fire_rate": (y[y != -1] == 1).mean()}
                for name, p in (probs or {}).items():
                    row[name] = p.flat[idx].mean()
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# Fire events (notebook 02, section 6)
def group_events(elev, k=16, starts=(0, 24, 48)):
    """Event id per patch: patches sharing an exact k x k elevation block are linked, links are chained."""
    n = len(elev)
    blocks = {}
    for a in range(n):
        for r0 in starts:
            for c0 in starts:
                block = elev[a, r0:r0 + k, c0:c0 + k]
                if block.std() > 0:
                    blocks.setdefault(block.tobytes(), set()).add(a)
    links = {i: set() for i in range(n)}
    for b in range(n):
        for w in sliding_window_view(elev[b], (k, k)).reshape(-1, k, k):
            for a in blocks.get(w.tobytes(), ()):
                if a != b:
                    links[a].add(b)
                    links[b].add(a)
    event_of = np.full(n, -1)
    for start in range(n):
        if event_of[start] != -1:
            continue
        event_of[start] = start
        stack = [start]
        while stack:
            i = stack.pop()
            for j in links[i]:
                if event_of[j] == -1:
                    event_of[j] = start
                    stack.append(j)
    return event_of


def event_scores(y_map, p_map, threshold, event_of):
    """PR AUC and recall at the eval threshold of one model, per event (events need fire and no-fire pixels)."""
    size = pd.Series(event_of).value_counts()
    rows = []
    for e in size.index:
        inside = event_of == e
        y, p = y_map[inside].ravel(), p_map[inside].ravel()
        keep = y != -1
        y, p = y[keep], p[keep]
        n_fire = int((y == 1).sum())
        if n_fire == 0 or n_fire == len(y):
            continue
        rows.append({"event": e, "patches": int(size[e]), "fire_px": n_fire,
                     "pr_auc": average_precision_score(y, p), "recall": (p[y == 1] >= threshold).mean()})
    return pd.DataFrame(rows).set_index("event")


def event_summary(s, pooled_pr_auc, median_fire):
    """One row of the event-level table from event_scores() output (columns as in Phase 1)."""
    return {
        "PR AUC (pixels pooled)": pooled_pr_auc,
        "PR AUC (mean over events)": s["pr_auc"].mean(),
        "PR AUC (median over events)": s["pr_auc"].median(),
        "PR AUC (small events)": s.loc[s["fire_px"] <= median_fire, "pr_auc"].mean(),
        "PR AUC (large events)": s.loc[s["fire_px"] > median_fire, "pr_auc"].mean(),
        "Recall (mean over events)": s["recall"].mean(),
        "Events with any fire found": (s["recall"] > 0).mean(),
    }


def to_markdown(df, fmt="{:.3f}", bold_max=True):
    """Markdown table; numeric cells formatted with fmt, the best (highest) value per column in bold."""
    lines = ["| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |",
             "|---|" + "---:|" * len(df.columns)]
    best = df.select_dtypes("number").max() if bold_max else pd.Series(dtype=float)
    for name, row in df.iterrows():
        cells = []
        for c, v in row.items():
            s = fmt.format(v) if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else str(v)
            cells.append(f"**{s}**" if c in best and v == best[c] else s)
        lines.append("| " + " | ".join([str(name)] + cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Phase 2 models (need torch; imported lazily so the Phase 1 parts run without it)
def norm_stats():
    """Train-only normalisation (lo, hi, mean, std) for the 12 channels, as fitted by src.dl_data.fit_norm.

    Saved once from data/processed/train.npz to reports/dl/norm_stats.json, so the test inputs can be built
    without the 3 GB train file.
    """
    d = json.loads(NORM_STATS.read_text())
    assert d["channels"] == CHANNELS, "channel order changed since the stats were saved"
    return tuple(np.array(d[k], np.float32) for k in ("lo", "hi", "mean", "std"))


def test_inputs():
    """Normalised test inputs (1689, 12, 64, 64), exactly as make_loaders() feeds them to the network."""
    from src.dl_data import apply_norm
    X, _ = load_split("test")
    return apply_norm(X, norm_stats())


def device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def load_model(name, arch="unet", **kw):
    """Build the architecture and load models/{name}.pt; kw as for build_model (default U-Net depth 3, width 32)."""
    import torch
    from src.dl_models import build_model
    kw = kw or ({"depth": 3, "width": 32} if arch == "unet" else {"width": 32})
    model = build_model(arch, **kw)
    # weights_only: load tensors only, never run code stored in a checkpoint file
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu", weights_only=True))
    return model.to(device()).eval()


def predict(model, X, batch_size=64):
    """Sigmoid probabilities (N, 64, 64) for normalised inputs X (N, 12, 64, 64)."""
    import torch
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i:i + batch_size]).to(dev)
            out.append(torch.sigmoid(model(x)).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def pr_auc(y, p):
    keep = y != -1
    return average_precision_score(y[keep].astype(int), p[keep])


def occlusion(model, X, y, groups):
    """Test PR AUC with each channel group set to 0 (= the train mean after normalisation), at inference only.

    For PrevFireMask, which is not normalised, 0 means "no fire anywhere". Returns one row per group with the
    PR AUC and its drop from the unoccluded model.
    """
    base = pr_auc(y, predict(model, X))
    rows = [{"group": "none", "channels": "", "pr_auc": base, "drop": 0.0}]
    for g, chans in groups.items():
        Xo = X.copy()
        Xo[:, [CHANNELS.index(c) for c in chans]] = 0
        s = pr_auc(y, predict(model, Xo))
        rows.append({"group": g, "channels": ", ".join(chans), "pr_auc": s, "drop": base - s})
    return pd.DataFrame(rows).set_index("group")


def gradient_saliency(model, X, idx):
    """Gradient of the summed predicted fire probability of each patch w.r.t. its inputs.

    Returns (len(idx), 12, 64, 64): how much a small change of each input pixel/channel changes the total
    predicted fire in the patch. BatchNorm is in eval mode, so patches do not influence each other.
    """
    import torch
    dev = next(model.parameters()).device
    x = torch.from_numpy(X[idx]).to(dev).requires_grad_(True)
    torch.sigmoid(model(x)).sum().backward()
    return x.grad.detach().cpu().numpy()
