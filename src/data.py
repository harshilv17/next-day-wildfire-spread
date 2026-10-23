"""TFRecords -> data/processed/{train,eval,test}.npz, without TensorFlow.

Run: python -m src.data
Use: from src.data import load_split, CHANNELS
"""
import struct
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS = ("train", "eval", "test")
SIZE = 64

# Input channels in the order used by Huot et al. (2022). Label is FireMask.
CHANNELS = [
    "elevation",     # SRTM, m
    "th",            # wind direction, degrees
    "vs",            # wind speed, m/s
    "tmmn",          # min temperature, K
    "tmmx",          # max temperature, K
    "sph",           # specific humidity, kg/kg
    "pr",            # precipitation, mm
    "pdsi",          # Palmer drought severity index
    "NDVI",          # vegetation index
    "population",    # people / km^2
    "erc",           # energy release component
    "PrevFireMask",  # fire mask on day t: 1 fire, 0 no fire, -1 unknown
]
LABEL = "FireMask"  # fire mask on day t+1, same encoding


def _varint(b, i):
    result = shift = 0
    while True:
        byte = b[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, i
        shift += 7


def _fields(b):
    """Yield (field_number, payload) for a protobuf message made only of length-delimited fields."""
    i = 0
    while i < len(b):
        tag, i = _varint(b, i)
        assert tag & 7 == 2, f"unexpected wire type {tag & 7}"
        length, i = _varint(b, i)
        yield tag >> 3, b[i:i + length]
        i += length


def parse_example(record):
    """tf.train.Example bytes -> {feature name: float32 array}.

    Example{1: Features{1: map entry{1: key, 2: Feature{2: FloatList{1: packed floats}}}}}
    """
    (_, features), = _fields(record)
    out = {}
    for _, entry in _fields(features):
        entry = dict(_fields(entry))
        (kind, float_list), = _fields(entry[2])
        assert kind == 2, "expected float_list"
        (_, packed), = _fields(float_list)
        out[bytes(entry[1]).decode()] = np.frombuffer(packed, "<f4")
    return out


def read_tfrecord(path):
    """Yield raw records. Format: uint64 length, uint32 crc, data, uint32 crc."""
    # ponytail: CRCs skipped, files are read-only Kaggle downloads
    with open(path, "rb") as f:
        while header := f.read(12):
            (length,) = struct.unpack("<Q", header[:8])
            yield memoryview(f.read(length))
            f.read(4)


def build_split(split):
    files = sorted(DATA_DIR.glob(f"next_day_wildfire_spread_{split}_*.tfrecord"))
    assert files, f"no {split} TFRecords in {DATA_DIR}"
    X, y = [], []
    for path in files:
        for record in read_tfrecord(path):
            ex = parse_example(record)
            X.append(np.stack([ex[c] for c in CHANNELS]).reshape(len(CHANNELS), SIZE, SIZE))
            y.append(ex[LABEL].reshape(SIZE, SIZE))
    return np.stack(X), np.stack(y)


def load_split(split):
    """Return X (N, 12, 64, 64) float32 and y (N, 64, 64) float32 with values {-1, 0, 1}."""
    path = PROCESSED_DIR / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run `python -m src.data` first")
    with np.load(path) as d:
        return d["X"], d["y"]


def summarise(split, X, y):
    prev = X[:, CHANNELS.index("PrevFireMask")]
    labelled = y != -1
    print(
        f"{split:>5}: {len(X):>6} patches | "
        f"next fire {(y == 1).sum() / labelled.sum():.4f} of labelled px | "
        f"prev fire {(prev == 1).mean():.4f} | "
        f"next -1 share {(y == -1).mean():.4f} | prev -1 share {(prev == -1).mean():.4f} | "
        f"no-fire:fire {(y == 0).sum() / max((y == 1).sum(), 1):.0f}:1"
    )


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        X, y = build_split(split)
        np.savez(PROCESSED_DIR / f"{split}.npz", X=X, y=y)
        summarise(split, X, y)
