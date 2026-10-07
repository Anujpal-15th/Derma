"""Dataset download, loading and preprocessing for DermaMNIST (HAM10000, 28x28 RGB)."""
import urllib.request
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CLASSES = [
    "actinic keratoses", "basal cell carcinoma", "benign keratosis-like lesions",
    "dermatofibroma", "melanoma", "melanocytic nevi", "vascular lesions",
]


def load_config():
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def download(cfg=None):
    cfg = cfg or load_config()
    path = ROOT / cfg["data_path"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {cfg['data_url']} -> {path}")
        urllib.request.urlretrieve(cfg["data_url"], path)
    return path


def load_splits(cfg=None):
    """Returns dict split -> (X uint8 [n,28,28,3], y int [n])."""
    d = np.load(download(cfg))
    return {s: (d[f"{s}_images"], d[f"{s}_labels"].ravel()) for s in ("train", "val", "test")}


def to_features(images):
    """Flattened pixels scaled to [0,1] plus a per-channel 16-bin colour histogram."""
    images = np.asarray(images, dtype=np.uint8)
    if images.ndim == 3:  # single image
        images = images[None]
    flat = images.reshape(len(images), -1) / 255.0
    hist = np.concatenate(
        [np.stack([np.histogram(img[..., c], bins=16, range=(0, 256))[0] for img in images]) for c in range(3)],
        axis=1,
    ) / (28 * 28)
    return np.hstack([flat, hist]).astype(np.float32)
