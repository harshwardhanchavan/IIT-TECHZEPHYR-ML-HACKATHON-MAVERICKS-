"""Google QuickDraw data for the optional sketch classifier (used by train.py).

QuickDraw (CC BY 4.0, https://github.com/googlecreativelab/quickdraw-dataset) provides 28x28 bitmaps.
Each category file is ~100 MB; files are cached in ./data/quickdraw.
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from pipeline.config import DATA_DIR

QD_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"
DEFAULT_ROOT = DATA_DIR / "quickdraw"
DEFAULT_CATEGORIES = ["chair", "table", "cup", "car", "airplane", "bicycle", "house", "cat", "dog", "tree",
                      "lamp", "clock", "guitar", "umbrella", "hat", "bed", "bus", "boat", "bottle", "key"]


def _progress(block, block_size, total):
    if total > 0:
        done = min(block * block_size, total)
        sys.stdout.write(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
        sys.stdout.flush()


def download_category(cat: str, root: Path = DEFAULT_ROOT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{cat.replace(' ', '_')}.npy"
    if not path.exists():
        print(f"  downloading '{cat}'")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(QD_URL.format(cat.replace(" ", "%20")), tmp, _progress)
        tmp.replace(path)
        print()
    return path


def load_category(cat: str, start: int, count: int, root: Path = DEFAULT_ROOT) -> np.ndarray:
    arr = np.load(download_category(cat, root), mmap_mode="r")[start:start + count]
    return arr.reshape(-1, 28, 28).astype(np.float32) / 255.0


def build_splits(categories, n_train=4000, n_test=500, root: Path = DEFAULT_ROOT):
    """Deterministic split per category: first n_test images -> test, next n_train -> train."""
    xtr, ytr, xte, yte = [], [], [], []
    for i, cat in enumerate(categories):
        te = load_category(cat, 0, n_test, root)
        tr = load_category(cat, n_test, n_train, root)
        xte.append(te); yte += [i] * len(te)
        xtr.append(tr); ytr += [i] * len(tr)
    return (np.concatenate(xtr)[:, None], np.array(ytr), np.concatenate(xte)[:, None], np.array(yte))


class QuickDrawDataset(Dataset):
    """Wraps arrays; optional light augmentation (random shift, random stroke thickening)."""

    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        if self.augment:
            dy, dx = np.random.randint(-2, 3, size=2)
            x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(1, 2))
            if np.random.rand() < 0.3:
                x = F.max_pool2d(x[None], 3, 1, 1)[0]
        return x, self.y[i]
