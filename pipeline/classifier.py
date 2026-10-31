"""Small CNN trained by train.py on Google QuickDraw. Used as a second opinion for sketch understanding."""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CLASSIFIER_PATH


class SketchCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        return self.fc2(self.drop(F.relu(self.fc1(x.flatten(1)))))


def control_to_28(control_img) -> np.ndarray:
    """White-on-black scribble map -> 28x28 float array formatted like QuickDraw bitmaps."""
    a = np.array(control_img.convert("L"))
    ys, xs = np.where(a > 0)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = max(a.shape)
    pad = np.zeros((s, s), np.uint8)
    oy, ox = (s - a.shape[0]) // 2, (s - a.shape[1]) // 2
    pad[oy:oy + a.shape[0], ox:ox + a.shape[1]] = a
    x = cv2.resize(pad, (24, 24), interpolation=cv2.INTER_AREA)
    return np.pad(x, 2).astype(np.float32) / 255.0


_cache = {}


def load_classifier(path=CLASSIFIER_PATH, device="cpu"):
    """Returns (model, categories) or None if no trained checkpoint exists."""
    key = (str(path), device)
    if key in _cache:
        return _cache[key]
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device)
    model = SketchCNN(len(ckpt["categories"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _cache[key] = (model, ckpt["categories"])
    return _cache[key]


@torch.inference_mode()
def predict(control_img, device="cpu"):
    """Returns (label, probability) or None when no checkpoint is available."""
    loaded = load_classifier(device=device)
    if loaded is None:
        return None
    model, cats = loaded
    x = torch.tensor(control_to_28(control_img))[None, None].to(device)
    p = F.softmax(model(x), dim=1)[0].cpu().numpy()
    i = int(p.argmax())
    return cats[i], round(float(p[i]), 3)
