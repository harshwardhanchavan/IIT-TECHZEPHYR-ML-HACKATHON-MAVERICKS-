"""Train the optional QuickDraw sketch classifier -> models/sketch_cnn.pt

    python train.py                 # 20 categories, 4000 train / 500 test per class, 8 epochs
    python train.py --quick         # smoke test: 5 categories, 500 per class, 2 epochs

The main pipeline (SD + ControlNet + TripoSR + CLIP) uses pretrained weights; this classifier is the only model
trained in this repository. It is used as a second opinion when CLIP is unsure what a sketch shows.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DEFAULT_CATEGORIES, QuickDrawDataset, build_splits
from pipeline.classifier import SketchCNN
from pipeline.config import CLASSIFIER_PATH


def evaluate(model, loader, device, n_classes):
    model.eval()
    correct = np.zeros(n_classes); total = np.zeros(n_classes)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            for t, p in zip(y.numpy(), pred.numpy()):
                total[t] += 1; correct[t] += int(t == p)
    return correct.sum() / total.sum(), correct / np.maximum(total, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=4000, help="training images per category")
    ap.add_argument("--n-test", type=int, default=500, help="test images per category")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    ap.add_argument("--out", default=str(CLASSIFIER_PATH))
    a = ap.parse_args()
    if a.quick:
        a.categories, a.n_train, a.n_test, a.epochs = a.categories[:5], 500, 100, 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} categories={len(a.categories)} train/cat={a.n_train} test/cat={a.n_test}")
    xtr, ytr, xte, yte = build_splits(a.categories, a.n_train, a.n_test)
    print("train", xtr.shape, "test", xte.shape)
    tr = DataLoader(QuickDrawDataset(xtr, ytr, augment=True), batch_size=a.batch_size, shuffle=True, num_workers=0)
    te = DataLoader(QuickDrawDataset(xte, yte), batch_size=512, num_workers=0)

    model = SketchCNN(len(a.categories)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tr))
    best, t0 = 0.0, time.time()
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(x)
        acc, per_class = evaluate(model, te, device, len(a.categories))
        print(f"epoch {ep + 1}/{a.epochs}  loss {tot / len(tr.dataset):.3f}  test acc {acc:.3f}  ({time.time() - t0:.0f}s)")
        if acc >= best:
            best = acc
            torch.save({"state_dict": model.state_dict(), "categories": a.categories}, a.out)
    metrics = {"best_test_acc": round(float(best), 4), "epochs": a.epochs, "categories": a.categories,
               "per_class_acc": {c: round(float(v), 3) for c, v in zip(a.categories, per_class)},
               "train_seconds": round(time.time() - t0, 1), "device": device}
    out_json = str(a.out).replace(".pt", "_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {a.out} and {out_json} (best test acc {best:.3f})")


if __name__ == "__main__":
    main()
