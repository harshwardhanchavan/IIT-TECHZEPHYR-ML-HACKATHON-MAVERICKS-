"""REQ 3 - Sketch understanding: contours, shape statistics, silhouette, CLIP label, optional CNN label."""
import cv2
import numpy as np
from scipy import ndimage

from . import config

_clip = None


def _get_clip(device):
    global _clip
    if _clip is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip = (CLIPProcessor.from_pretrained(config.CLIP_ID),
                 CLIPModel.from_pretrained(config.CLIP_ID).to(device).eval())
    return _clip


def clip_label(clean_img, device="cpu", topk=3):
    import torch
    proc, model = _get_clip(device)
    texts = [f"a simple line drawing of a {l}" for l in config.LABELS]
    with torch.inference_mode():
        inp = proc(text=texts, images=clean_img, return_tensors="pt", padding=True).to(device)
        probs = model(**inp).logits_per_image.softmax(dim=-1)[0].cpu().numpy()
    idx = probs.argsort()[::-1][:topk]
    return [(config.LABELS[i], round(float(probs[i]), 3)) for i in idx]


def shape_statistics(control_img):
    """Classical CV part (no models): returns (stats dict, filled silhouette or None if strokes are open)."""
    ink = (np.array(control_img.convert("L")) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(ink * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    sil = ndimage.binary_fill_holes(ink > 0)
    ys, xs = np.where(ink > 0)
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    hull_area = max(cv2.contourArea(cv2.convexHull(np.vstack(contours))), 1) if contours else 1
    flip = sil[:, ::-1]
    sym = float((sil & flip).sum() / max((sil | flip).sum(), 1))
    closed = bool(sil.sum() > 2.5 * ink.sum())
    stats = {"n_contours": len(contours),
             "n_closed_regions": int(sum(1 for a in areas if a > 400)),
             "aspect_ratio_w_over_h": round(float(bw / bh), 2),
             "stroke_density": round(float(ink.mean()), 4),
             "solidity": round(float(sil.sum() / hull_area), 3),
             "lr_symmetry": round(sym, 3),
             "silhouette_closed": closed}
    return stats, (sil if closed else None)


def understand_sketch(pre: dict, device: str = "cpu", use_clip: bool = True, use_classifier: bool = True):
    stats, sil = shape_statistics(pre["control"])
    stats["label_top3"] = clip_label(pre["clean"], device) if use_clip else []
    stats["classifier_guess"] = None
    if use_classifier:
        try:
            from .classifier import predict
            stats["classifier_guess"] = predict(pre["control"], device)
        except Exception:
            pass
    return stats, sil
