"""REQ 2 - Image preprocessing: any sketch -> clean, centered 512x512 scribble map."""
import cv2
import numpy as np
from PIL import Image


class EmptySketchError(ValueError):
    """Raised when the canvas / image contains (almost) no strokes."""


def _to_gray(img: Image.Image) -> np.ndarray:
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img)
    return np.array(bg.convert("L"))


def preprocess_sketch(img: Image.Image, size: int = 512, margin: float = 0.12,
                      min_ink_ratio: float = 0.002) -> dict:
    """Returns {'control': white strokes on black (ControlNet), 'clean': black strokes on white}."""
    gray = _to_gray(img)
    h, w = gray.shape
    s = 1024 / max(h, w)
    if s < 1:
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)                      # noise reduction
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)     # contrast adjustment
    if gray.mean() < 127:                                         # dark paper -> invert
        gray = 255 - gray
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if (ink > 0).mean() < min_ink_ratio:
        raise EmptySketchError("The sketch looks empty. Draw something or upload a sketch image.")
    ys, xs = np.where(ink > 0)
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh, ww = ink.shape
    side = int(max(hh, ww) * (1 + 2 * margin))
    canvas = np.zeros((side, side), np.uint8)
    y0, x0 = (side - hh) // 2, (side - ww) // 2
    canvas[y0:y0 + hh, x0:x0 + ww] = ink
    ink = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    ink = cv2.dilate(ink, np.ones((3, 3), np.uint8), iterations=1)
    ink = (ink > 60).astype(np.uint8) * 255
    return {"control": Image.fromarray(ink).convert("RGB"),
            "clean": Image.fromarray(255 - ink).convert("RGB")}
