"""REQ 4 - Background removal (rembg / U2-Net) and foreground preparation for image-to-3D."""
import numpy as np
from PIL import Image


class BackgroundRemover:
    def __init__(self, model_name: str = "u2net"):
        self.model_name = model_name
        self._session = None

    def remove(self, img: Image.Image) -> Image.Image:
        from rembg import new_session, remove
        if self._session is None:
            self._session = new_session(self.model_name)
        return remove(img, session=self._session)       # RGBA, same size as input


def prepare_foreground(rgba: Image.Image, size: int = 512, ratio: float = 0.85) -> Image.Image:
    """Crop to the object, scale to `ratio` of the frame, center on a transparent canvas."""
    rgba = rgba.convert("RGBA")
    bbox = rgba.getchannel("A").point(lambda a: 255 if a > 10 else 0).getbbox()
    if bbox is None:
        raise ValueError("Background removal produced an empty image.")
    obj = rgba.crop(bbox)
    k = size * ratio / max(obj.size)
    obj = obj.resize((max(1, int(obj.width * k)), max(1, int(obj.height * k))), Image.LANCZOS)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(obj, ((size - obj.width) // 2, (size - obj.height) // 2), obj)
    return out


def rgba_to_model_input(rgba: Image.Image) -> Image.Image:
    """Composite on neutral grey (what TripoSR expects)."""
    a = np.asarray(rgba).astype(np.float32) / 255.0
    rgb = a[..., :3] * a[..., 3:4] + (1 - a[..., 3:4]) * 0.5
    return Image.fromarray((rgb * 255).astype(np.uint8))
