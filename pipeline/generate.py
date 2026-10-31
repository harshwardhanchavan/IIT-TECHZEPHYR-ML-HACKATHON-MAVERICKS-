"""REQ 6 - AI image refinement: SD1.5 + ControlNet-Scribble candidates, ranking, optional img2img polish."""
import gc

import cv2
import numpy as np
from scipy import ndimage

from . import config


class SketchGenerator:
    def __init__(self, device="cuda", low_vram=False):
        self.device, self.low_vram = device, low_vram
        self.pipe = None
        self._img2img = None

    def load(self):
        if self.pipe is not None:
            return
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
        dt = torch.float16 if self.device == "cuda" else torch.float32
        cn = ControlNetModel.from_pretrained(config.CN_ID, torch_dtype=dt)
        p = StableDiffusionControlNetPipeline.from_pretrained(
            config.SD_ID, controlnet=cn, torch_dtype=dt, safety_checker=None, requires_safety_checker=False)
        p.scheduler = UniPCMultistepScheduler.from_config(p.scheduler.config)
        if self.device == "cuda" and self.low_vram:
            p.enable_model_cpu_offload()
        else:
            p.to(self.device)
        p.set_progress_bar_config(disable=True)
        self.pipe = p

    def unload(self):
        self.pipe, self._img2img = None, None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def candidates(self, control, prompt, seed=42, steps=20, n=3, control_scale=1.0, guidance=7.5):
        import torch
        self.load()
        g = torch.Generator("cpu").manual_seed(int(seed))
        with torch.inference_mode():
            out = self.pipe(prompt, image=control, negative_prompt=config.NEGATIVE,
                            num_inference_steps=int(steps), guidance_scale=guidance,
                            controlnet_conditioning_scale=control_scale, generator=g,
                            num_images_per_prompt=int(n), height=512, width=512)
        return out.images

    def refine(self, image, control, prompt, seed=42, steps=25, strength=0.4):
        """Optional low-strength ControlNet img2img pass that polishes the chosen candidate."""
        import torch
        from diffusers import StableDiffusionControlNetImg2ImgPipeline
        self.load()
        if self._img2img is None:
            self._img2img = StableDiffusionControlNetImg2ImgPipeline(**self.pipe.components)  # shares weights
            self._img2img.set_progress_bar_config(disable=True)
        g = torch.Generator("cpu").manual_seed(int(seed))
        with torch.inference_mode():
            return self._img2img(prompt, image=image, control_image=control, negative_prompt=config.NEGATIVE,
                                 strength=strength, num_inference_steps=int(steps), guidance_scale=7.0,
                                 controlnet_conditioning_scale=1.0, generator=g).images[0]


def candidate_score(control, sketch_sil, gen_img, rgba_raw) -> float:
    """0.5*edge-recall (strokes preserved) + 0.5*silhouette IoU (shape preserved, if sketch is closed)."""
    ink = np.array(control.convert("L")) > 0
    edges = cv2.Canny(np.array(gen_img.convert("L")), 80, 160) > 0
    if edges.any():
        edge_recall = float((ndimage.distance_transform_edt(~edges)[ink] < 12).mean())
    else:
        edge_recall = 0.0
    if sketch_sil is None:
        return edge_recall
    mask = np.array(rgba_raw.getchannel("A")) > 10
    iou = float((mask & sketch_sil).sum() / max((mask | sketch_sil).sum(), 1))
    return 0.5 * edge_recall + 0.5 * iou
