"""REQ 7 - 3D reconstruction: TripoSR (primary) and a no-AI silhouette-inflation fallback."""
import gc
import sys
import types

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy import ndimage

from . import config


def install_mcubes_stub():
    """TripoSR imports `torchmcubes`, a CUDA extension that is painful to compile (esp. on Windows).
    We register a scikit-image implementation under the same name instead (CPU, ~1-3 s)."""
    if getattr(sys.modules.get("torchmcubes"), "_is_stub", False):
        return
    from skimage import measure

    def marching_cubes(vol, thresh):
        v = vol.detach().float().cpu().numpy()
        if not (v.min() < thresh < v.max()):
            raise RuntimeError("TripoSR found no object surface. Try another seed or a cleaner sketch.")
        verts, faces, _, _ = measure.marching_cubes(v, level=float(thresh))
        return torch.from_numpy(verts.copy()).float(), torch.from_numpy(faces.astype(np.int64))

    mod = types.ModuleType("torchmcubes")
    mod.marching_cubes = marching_cubes
    mod._is_stub = True
    sys.modules["torchmcubes"] = mod


class Reconstructor:
    def __init__(self, device="cuda", resolution=256):
        self.device, self.resolution = device, resolution
        self.model = None

    def load(self):
        if self.model is not None:
            return
        if not config.TRIPOSR_DIR.is_dir():
            raise RuntimeError(f"TripoSR source not found at {config.TRIPOSR_DIR}. Run `run.bat setup` "
                               "(or: git clone https://github.com/VAST-AI-Research/TripoSR models/TripoSR).")
        install_mcubes_stub()
        sys.path.insert(0, str(config.TRIPOSR_DIR))
        from tsr.system import TSR
        m = TSR.from_pretrained(config.TRIPOSR_ID, config_name="config.yaml", weight_name="model.ckpt")
        m.renderer.set_chunk_size(8192)
        m.to(self.device)
        self.model = m

    def unload(self):
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(self, model_input: Image.Image, resolution=None) -> trimesh.Trimesh:
        self.load()
        with torch.inference_mode():
            codes = self.model([model_input], device=self.device)
            return self.model.extract_mesh(codes, True, resolution=resolution or self.resolution)[0]


def inflate_fallback(rgba: Image.Image, grid: int = 128, depth: int = 64) -> trimesh.Trimesh:
    """No-AI emergency fallback: rounded volume from the silhouette, coloured from the image. Low fidelity."""
    from skimage import measure
    arr = np.asarray(rgba.convert("RGBA").resize((grid, grid), Image.LANCZOS))
    mask = arr[..., 3] > 40
    if mask.sum() < 20:
        raise ValueError("Silhouette too small for fallback reconstruction.")
    dist = ndimage.distance_transform_edt(mask)
    half = np.sqrt(dist / dist.max()) * (depth / 2 - 2)
    z = np.arange(depth) - (depth - 1) / 2
    vol = (np.abs(z)[None, None, :] <= half[..., None]) & mask[..., None]
    vol = ndimage.gaussian_filter(vol.astype(np.float32), 1.0)
    verts, faces, _, _ = measure.marching_cubes(np.pad(vol, 1), level=0.5)
    verts -= 1
    px = np.clip(verts[:, 1].round().astype(int), 0, grid - 1)
    py = np.clip(verts[:, 0].round().astype(int), 0, grid - 1)
    v = np.stack([verts[:, 1], grid - verts[:, 0], verts[:, 2]], axis=1)
    v -= v.mean(axis=0)
    v /= np.abs(v).max() * 2
    return trimesh.Trimesh(v, faces, vertex_colors=arr[py, px][:, :3], process=True)


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Drop degenerate faces / floating debris, fix normals, center and scale to a unit box."""
    mesh = mesh.copy()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    try:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            mesh = max(parts, key=lambda m: len(m.faces))
    except Exception:
        pass
    try:
        mesh.fix_normals()
    except Exception:
        pass
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(1.0 / max(mesh.extents))
    return mesh
