"""End-to-end orchestration of all ten stages with timing + hardware report."""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .background import BackgroundRemover, prepare_foreground, rgba_to_model_input
from .export import export_mesh
from .generate import SketchGenerator, candidate_score
from .preprocess import preprocess_sketch
from .preview import save_preview_files
from .prompt import build_prompt
from .reconstruct import Reconstructor, clean_mesh, inflate_fallback
from .texture import bake_uv_texture
from .understand import understand_sketch, _get_clip


class Sketch2ShapePipeline:
    def __init__(self, low_vram=None, out_root=None):
        self.hw = config.hardware_info()
        self.device = self.hw["device"]
        self.low_vram = low_vram if low_vram is not None else (self.device == "cuda" and self.hw["vram_gb"] < 12)
        self.out_root = Path(out_root or config.OUTPUT_DIR)
        self.gen = SketchGenerator(self.device, self.low_vram)
        self.bg = BackgroundRemover()
        self.rec = Reconstructor(self.device)

    def warmup(self):
        """Download and load every model once (use before a demo)."""
        _get_clip(self.device)
        self.gen.load()
        self.bg.remove(Image.new("RGB", (64, 64), "white"))
        self.rec.load()

    def run(self, sketch, prompt="", style="none", seed=42, steps=20, n_candidates=3, refine=False,
            texture=True, use_fallback=False, progress=None):
        log = progress or (lambda f, d="": None)
        if self.device != "cuda" and not use_fallback:
            raise RuntimeError("No NVIDIA GPU detected. Install the CUDA build of PyTorch, use Google Colab, "
                               "or enable fallback mode (low fidelity).")
        run_dir = self.out_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True)
        T, t0 = {}, time.perf_counter()

        def tick(name, s):
            T[name] = round(time.perf_counter() - s, 2)

        log(0.05, "Preprocessing sketch"); s = time.perf_counter()                       # REQ 2
        pre = preprocess_sketch(sketch)
        pre["clean"].save(run_dir / "1_sketch_clean.png")
        pre["control"].save(run_dir / "1_control.png")
        tick("preprocess", s)

        log(0.10, "Understanding sketch"); s = time.perf_counter()                       # REQ 3
        und, sil = understand_sketch(pre, device=self.device)
        tick("understanding", s)

        prompt_final, prompt_source = build_prompt(prompt, und, style)                   # REQ 5

        log(0.20, "ControlNet: generating candidates"); s = time.perf_counter()          # REQ 6A
        cands = self.gen.candidates(pre["control"], prompt_final, seed, steps, n_candidates)
        tick("controlnet", s)

        log(0.50, "Removing background + ranking"); s = time.perf_counter()              # REQ 4
        raws = [self.bg.remove(c) for c in cands]
        scores = [candidate_score(pre["control"], sil, c, r) for c, r in zip(cands, raws)]
        best = int(np.argmax(scores))
        generated, raw = cands[best], raws[best]
        for i, c in enumerate(cands):
            c.save(run_dir / f"2_candidate_{i}_score{scores[i]:.2f}.png")
        tick("bg_removal+ranking", s)

        if refine:                                                                       # REQ 6B
            log(0.58, "Refining image"); s = time.perf_counter()
            generated = self.gen.refine(generated, pre["control"], prompt_final, seed)
            raw = self.bg.remove(generated)
            generated.save(run_dir / "2_refined.png")
            tick("refine", s)
        if self.low_vram:
            self.gen.unload()
        rgba = prepare_foreground(raw)
        rgba.save(run_dir / "3_object_rgba.png")

        log(0.65, "Reconstructing 3D mesh"); s = time.perf_counter()                     # REQ 7
        if use_fallback:
            method, mesh = "silhouette-inflation (fallback)", inflate_fallback(rgba)
        else:
            method, mesh = "TripoSR", self.rec(rgba_to_model_input(rgba))
        mesh = clean_mesh(mesh)
        tick("reconstruction", s)

        tex_img, tex_note = None, "vertex colours"                                       # REQ 8
        if texture:
            log(0.82, "Baking UV texture"); s = time.perf_counter()
            try:
                mesh, tex_img = bake_uv_texture(mesh)
                tex_img.save(run_dir / "4_texture.png")
                tex_note = "UV texture 1024x1024"
            except Exception as e:
                tex_note = f"vertex colours (texture bake failed: {str(e)[:60]})"
            tick("texture", s)

        log(0.93, "Exporting + preview"); s = time.perf_counter()                        # REQ 10, 9
        info_mesh = export_mesh(mesh, str(run_dir / "model"))
        previews = save_preview_files(info_mesh["glb"], run_dir)
        tick("export+preview", s)
        T["total"] = round(time.perf_counter() - t0, 2)

        report = {"method": method, "prompt_final": prompt_final, "prompt_source": prompt_source,
                  "style": style, "seed": seed, "steps": steps, "candidates": n_candidates, "refined": refine,
                  "fidelity_scores": [round(x, 3) for x in scores], "chosen_candidate": best,
                  "understanding": und, "texture": tex_note, "hardware": self.hw, "timings_sec": T,
                  "mesh": info_mesh, "preview_files": previews}
        (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(1.0, "Done")
        return {"run_dir": str(run_dir), "control": pre["control"], "candidates": cands,
                "generated": generated, "rgba": rgba, "texture": tex_img, "glb": info_mesh["glb"],
                "obj": info_mesh["obj"], "report": report}
