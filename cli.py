"""Command-line interface.

    python cli.py examples/mug.png --prompt "red ceramic mug" --style ceramic
    python cli.py examples/chair.png                  # prompt is optional
    python cli.py --warmup                            # download + load all models once
"""
import argparse
import json

from PIL import Image

from pipeline import Sketch2ShapePipeline
from pipeline.config import STYLE_PRESETS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sketch", nargs="?", help="path to a sketch image (png/jpg)")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--style", default="none", choices=list(STYLE_PRESETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--refine", action="store_true", help="extra img2img refinement pass")
    ap.add_argument("--no-texture", action="store_true", help="skip UV texture baking (keep vertex colours)")
    ap.add_argument("--fallback", action="store_true", help="no-AI silhouette inflation (low fidelity)")
    ap.add_argument("--warmup", action="store_true", help="download and load all models, then exit")
    a = ap.parse_args()

    pipe = Sketch2ShapePipeline()
    print("hardware:", pipe.hw)
    if a.warmup:
        pipe.warmup()
        print("All models downloaded and loaded.")
        return
    if not a.sketch:
        ap.error("give a sketch path (or use --warmup)")
    r = pipe.run(Image.open(a.sketch), a.prompt, a.style, a.seed, a.steps, a.candidates, a.refine,
                 not a.no_texture, a.fallback,
                 progress=lambda f, d="": print(f"[{f * 100:3.0f}%] {d}"))
    print(json.dumps({k: r["report"][k] for k in ("method", "prompt_final", "texture", "timings_sec")}, indent=2))
    print("Model:", r["glb"])
    print("Open  :", r["run_dir"])


if __name__ == "__main__":
    main()
