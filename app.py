"""Gradio web app: draw or upload a sketch -> textured 3D model.   python app.py [--share] [--port 7860]"""
import argparse
import json

import gradio as gr
import torch
from PIL import Image

from pipeline import Sketch2ShapePipeline
from pipeline.config import STYLE_PRESETS
from pipeline.export import MeshValidationError
from pipeline.preprocess import EmptySketchError

pipe = Sketch2ShapePipeline()


def generate(editor_value, uploaded, prompt, style, seed, steps, n_cand, refine, texture, fallback,
             progress=gr.Progress()):
    if uploaded is not None:                                      # upload overrides the canvas
        sketch = uploaded
    elif editor_value and editor_value.get("composite") is not None:
        sketch = editor_value["composite"]
    else:
        raise gr.Error("Draw a sketch on the canvas or upload an image first.")
    if not isinstance(sketch, Image.Image):
        sketch = Image.fromarray(sketch)
    try:
        r = pipe.run(sketch, prompt, style, int(seed), int(steps), int(n_cand), refine, texture, fallback,
                     progress=lambda f, d="": progress(f, desc=d))
    except EmptySketchError as e:
        raise gr.Error(str(e))
    except MeshValidationError as e:
        raise gr.Error(f"3D validation failed: {e} Try another seed or a clearer sketch.")
    except torch.cuda.OutOfMemoryError:
        raise gr.Error("GPU out of memory. Lower diffusion steps / candidates and retry.")
    except RuntimeError as e:
        raise gr.Error(str(e))
    rep = r["report"]
    cands = [(c, f"score {s}") for c, s in zip(r["candidates"], rep["fidelity_scores"])]
    info = (f"Detected: {rep['understanding']['label_top3']}\nPrompt ({rep['prompt_source']}): "
            f"{rep['prompt_final']}\nTexture: {rep['texture']}\nSaved in: {r['run_dir']}\n\n"
            + json.dumps(rep["timings_sec"], indent=2))
    return r["control"], cands, r["rgba"], r["glb"], [r["glb"], r["obj"]], info


def build_ui():
    with gr.Blocks(title="Sketch2Shape AI") as demo:
        gr.Markdown("# Sketch2Shape AI\nDraw or upload a sketch, optionally describe it, get a textured 3D model.")
        with gr.Row():
            with gr.Column():
                editor = gr.ImageEditor(
                    label="Draw your sketch (brush / eraser / undo / clear are in the toolbar)", type="pil",
                    image_mode="RGBA", canvas_size=(512, 512), sources=(), layers=False, transforms=(),
                    brush=gr.Brush(colors=["#000000"], default_color="#000000", default_size=4),
                    eraser=gr.Eraser(default_size=20))
                upload = gr.Image(label="...or upload a sketch image (overrides the canvas)", type="pil",
                                  sources=["upload"])
                prompt = gr.Textbox(label="Optional prompt", placeholder="e.g. wooden office chair, matte black")
                style = gr.Dropdown(list(STYLE_PRESETS), value="none", label="Style preset")
                with gr.Accordion("Advanced", open=False):
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    steps = gr.Slider(10, 40, value=20, step=1, label="Diffusion steps (fewer = faster)")
                    n_cand = gr.Slider(1, 4, value=3, step=1, label="Candidates (best is chosen automatically)")
                    refine = gr.Checkbox(label="Extra img2img refinement (slower)")
                    texture = gr.Checkbox(value=True, label="Bake UV texture map")
                    fallback = gr.Checkbox(label="Fallback mode (no TripoSR, low fidelity, emergencies only)")
                btn = gr.Button("Generate 3D model", variant="primary")
            with gr.Column():
                model3d = gr.Model3D(label="3D preview (drag to rotate)", clear_color=(0.95, 0.95, 0.95, 1))
                files_out = gr.File(label="Download .glb / .obj", file_count="multiple")
                with gr.Row():
                    o1 = gr.Image(label="Processed sketch", height=170)
                    o3 = gr.Image(label="Isolated object", height=170)
                gallery = gr.Gallery(label="Generated candidates", columns=4, height=190)
                info = gr.Textbox(label="Run info", lines=9)
        btn.click(generate, [editor, upload, prompt, style, seed, steps, n_cand, refine, texture, fallback],
                  [o1, gallery, o3, model3d, files_out, info])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build_ui().queue().launch(server_name=a.host, server_port=a.port, share=a.share, inbrowser=True)
