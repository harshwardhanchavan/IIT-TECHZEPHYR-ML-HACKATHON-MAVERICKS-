# Sketch2Shape AI

**From a 2D sketch to a textured 3D asset.** Draw a rough sketch (or upload one), optionally add a text prompt, and get a
`.glb` / `.obj` 3D model in under a minute on a consumer GPU. Built for **TechZephyr 4.0 (IIT Bhubaneswar)** -
*Sketch2Shape AI: 24-hour live hackathon, Computer Vision & Generative AI*.

```
sketch ──► preprocess ──► understand ──► prompt ──► SD1.5 + ControlNet ──► background removal ──► TripoSR 3D
 (draw /     (denoise,     (contours,    (optional,   (N candidates,        (rembg, center)       (image → mesh)
  upload)     scribble)     CLIP label)   auto-fill)   ranked by fidelity)                              │
                                                                                                        ▼
                         3D preview ◄── export + validation ◄── UV texture bake (xatlas) ◄── mesh cleanup
                         (viewer, PNG)   (.glb + .obj)           (1024² texture map)
```

## Features (the ten required stages)

| # | Stage | Module | Technique |
|---|---|---|---|
| 1 | Sketch input | `app.py`, `cli.py` | in-browser canvas (brush, eraser, undo, clear) **and** image upload |
| 2 | Image preprocessing | `pipeline/preprocess.py` | denoise, contrast stretch, auto-invert, Otsu, crop/center, 512² scribble map |
| 3 | Sketch understanding | `pipeline/understand.py` | contours, symmetry, silhouette, CLIP zero-shot label, optional trained CNN |
| 4 | Background removal | `pipeline/background.py` | rembg (U²-Net) + foreground centering |
| 5 | Text prompt | `pipeline/prompt.py` | optional; user text > detected label > default; style presets |
| 6 | AI image refinement | `pipeline/generate.py` | SD 1.5 + ControlNet-Scribble, N candidates ranked by sketch fidelity, optional img2img polish |
| 7 | 3D reconstruction | `pipeline/reconstruct.py` | TripoSR (image → mesh); no-AI fallback for emergencies |
| 8 | Texture generation | `pipeline/texture.py` | xatlas UV unwrap + baked 1024² texture map |
| 9 | 3D preview | `pipeline/preview.py`, `app.py` | interactive viewer in the app, standalone `preview.html`, multi-view `preview.png` |
| 10 | 3D export | `pipeline/export.py` | validated `.glb` + `.obj`, files re-opened after export |

Candidate ranking score = ½·(share of your strokes that still appear as edges in the generated image) + ½·(IoU between
your filled silhouette and the generated object's mask). This keeps the result tied to **your sketch**, not just the prompt.

## Repository layout

```
sketch2shape-ai/
├── app.py               Gradio web app (canvas + upload + prompt + 3D viewer + downloads)
├── cli.py               command-line interface (also --warmup to pre-download models)
├── create_examples.py   generates 6 test sketches into examples/
├── dataset.py           Google QuickDraw loader (for the optional classifier)
├── train.py             trains the optional sketch classifier -> models/sketch_cnn.pt
├── run.bat / run.ps1    one-command setup and launchers for Windows
├── requirements.txt
├── pipeline/            the library: config, preprocess, understand, prompt, generate, background,
│                        reconstruct, texture, export, preview, classifier, pipeline (orchestrator)
├── examples/            input sketches (mug, chair, car, house, lamp, bottle)
├── models/              TripoSR source (cloned), Hugging Face cache, trained classifier (all git-ignored except .gitkeep)
├── outputs/             one timestamped folder per run (git-ignored)
└── tests/               unit tests (CPU only)
```

## Quick start (Windows)

Requirements: **Python 3.10** (3.11 also works), **Git**, an **NVIDIA GPU with >= 8 GB VRAM** and a recent driver (`nvidia-smi` must work).

```bat
git clone https://github.com/<your-username>/sketch2shape-ai.git
cd sketch2shape-ai
run.bat setup      :: venv + PyTorch (CUDA 12.1) + dependencies + TripoSR clone
run.bat warmup     :: downloads ~10 GB of model weights once
run.bat app        :: opens http://127.0.0.1:7860
```

PowerShell users: `powershell -ExecutionPolicy Bypass -File run.ps1 setup`, then `.\run.ps1 app`.

**Linux / macOS / manual install**
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
git clone https://github.com/VAST-AI-Research/TripoSR.git models/TripoSR
python create_examples.py && python app.py
```
**No local GPU?** Run on Google Colab (GPU runtime): same commands, then `python app.py --share` for a public link.

No `torchmcubes` compilation is needed: the repo registers a scikit-image marching-cubes implementation under that name
(`pipeline/reconstruct.py::install_mcubes_stub`), which removes the most common install failure on Windows.

## Usage

```bat
run.bat app --share                                   :: web UI with a public link
run.bat cli examples\mug.png --prompt "red ceramic mug" --style ceramic
run.bat cli examples\chair.png                        :: prompt is optional
run.bat cli examples\car.png --candidates 4 --refine  :: more candidates + img2img polish
run.bat test                                          :: unit tests
```

Each run creates `outputs/<timestamp>/` containing:

| File | Content |
|---|---|
| `model.glb`, `model.obj` | the 3D asset (open in Blender, Windows 3D Viewer, https://gltf-viewer.donmccurdy.com) |
| `4_texture.png` | baked UV texture (when baking succeeds) |
| `preview.html`, `preview.png` | interactive viewer page and 4-angle render |
| `1_control.png`, `2_candidate_*.png`, `3_object_rgba.png` | intermediate images |
| `report.json` | prompt, detected label, fidelity scores, timings per stage, hardware, mesh statistics |

## Optional: train the sketch classifier

The main models (Stable Diffusion, ControlNet, TripoSR, CLIP) are **pretrained**. The only model trained in this repo is a
small CNN on Google QuickDraw that gives a second opinion on what a sketch shows (used only when CLIP is unsure).

```bat
run.bat train --quick      :: smoke test: 5 categories, ~1 minute after download
run.bat train              :: 20 categories x 4000 images, 8 epochs; writes models\sketch_cnn.pt + metrics json
```
Downloads about 100 MB per category into `data/` (git-ignored). QuickDraw doodles differ from real sketches, so treat its
predictions as hints.

## Models and third-party components (disclosure)

| Component | Source | License |
|---|---|---|
| Stable Diffusion 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` (Hugging Face) | CreativeML OpenRAIL-M |
| ControlNet scribble | `lllyasviel/sd-controlnet-scribble` | CreativeML OpenRAIL-M |
| TripoSR | github.com/VAST-AI-Research/TripoSR, weights `stabilityai/TripoSR` | MIT |
| rembg / U²-Net | github.com/danielgatis/rembg | MIT |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | MIT |
| QuickDraw dataset (optional) | Google Creative Lab | CC BY 4.0 |
| Diffusers, Transformers, PyTorch, OpenCV, trimesh, xatlas, scikit-image, Gradio | PyPI | Apache-2.0 / BSD / MIT |

No paid external APIs. The internet is only needed to download weights and for the browser-side 3D viewer script.

## Hardware, runtime and reproducibility

| | |
|---|---|
| Tested configuration | **<fill in: GPU model, VRAM, driver, OS, Python>** |
| Time per model | **<fill in from outputs/\*/report.json -> timings_sec.total>** (first run is slower: model loading) |
| Minimum | NVIDIA GPU with 8 GB VRAM (low-VRAM mode switches on automatically below 12 GB), ~15 GB free disk |
| Weights location | `models/hf_cache` (set `HF_HOME` to override) |
| Random seed | `--seed` (default 42) makes candidate generation repeatable on the same hardware |

## Testing

`run.bat test` runs `tests/test_cpu.py`: preprocessing (including inverted and empty inputs), shape statistics, prompt logic,
candidate scoring, mesh validation (flat meshes are rejected), export round-trip, UV texture baking, classifier shapes. Tests
that need trimesh/torch/xatlas skip automatically if those packages are missing. Full end-to-end verification needs a GPU:
run `run.bat cli examples\mug.png` and open the resulting `model.glb`.

## Limitations

- Hidden sides are inferred, not recovered; the back of the object is plausible but not exact.
- Thin structures, very abstract doodles and multi-object scenes can fail or drift from the sketch.
- The texture is baked from TripoSR's per-vertex colours (per-triangle colour at 1024²); it is not a repainted PBR material.
- CLIP / QuickDraw labels are hints and can be wrong; a user prompt always wins.
- Output is a good starting mesh, not production topology, not 3D-print-ready.
- Fallback mode (silhouette inflation) is low fidelity and exists only so a demo never dies.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot uninstall numpy ... no RECORD file` | You installed into the base Anaconda env. Use `run.bat setup` (creates `.venv`) or a fresh conda env. |
| `torchmcubes must be built against an installed PyTorch` | Not needed here - do not install `torchmcubes`; the repo ships a replacement. |
| `No NVIDIA GPU detected` | Install the CUDA build of PyTorch (`nvidia-smi` must work) or use Google Colab. |
| CUDA out of memory | Lower *Diffusion steps* / *Candidates*; close other GPU programs. Low-VRAM mode is automatic below 12 GB. |
| `TripoSR source not found` | `git clone https://github.com/VAST-AI-Research/TripoSR.git models/TripoSR` |
| 3D viewer blank in the app | Open `outputs/<run>/preview.html` or load `model.glb` in https://gltf-viewer.donmccurdy.com |
| Mesh looks mirrored or scrambled | Marching-cubes axis order: open an issue with a screenshot; flip axes in `install_mcubes_stub`. |
| Result ignores the sketch | Raise `control_scale` in `pipeline/generate.py` (try 1.2) or simplify the prompt. |

## Acknowledgements

TripoSR (Stability AI & Tripo AI), Stable Diffusion (CompVis / Stability AI / Runway), ControlNet (L. Zhang et al.),
rembg (D. Gatis), CLIP (OpenAI), Google QuickDraw, Hugging Face Diffusers, trimesh, xatlas.

## License

MIT for this repository's code (see `LICENSE`). The pretrained models keep their own licenses listed above.
