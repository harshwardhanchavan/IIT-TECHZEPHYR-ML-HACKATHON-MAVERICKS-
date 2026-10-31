# Sketch2Shape AI: Project Documentation

**From a 2D sketch to a textured 3D asset.** TechZephyr 4.0, IIT Bhubaneswar, 24-hour live hackathon (Computer Vision & Generative AI).

Repository: https://github.com/harshwardhanchavan/IIT-TECHZEPHYR

---

## Table of contents

1. [Overview](#1-overview)
2. [Problem statement and how it is solved](#2-problem-statement-and-how-it-is-solved)
3. [System architecture](#3-system-architecture)
4. [Pipeline stages in detail](#4-pipeline-stages-in-detail)
5. [Repository structure](#5-repository-structure)
6. [Installation](#6-installation)
7. [Usage](#7-usage)
8. [Outputs and the run report](#8-outputs-and-the-run-report)
9. [Models and third-party components](#9-models-and-third-party-components)
10. [Hardware, runtime and reproducibility](#10-hardware-runtime-and-reproducibility)
11. [Training the optional sketch classifier](#11-training-the-optional-sketch-classifier)
12. [Testing](#12-testing)
13. [Design decisions](#13-design-decisions)
14. [Code reference](#14-code-reference)
15. [Deployment options](#15-deployment-options)
16. [Limitations](#16-limitations)
17. [Troubleshooting](#17-troubleshooting)
18. [Mapping to the evaluation rubric](#18-mapping-to-the-evaluation-rubric)
19. [Future work](#19-future-work)
20. [License and acknowledgements](#20-license-and-acknowledgements)

---

## 1. Overview

Sketch2Shape AI turns a rough single-view sketch into a textured 3D model that can be opened in standard 3D software.

**What the user does**

1. Draws a sketch in the web app (brush, eraser, undo, clear) or uploads a sketch image.
2. Optionally types a text prompt (material, colour, style). The prompt is never required.
3. Clicks **Generate 3D model**.
4. Rotates the result in the built-in 3D viewer and downloads `.glb` / `.obj` files.

**What the system does**

1. Cleans the sketch and converts it to a scribble map.
2. Works out what the sketch probably shows (shape statistics and a CLIP guess).
3. Generates several clean product-style images with Stable Diffusion guided by ControlNet.
4. Removes the background, scores each candidate by how faithfully it follows the sketch, and keeps the best.
5. Reconstructs a 3D mesh with TripoSR, cleans it, bakes a UV texture map, validates it, and exports it.

**Typical use cases:** product design and prototyping, architecture and interior design, animation / VFX / games, education and creative exploration.

---

## 2. Problem statement and how it is solved

The hackathon asks for a working end-to-end application that accepts a single-view 2D sketch, optionally a text prompt, and produces an exportable, textured 3D asset.

| Requirement | Where it is implemented |
|---|---|
| In-app drawing canvas (brush, eraser, reset) | `app.py`, Gradio `ImageEditor` |
| Image upload | `app.py`, Gradio `Image` input (overrides the canvas) |
| Input preparation (resize, background, contrast, noise) | `pipeline/preprocess.py` |
| Optional text prompt | `pipeline/prompt.py` (user text, else detected label, else default) |
| Sketch understanding | `pipeline/understand.py` |
| Sketch-guided image generation that stays related to the sketch | `pipeline/generate.py` (ControlNet-Scribble plus fidelity ranking) |
| Background removal / foreground isolation | `pipeline/background.py` |
| Image-to-3D reconstruction | `pipeline/reconstruct.py` (TripoSR) |
| Texture and appearance | `pipeline/texture.py` (UV texture map) |
| 3D preview | Gradio `Model3D` viewer, `preview.html`, `preview.png` |
| Export in .glb or .obj | `pipeline/export.py` (both formats) |
| Mesh validation | `pipeline/export.py` (empty, flat, and re-open checks) |
| Setup instructions and reproducible run | `README.md`, `run.bat`, `run.ps1`, `colab_launcher.ipynb`, this document |
| Report of hardware, runtime, limitations | Sections 10 and 16, plus `report.json` per run |
| Disclosure of pretrained models | Section 9 |

---

## 3. System architecture

```mermaid
flowchart LR
  A[Sketch: canvas or upload] --> B[Preprocess]
  B --> C[Sketch understanding]
  C --> D[Prompt builder]
  D --> E[SD 1.5 + ControlNet: N candidates]
  E --> F[Background removal]
  F --> G[Rank candidates by fidelity]
  G --> H[TripoSR image-to-3D]
  H --> I[Mesh cleanup]
  I --> J[UV texture bake]
  J --> K[Validate + export .glb / .obj]
  K --> L[Preview + report.json]
```

**Layers**

| Layer | Files | Responsibility |
|---|---|---|
| Interfaces | `app.py`, `cli.py` | Web UI and command line. No ML logic. |
| Orchestration | `pipeline/pipeline.py` | Runs all stages in order, collects timings, writes `report.json`. |
| Stages | `pipeline/*.py` | One module per stage, each independently testable. |
| Configuration | `pipeline/config.py` | Paths, model ids, prompt constants. Sets the model cache inside `models/`. |
| Training | `dataset.py`, `train.py` | Optional QuickDraw classifier. |
| Tooling | `run.bat`, `run.ps1`, `create_examples.py`, `tests/` | Setup, launch, sample data, tests. |

Models are loaded lazily and reused. On GPUs with less than 12 GB of VRAM the app switches to a low-VRAM mode: Stable Diffusion is offloaded and then unloaded before the 3D model runs.

---

## 4. Pipeline stages in detail

### 4.1 Sketch input (REQ 1)
The web app offers a drawing canvas (black brush, eraser, undo, clear) and an upload box. If an image is uploaded, it overrides the canvas. The CLI takes a file path. Transparent images are flattened on white.

### 4.2 Image preprocessing (REQ 2), `pipeline/preprocess.py`
Function `preprocess_sketch(img)`:

1. Flatten transparency on white and convert to grayscale.
2. Cap the longest side at 1024 px.
3. Gaussian blur (noise reduction) and min-max contrast normalisation.
4. Auto-invert if the image is dark (white-on-black input).
5. Otsu threshold to a binary stroke mask, then remove tiny specks.
6. Reject near-empty sketches with `EmptySketchError` (less than 0.2 % ink).
7. Crop to the strokes, pad to a square with margin, resize to 512 x 512.
8. Thicken strokes slightly so ControlNet can see them.

Returns `control` (white strokes on black, ControlNet's scribble format) and `clean` (black on white).

### 4.3 Sketch understanding (REQ 3), `pipeline/understand.py`
Two parts:

- **Classical computer vision** (`shape_statistics`): contour count, number of closed regions, aspect ratio, stroke density, solidity, left-right symmetry, and a filled **silhouette** if the strokes enclose real area. The silhouette is later used to score candidates.
- **Learned recognition**: CLIP ViT-B/32 zero-shot classification against 95 common object names (`pipeline/config.py::LABELS`) using the template "a simple line drawing of a {label}". Returns the top three guesses. If `models/sketch_cnn.pt` exists, a small QuickDraw-trained CNN adds a second guess (`classifier_guess`).

### 4.4 Text prompt (REQ 5), `pipeline/prompt.py`
`build_prompt(user_prompt, understanding, style)` picks the prompt source in this order:

1. The user's text, if given.
2. The CLIP label, if its probability is at least 0.25 (or the CNN guess if CLIP is unsure and the CNN is at least 0.6 confident).
3. A generic default: "a single object, product design render".

A style preset (none, realistic, toy / cartoon, low-poly, metallic, wood, ceramic) is appended, followed by a fixed suffix that asks for a centred, fully visible object on a plain white background. A negative prompt suppresses clutter, text, and cropped objects.

### 4.5 AI image refinement (REQ 6), `pipeline/generate.py`
- **Stage A, candidates:** Stable Diffusion 1.5 with the ControlNet-Scribble adapter turns the scribble map into N candidate images (1 to 4, default 3). UniPC scheduler, 20 steps by default, fixed seed for repeatability.
- **Stage B, optional polish:** a low-strength (0.4) ControlNet img2img pass on the chosen candidate. It shares weights with Stage A.
- **Candidate score** (`candidate_score`) keeps the output tied to the sketch:

  `score = 0.5 * edge_recall + 0.5 * silhouette_IoU`

  - *edge recall*: share of the user's stroke pixels that have a Canny edge of the generated image within 12 px.
  - *silhouette IoU*: overlap between the sketch's filled silhouette and the object mask from background removal. Used only when the sketch is closed; otherwise the score is edge recall alone.

### 4.6 Background removal (REQ 4), `pipeline/background.py`
rembg with the U²-Net model isolates the object. `prepare_foreground` crops to the object, scales it to 85 % of a 512 x 512 frame and centres it. `rgba_to_model_input` composites it on neutral grey, which is what TripoSR expects. This runs on every candidate (needed for ranking) and again after the optional refinement.

### 4.7 3D reconstruction (REQ 7), `pipeline/reconstruct.py`
- **Primary: TripoSR** (single image to triplane to marching cubes), mesh resolution 256, per-vertex colour output.
- **`torchmcubes` replacement:** TripoSR imports `torchmcubes`, a CUDA extension that is difficult to compile (especially on Windows). `install_mcubes_stub()` registers a scikit-image marching-cubes function under that name. It runs on the CPU in roughly 1 to 3 seconds and needs no compiler.
- **Fallback mode:** a no-AI "silhouette inflation" mesh (`inflate_fallback`), for emergencies only. It is low fidelity and clearly labelled in `report.json`.
- **Cleanup** (`clean_mesh`): remove degenerate faces, keep the largest connected component, fix normals, centre, and scale to a unit box.

### 4.8 Texture generation (REQ 8), `pipeline/texture.py`
`bake_uv_texture(mesh)` converts per-vertex colours into a real UV texture map:

1. Decimate meshes above 60,000 faces; colours are transferred to the decimated vertices by nearest neighbour.
2. UV-unwrap with xatlas.
3. Rasterise every triangle into a 1024 x 1024 texture with its mean vertex colour, then fill seams by inpainting.
4. Attach the texture to the mesh (`TextureVisuals`). The exported GLB then carries a base-colour texture.

If baking fails, the pipeline keeps vertex colours and records this in `report.json`. The texture is baked from TripoSR's colours; it is not a repainted PBR material.

### 4.9 3D preview (REQ 9), `pipeline/preview.py`
- Interactive viewer inside the app (Gradio `Model3D`).
- `preview.html`: a standalone page using Google `<model-viewer>`; open it by double-click (the browser needs internet for the viewer script).
- `preview.png`: a 4-angle point render that works without any viewer.

### 4.10 3D export and validation (REQ 10), `pipeline/export.py`
`export_mesh` validates then writes `model.glb` and `model.obj`:

- Rejects empty meshes.
- Rejects flat meshes (smallest extent below 3 % of the largest), so a textured plane can never be exported as "3D".
- Re-opens both exported files and checks that vertices and faces survive.
- Reports vertex and face counts, extents, watertightness, and whether a texture survived the reload.

---

## 5. Repository structure

```
IIT-TECHZEPHYR/
├── app.py                 Gradio web app (canvas, upload, prompt, 3D viewer, downloads)
├── cli.py                 Command-line interface (also --warmup)
├── create_examples.py     Generates 6 example sketches into examples/
├── dataset.py             QuickDraw loader for the optional classifier
├── train.py               Trains the optional classifier -> models/sketch_cnn.pt
├── run.bat / run.ps1      Windows setup and launchers
├── colab_launcher.ipynb   One-click Google Colab launcher (optional file)
├── requirements.txt       Pinned dependencies (PyTorch is installed separately)
├── README.md              Quick start
├── DOCUMENTATION.md       This document
├── LICENSE                MIT
├── pipeline/
│   ├── __init__.py        Lazy exports
│   ├── config.py          Paths, model ids, prompt constants
│   ├── preprocess.py      REQ 2
│   ├── understand.py      REQ 3
│   ├── classifier.py      Small QuickDraw CNN (optional)
│   ├── prompt.py          REQ 5
│   ├── generate.py        REQ 6
│   ├── background.py      REQ 4
│   ├── reconstruct.py     REQ 7
│   ├── texture.py         REQ 8
│   ├── preview.py         REQ 9
│   ├── export.py          REQ 10
│   └── pipeline.py        Orchestrator
├── examples/              mug, chair, car, house, lamp, bottle sketches
├── models/                TripoSR source, Hugging Face cache, sketch_cnn.pt (git-ignored except .gitkeep)
├── outputs/               One timestamped folder per run (git-ignored)
└── tests/test_cpu.py      Unit tests (no GPU needed)
```

---

## 6. Installation

### 6.1 Requirements
- An **NVIDIA GPU with at least 8 GB of VRAM** (a driver where `nvidia-smi` works), or Google Colab with a GPU runtime.
- About 15 GB of free disk (about 10 GB of model weights).
- Python 3.10 (3.11 also works) and Git for local installs.

### 6.2 Windows (local)
```bat
git clone https://github.com/harshwardhanchavan/IIT-TECHZEPHYR.git
cd IIT-TECHZEPHYR
run.bat setup      :: creates .venv, installs CUDA PyTorch + dependencies, clones TripoSR
run.bat warmup     :: downloads about 10 GB of weights once
run.bat app        :: opens http://127.0.0.1:7860
```
PowerShell: `powershell -ExecutionPolicy Bypass -File run.ps1 setup`, then `.\run.ps1 app`.

> Do not install into the base Anaconda environment: it commonly fails with `Cannot uninstall numpy ... no RECORD file`. `run.bat setup` creates an isolated `.venv`.

### 6.3 Linux / macOS (manual)
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
git clone https://github.com/VAST-AI-Research/TripoSR.git models/TripoSR
python create_examples.py
python app.py
```

### 6.4 Google Colab
Open `colab_launcher.ipynb` in Colab, choose **Runtime, Change runtime type, T4 GPU**, then **Runtime, Run all**. The last cell prints a public `https://xxxx.gradio.live` link and keeps running while the app is live.

Colab runs a newer Python (3.13 at the time of writing) than the pins in `requirements.txt`, so the launcher installs different versions:
`transformers>=4.46,<5`, `huggingface_hub>=0.34,<1.0`, `diffusers==0.35.2`, `gradio==4.44.1`, `pydantic==2.10.6`.

Manual Colab commands (one cell each):
```
%cd /content
!git clone https://github.com/harshwardhanchavan/IIT-TECHZEPHYR.git
%cd IIT-TECHZEPHYR
!git clone https://github.com/VAST-AI-Research/TripoSR.git models/TripoSR
!pip install -q trimesh networkx fast-simplification xatlas rembg onnxruntime omegaconf einops imageio scikit-image gradio==4.44.1 pydantic==2.10.6 "transformers>=4.46,<5" "huggingface_hub>=0.34,<1.0" "diffusers==0.35.2" accelerate
!python create_examples.py
!python cli.py --warmup
!python app.py --share
```
Pip may print red dependency-conflict warnings about packages that Colab pre-installs (for example `google-genai` or `langsmith`). They concern packages this project does not use and can be ignored.

---

## 7. Usage

### 7.1 Web app
```
python app.py                      # local, http://127.0.0.1:7860
python app.py --share              # public gradio.live link (Colab or any machine)
python app.py --host 0.0.0.0 --port 8080
```

| Control | Meaning |
|---|---|
| Canvas | Draw with the brush, erase, undo or clear. |
| Upload | A sketch image; overrides the canvas. |
| Optional prompt | Free text, e.g. `red ceramic coffee mug`. |
| Style preset | Adds material and style keywords. |
| Seed | Makes candidate generation repeatable. |
| Diffusion steps (10 to 40) | Fewer steps are faster. Default 20. |
| Candidates (1 to 4) | More candidates give a better match to the sketch but take longer. Default 3. |
| Extra img2img refinement | Optional polish pass, slower. |
| Bake UV texture map | On by default. |
| Fallback mode | No-AI emergency mode, low fidelity. |

**Fast settings for a live demo:** steps 12, candidates 1, texture off. **Best quality:** steps 20 to 25, candidates 3 to 4, texture on, refinement on.

### 7.2 Command line
```
python cli.py examples/mug.png --prompt "red ceramic mug" --style ceramic
python cli.py examples/chair.png                      # prompt is optional
python cli.py examples/car.png --candidates 4 --refine
python cli.py examples/lamp.png --no-texture --steps 12 --candidates 1
python cli.py --warmup                                # download and load all models
```

### 7.3 Launcher shortcuts (Windows)
`run.bat setup | warmup | app | cli ... | examples | train | test`

### 7.4 Drawing tips
- One object, large and centred, with thick clear lines.
- Closed outlines help (the silhouette is used for ranking).
- Simple side or slight 3/4 view, few details.
- Add a short prompt for material and colour.
- Easiest objects: mug, cup, bottle, vase, lamp, chair, stool, table, teapot, bowl. Harder: bicycles, trees, animals, anything with thin spokes or legs.

---

## 8. Outputs and the run report

Every run creates `outputs/<timestamp>/`:

| File | Content |
|---|---|
| `model.glb`, `model.obj` | The 3D asset (open in Blender, Windows 3D Viewer, or https://gltf-viewer.donmccurdy.com). |
| `4_texture.png` | The baked UV texture (when baking succeeds). |
| `preview.html`, `preview.png` | Interactive viewer page and 4-angle render. |
| `1_sketch_clean.png`, `1_control.png` | Processed sketch and the ControlNet scribble map. |
| `2_candidate_<i>_score<s>.png` | Each generated candidate with its fidelity score. |
| `2_refined.png` | Refined image (only when refinement is on). |
| `3_object_rgba.png` | The isolated object that was sent to TripoSR. |
| `report.json` | Everything below. |

**`report.json` fields**

| Field | Meaning |
|---|---|
| `method` | `TripoSR` or `silhouette-inflation (fallback)` |
| `prompt_final`, `prompt_source` | The prompt used and where it came from (user, auto label, default) |
| `understanding` | Shape statistics, CLIP top-3 labels, optional CNN guess |
| `fidelity_scores`, `chosen_candidate` | Score per candidate and the winner |
| `texture` | `UV texture 1024x1024` or why vertex colours were kept |
| `timings_sec` | Seconds per stage and `total` |
| `hardware` | GPU name and VRAM |
| `mesh` | Vertex and face counts, extents, watertight flag, texture survival after reload |

---

## 9. Models and third-party components

All pretrained models are disclosed here, as the problem statement requires.

| Component | Used for | Source | License |
|---|---|---|---|
| Stable Diffusion 1.5 | Image generation | `stable-diffusion-v1-5/stable-diffusion-v1-5` (Hugging Face) | CreativeML OpenRAIL-M |
| ControlNet scribble | Sketch conditioning | `lllyasviel/sd-controlnet-scribble` | CreativeML OpenRAIL-M |
| TripoSR | Image-to-3D | github.com/VAST-AI-Research/TripoSR, weights `stabilityai/TripoSR` | MIT |
| rembg (U²-Net) | Background removal | github.com/danielgatis/rembg | MIT |
| CLIP ViT-B/32 | Sketch label guess | `openai/clip-vit-base-patch32` | MIT |
| QuickDraw (optional) | Training data for the small CNN | Google Creative Lab | CC BY 4.0 |
| Libraries | PyTorch, Diffusers, Transformers, Accelerate, OpenCV, scikit-image, SciPy, trimesh, xatlas, Gradio, matplotlib | PyPI | Apache-2.0 / BSD / MIT |

No paid external APIs are used. The internet is needed only to download weights and, in the browser, for the 3D viewer script.

---

## 10. Hardware, runtime and reproducibility

| Item | Value |
|---|---|
| Minimum hardware | NVIDIA GPU with 8 GB VRAM, about 15 GB disk. Low-VRAM mode switches on automatically below 12 GB. |
| Tested configuration | **<FILL IN: GPU model, VRAM, driver, OS, Python version>** |
| Time per 3D model | **<FILL IN from `report.json`, `timings_sec.total`>** with default settings (3 candidates, 20 steps, texture on) |
| First run | Slower: model loading, plus a one-time download of about 10 GB of weights |
| Weight location | `models/hf_cache` (override with the `HF_HOME` environment variable) |
| Determinism | `--seed` fixes candidate generation on the same hardware and library versions |

**Reproduction checklist**

1. Use a clean environment (`run.bat setup`, or the Colab launcher).
2. Run `python create_examples.py` to create the test sketches.
3. Run `python cli.py --warmup` once.
4. Run `python cli.py examples/mug.png --prompt "red ceramic coffee mug"` and open `outputs/<latest>/model.glb`.
5. Compare your `report.json` with the measured values above.

---

## 11. Training the optional sketch classifier

The main models are pretrained. The only model trained in this repository is a small CNN on Google QuickDraw. It gives a second opinion on what a sketch shows, and it is used only when CLIP is unsure.

```
python train.py --quick      # smoke test: 5 categories, 500 images each, 2 epochs
python train.py              # 20 categories x 4000 images, 8 epochs
python train.py --categories chair table cup car --epochs 10 --n-train 3000
```

- **Data:** QuickDraw 28 x 28 bitmaps, downloaded to `data/quickdraw` (about 100 MB per category).
- **Model:** three convolution layers, two fully connected layers, dropout 0.3, about 0.4 M parameters.
- **Training:** AdamW with a OneCycle schedule, light augmentation (random shift, random stroke thickening).
- **Outputs:** `models/sketch_cnn.pt` (weights and category list) and `models/sketch_cnn_metrics.json` (test accuracy, per-class accuracy, training time).
- **Caveat:** QuickDraw doodles differ from real sketches, so treat predictions as hints.

**Results:** <FILL IN: best test accuracy and training time from the metrics file>

---

## 12. Testing

Run `python -m pytest -q tests` (or `run.bat test`). Tests are in `tests/test_cpu.py` and need no GPU.

| Test | What it checks |
|---|---|
| Preprocess output format | 512 x 512, strokes white on black |
| Empty sketch rejected | `EmptySketchError` on a blank image |
| Dark paper | An inverted input gives the same scribble map |
| All six examples | Every example sketch preprocesses and has contours |
| Silhouette | Closed sketches produce a filled silhouette |
| Prompt logic | User prompt wins; auto label; default when unsure |
| Candidate scoring | A matching image scores higher than a blank one |
| Mesh validation | A flat mesh is rejected; a box exports and re-opens |
| Fallback mesh | The fallback produces a real 3D volume |
| Texture bake | A sphere gets a UV texture of the requested size |
| Classifier shapes | CNN output shape and the 28 x 28 conversion |

Tests that need `trimesh`, `torch` or `xatlas` skip automatically if those packages are missing.

**End-to-end verification (needs a GPU):** run the CLI on each example, open the `.glb`, and record the results.

| Sketch | Detected label | Best score | Time (s) | Looks right? |
|---|---|---|---|---|
| mug | <fill in> | <fill in> | <fill in> | <fill in> |
| chair | | | | |
| car | | | | |
| house | | | | |
| lamp | | | | |
| bottle | | | | |

---

## 13. Design decisions

| Decision | Reason |
|---|---|
| **ControlNet-Scribble** instead of plain text-to-image | The generated image must follow the sketch, not drift to an unrelated object. |
| **Several candidates, ranked by edge recall and silhouette IoU** | A cheap, model-free way to keep the output tied to the user's drawing. |
| **TripoSR** instead of InstantMesh or Unique3D | Fast (seconds), modest VRAM, MIT license, and a simpler install. InstantMesh and Unique3D need heavier dependencies. |
| **scikit-image marching cubes in place of `torchmcubes`** | Removes the most common install failure, especially on Windows. |
| **CLIP zero-shot labels** | Open vocabulary, so it handles many object types without training. |
| **Optional QuickDraw CNN** | Gives the project a genuinely trained component and a second opinion, without being required. |
| **Baked UV texture** | A standard texture map works in all 3D software, beyond per-vertex colours. |
| **Strict mesh validation** | Rejects empty or flat output and proves the exported files re-open. |
| **Model cache inside `models/`** | Keeps the project self-contained and easy to clean. |
| **Fallback mode** | Keeps a demo alive if the main path fails; clearly labelled so it is never mistaken for the real result. |

---

## 14. Code reference

| Module | Key functions and classes |
|---|---|
| `pipeline/config.py` | `hardware_info()`, path and model-id constants, `STYLE_PRESETS`, `LABELS`, `NEGATIVE` |
| `pipeline/preprocess.py` | `preprocess_sketch(img)`, `EmptySketchError` |
| `pipeline/understand.py` | `understand_sketch(pre, device)`, `shape_statistics(control)`, `clip_label(img, device)` |
| `pipeline/classifier.py` | `SketchCNN`, `control_to_28(img)`, `load_classifier()`, `predict(img)` |
| `pipeline/prompt.py` | `build_prompt(user_prompt, understanding, style)` |
| `pipeline/generate.py` | `SketchGenerator.candidates(...)`, `.refine(...)`, `.unload()`, `candidate_score(...)` |
| `pipeline/background.py` | `BackgroundRemover.remove(img)`, `prepare_foreground(rgba)`, `rgba_to_model_input(rgba)` |
| `pipeline/reconstruct.py` | `Reconstructor`, `install_mcubes_stub()`, `inflate_fallback(rgba)`, `clean_mesh(mesh)` |
| `pipeline/texture.py` | `bake_uv_texture(mesh, tex_size, max_faces)` |
| `pipeline/export.py` | `validate_mesh(mesh)`, `export_mesh(mesh, stem)`, `MeshValidationError` |
| `pipeline/preview.py` | `save_html_viewer(...)`, `save_multiview_png(...)`, `save_preview_files(...)` |
| `pipeline/pipeline.py` | `Sketch2ShapePipeline.run(sketch, prompt, style, seed, steps, n_candidates, refine, texture, use_fallback, progress)`, `.warmup()` |

**Using the pipeline from Python**
```python
from PIL import Image
from pipeline import Sketch2ShapePipeline

pipe = Sketch2ShapePipeline()
result = pipe.run(Image.open("examples/mug.png"), prompt="red ceramic coffee mug")
print(result["glb"])
print(result["report"]["timings_sec"])
```

---

## 15. Deployment options

| Option | Suitability |
|---|---|
| **Google Colab with `--share`** | Best for the hackathon demo. Free T4 GPU, public link while the notebook runs. Links expire and the session can disconnect. |
| **Local GPU machine** | Most reliable backup: `run.bat app`. |
| **Hugging Face Spaces (GPU)** | A permanent public link. GPU hardware needs a paid plan or a grant (check current pricing). |
| **Rented GPU server** | Always-on hosting, paid by the hour. |
| Vercel, Streamlit Community Cloud | Not suitable: no GPU, small memory, and size and time limits. |

A public website is not required by the problem statement. A working live demonstration and a reproducible repository are.

---

## 16. Limitations

- A single view cannot specify hidden surfaces. The back of the object is plausible, not exact.
- Thin structures, very abstract doodles, and multi-object scenes may fail or drift from the sketch.
- The texture is baked from TripoSR's colours (one colour per triangle at 1024 x 1024). It is not a repainted PBR material.
- CLIP and CNN labels are hints and can be wrong. A user prompt always takes priority.
- The mesh is a good starting point for refinement, not production topology and not 3D-print-ready.
- Quality depends on the drawing: simple, large, closed outlines work best.
- First-run time includes model loading and a large one-time download.
- Colab sessions are temporary: the public link works only while the session runs.
- Fallback mode is low fidelity and is meant only as an emergency option.

---

## 17. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Cannot uninstall numpy ... no RECORD file` | Installing into the base Anaconda environment. Use `run.bat setup` (creates `.venv`) or a fresh conda environment. |
| `torchmcubes must be built against an installed PyTorch` | Do not install `torchmcubes`. The repo ships a replacement. |
| `No NVIDIA GPU detected` | Install the CUDA build of PyTorch (`nvidia-smi` must work), use Colab with a GPU runtime, or enable fallback mode. |
| `No module named 'trimesh'` (or another package) | The install step did not finish. Re-run the install cell or `pip install -r requirements.txt`. |
| `Missing key(s) in state_dict ... image_tokenizer ... q_proj` when loading TripoSR | A newer `transformers` (5.x) uses different layer names than the TripoSR weights. Install `transformers>=4.46,<5` (or the pinned 4.44.2 locally) and restart. |
| `cannot import name 'resolve_revision' from 'huggingface_hub'` | `huggingface_hub` and `diffusers` versions do not match. Install `huggingface_hub>=0.34,<1.0` with `diffusers==0.35.2`, restart the session. |
| Red pip warnings about `google-genai`, `langsmith`, `yfinance`, `websockets`, `pydantic` | Conflicts with packages Colab pre-installs. They do not affect this project. Ignore them unless the app later crashes mentioning these packages. |
| `No module named 'audioop'` on Python 3.13 | `pip install audioop-lts`. |
| `TripoSR source not found` | `git clone https://github.com/VAST-AI-Research/TripoSR.git models/TripoSR` |
| CUDA out of memory | Lower diffusion steps and candidates, close other GPU programs, or restart the session. |
| The app cell never finishes on Colab | Expected. The cell keeps running while the app is live. |
| 3D viewer is blank in the app | Open `outputs/<run>/preview.html`, or load `model.glb` at https://gltf-viewer.donmccurdy.com. |
| Model looks mirrored or scrambled | Marching-cubes axis order. Open an issue with a screenshot; the axis order can be flipped in `install_mcubes_stub`. |
| Result ignores the sketch | Raise `control_scale` in `pipeline/generate.py` (try 1.2) or simplify the prompt. |
| "Texture bake failed" in `report.json` | The model still exports with vertex colours. Check that `xatlas` is installed. |

---

## 18. Mapping to the evaluation rubric

| Criterion (points) | Where it is addressed |
|---|---|
| Technical implementation (25) | Modular `pipeline/` package, preprocessing, candidate ranking, error handling (empty sketch, validation, out-of-memory), unit tests |
| 3D output quality and design (25) | ControlNet fidelity plus candidate ranking, TripoSR geometry, mesh cleanup, UV texture, preview |
| End-to-end pipeline validity and efficiency (20) | Draw or upload through export in one run, per-stage timings in `report.json`, fast settings and low-VRAM mode |
| Code reproducibility (15) | Pinned `requirements.txt`, `run.bat` and `run.ps1`, Colab launcher, weight cache in `models/`, seed control |
| Documentation and README (15) | `README.md` (quick start) and this document (architecture, setup, models, runtime, limitations, troubleshooting) |

---

## 19. Future work

- Multi-view diffusion plus a multi-view reconstruction model (for example Zero123++ with InstantMesh) for better hidden geometry.
- Repainted PBR textures (roughness and metallic) instead of baked vertex colours.
- Retopology and mesh simplification for animation and game use.
- User-selectable candidates and region-based sketch editing in the UI.
- A permanent GPU-backed deployment.

---

## 20. License and acknowledgements

The code in this repository is released under the **MIT License** (see `LICENSE`). The pretrained models keep their own licenses, listed in section 9.

Thanks to TripoSR (Stability AI and Tripo AI), Stable Diffusion (CompVis, Stability AI, Runway), ControlNet (L. Zhang et al.), rembg (D. Gatis), CLIP (OpenAI), Google QuickDraw, Hugging Face Diffusers and Transformers, trimesh, and xatlas.
