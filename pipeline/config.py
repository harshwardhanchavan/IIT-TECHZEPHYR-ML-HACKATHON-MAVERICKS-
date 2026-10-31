"""Paths, model ids and prompt constants. Imported first so weights are cached inside ./models."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
EXAMPLES_DIR = ROOT / "examples"
DATA_DIR = ROOT / "data"
TRIPOSR_DIR = MODELS_DIR / "TripoSR"            # cloned by run.bat / run.ps1 setup
CLASSIFIER_PATH = MODELS_DIR / "sketch_cnn.pt"  # produced by train.py (optional)

# Keep all Hugging Face weights inside the project folder (reproducible, easy to delete).
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
for _d in (MODELS_DIR, OUTPUT_DIR, EXAMPLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CN_ID = "lllyasviel/sd-controlnet-scribble"
TRIPOSR_ID = "stabilityai/TripoSR"
CLIP_ID = "openai/clip-vit-base-patch32"

DEFAULT_PROMPT = "a single object, product design render"
STYLE_PRESETS = {
    "none": "",
    "realistic": "photorealistic materials, studio product photo",
    "toy / cartoon": "cute stylized toy, smooth plastic, bright colors",
    "low-poly": "low poly game asset, flat shaded, simple colors",
    "metallic": "brushed metal, reflective surface",
    "wood": "natural wood grain, matte finish",
    "ceramic": "glossy ceramic glaze",
}
SUFFIX = (", centered, full object visible, 3/4 view, studio lighting, soft shadows, "
          "plain white background, highly detailed")
NEGATIVE = ("multiple objects, cropped, out of frame, text, watermark, people, hands, "
            "busy background, blurry, low quality, deformed")

LABELS = ["chair", "table", "sofa", "bed", "lamp", "mug", "cup", "bottle", "vase", "bowl", "teapot", "kettle",
          "car", "truck", "bus", "bicycle", "motorcycle", "airplane", "helicopter", "boat", "train", "house",
          "castle", "tower", "bridge", "tree", "flower", "cat", "dog", "horse", "elephant", "bird", "fish",
          "rabbit", "dinosaur", "robot", "toy", "teddy bear", "guitar", "piano", "violin", "clock", "phone",
          "laptop", "television", "camera", "headphones", "backpack", "shoe", "hat", "umbrella", "key", "hammer",
          "scissors", "knife", "fork", "spoon", "axe", "sword", "shield", "rocket", "spaceship", "lantern",
          "candle", "book", "cube", "sphere", "pencil", "lightbulb", "fan", "cabinet", "bench", "stool",
          "wardrobe", "bathtub", "sink", "door", "window", "ladder", "crown", "trophy", "watch", "ring",
          "glasses", "drone", "tractor", "tank", "hot air balloon", "pumpkin", "apple", "banana", "pizza",
          "cake", "ice cream", "cactus", "mushroom"]


def hardware_info() -> dict:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return {"device": "cuda", "gpu": p.name, "vram_gb": round(p.total_memory / 1e9, 1)}
    return {"device": "cpu", "gpu": "CPU", "vram_gb": 0}
