"""Unit tests that need no GPU and no model downloads.   python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.preprocess import EmptySketchError, preprocess_sketch      # noqa: E402
from pipeline.prompt import build_prompt                                  # noqa: E402
from pipeline.understand import shape_statistics                          # noqa: E402

MUG = Image.open(ROOT / "examples" / "mug.png")


def test_preprocess_output_format():
    p = preprocess_sketch(MUG)
    assert p["control"].size == (512, 512)
    arr = np.array(p["control"])
    assert arr.max() == 255 and arr.min() == 0
    assert (arr[..., 0] > 0).mean() < 0.4                       # mostly black background


def test_empty_sketch_rejected():
    with pytest.raises(EmptySketchError):
        preprocess_sketch(Image.new("RGB", (300, 300), "white"))


def test_dark_paper_gives_same_result():
    inverted = Image.fromarray(255 - np.array(MUG.convert("RGB")))
    a, b = preprocess_sketch(MUG)["control"], preprocess_sketch(inverted)["control"]
    assert np.mean(np.array(a) == np.array(b)) > 0.9


@pytest.mark.parametrize("name", ["mug", "chair", "car", "house", "lamp", "bottle"])
def test_all_examples_preprocess(name):
    p = preprocess_sketch(Image.open(ROOT / "examples" / f"{name}.png"))
    stats, _ = shape_statistics(p["control"])
    assert stats["n_contours"] >= 1 and 0 < stats["stroke_density"] < 0.5


def test_closed_sketch_gives_silhouette():
    stats, sil = shape_statistics(preprocess_sketch(MUG)["control"])
    assert stats["silhouette_closed"] and sil is not None and sil.shape == (512, 512)


def test_prompt_is_optional_and_prioritised():
    und = {"label_top3": [("mug", 0.6)], "classifier_guess": None}
    assert build_prompt("", und)[1].startswith("auto")
    assert build_prompt("red teapot", und)[1] == "user"
    assert build_prompt("", {"label_top3": [("x", 0.01)], "classifier_guess": None})[1] == "default"


def test_candidate_score_prefers_matching_image():
    from pipeline.generate import candidate_score
    pre = preprocess_sketch(MUG)
    _, sil = shape_statistics(pre["control"])
    good = pre["clean"]                                          # same strokes as the sketch
    blank = Image.new("RGB", (512, 512), "white")
    rgba_good = good.convert("RGBA"); rgba_blank = blank.convert("RGBA")
    rgba_good.putalpha(Image.fromarray((sil * 255).astype(np.uint8)))
    rgba_blank.putalpha(Image.fromarray(np.zeros((512, 512), np.uint8)))
    assert candidate_score(pre["control"], sil, good, rgba_good) > candidate_score(pre["control"], sil, blank, rgba_blank)


# ---- tests below need trimesh / xatlas / torch; skipped automatically when missing ----
def test_flat_mesh_rejected_and_box_exports(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from pipeline.export import MeshValidationError, export_mesh, validate_mesh
    flat = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    with pytest.raises(MeshValidationError):
        validate_mesh(flat)
    info = export_mesh(trimesh.creation.box(), str(tmp_path / "box"))
    assert info["faces"] > 0 and Path(info["glb"]).exists() and Path(info["obj"]).exists()


def test_fallback_mesh_is_real_3d():
    pytest.importorskip("trimesh")
    from pipeline.background import prepare_foreground
    from pipeline.reconstruct import clean_mesh, inflate_fallback
    rgba = MUG.convert("RGBA")
    a = np.array(rgba)
    mask = (a[..., :3].min(axis=2) < 200)
    from scipy import ndimage
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=3))
    a[..., 3] = mask * 255
    mesh = clean_mesh(inflate_fallback(prepare_foreground(Image.fromarray(a))))
    assert len(mesh.faces) > 100 and np.sort(mesh.extents)[0] > 0.05


def test_texture_bake_on_sphere():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("xatlas")
    from pipeline.texture import bake_uv_texture
    s = trimesh.creation.icosphere(subdivisions=3)
    s.visual.vertex_colors = np.tile([200, 40, 40, 255], (len(s.vertices), 1))
    m, img = bake_uv_texture(s, tex_size=256)
    assert m.visual.kind == "texture" and img.size == (256, 256)


def test_classifier_forward_shape():
    torch = pytest.importorskip("torch")
    from pipeline.classifier import SketchCNN, control_to_28
    assert SketchCNN(7)(torch.zeros(2, 1, 28, 28)).shape == (2, 7)
    assert control_to_28(preprocess_sketch(MUG)["control"]).shape == (28, 28)
