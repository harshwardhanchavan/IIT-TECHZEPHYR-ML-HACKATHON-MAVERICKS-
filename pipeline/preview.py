"""REQ 9 - 3D preview files: interactive HTML viewer + static multi-view PNG (both saved next to the model)."""
import base64
from pathlib import Path

import numpy as np
import trimesh


def save_html_viewer(glb_path: str, out_html: str) -> str:
    """Standalone page using Google <model-viewer> (needs internet in the browser). Open by double-click."""
    b64 = base64.b64encode(open(glb_path, "rb").read()).decode()
    html = ("<!doctype html><meta charset='utf-8'><title>Sketch2Shape preview</title>"
            "<script type='module' src='https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/"
            "model-viewer.min.js'></script>"
            f"<model-viewer src='data:model/gltf-binary;base64,{b64}' camera-controls auto-rotate "
            "shadow-intensity='1' style='width:100vw;height:100vh;background:#f3f3f3'></model-viewer>")
    Path(out_html).write_text(html, encoding="utf-8")
    return out_html


def save_multiview_png(glb_path: str, out_png: str, n_points: int = 25000) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = trimesh.load(glb_path, force="mesh")
    pts, fidx = trimesh.sample.sample_surface(m, n_points)
    try:
        col = m.visual.to_color().vertex_colors[m.faces[fidx][:, 0]][:, :3] / 255.0
    except Exception:
        col = np.full((n_points, 3), 0.6)
    fig = plt.figure(figsize=(12, 3.2))
    for i, az in enumerate([0, 90, 180, 270]):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=col, s=1, depthshade=False)
        ax.view_init(elev=15, azim=az)
        ax.set_axis_off()
        r = 0.55
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_png


def save_preview_files(glb_path: str, run_dir) -> dict:
    run_dir = Path(run_dir)
    out = {}
    try:
        out["html"] = save_html_viewer(glb_path, str(run_dir / "preview.html"))
        out["png"] = save_multiview_png(glb_path, str(run_dir / "preview.png"))
    except Exception as e:                      # preview is a convenience; never fail the run because of it
        out["error"] = str(e)[:100]
    return out
