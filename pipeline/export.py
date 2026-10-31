"""REQ 10 - Mesh validation and export (.glb + .obj), with a re-open check."""
from pathlib import Path

import numpy as np
import trimesh


class MeshValidationError(RuntimeError):
    pass


def validate_mesh(mesh: trimesh.Trimesh) -> bool:
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise MeshValidationError("Empty mesh (no vertices/faces).")
    ext = np.sort(mesh.extents)
    if ext[0] <= 1e-6 or ext[0] / ext[-1] < 0.03:
        raise MeshValidationError(f"Mesh is flat (extents={mesh.extents.tolist()}), not real 3D.")
    return True


def export_mesh(mesh: trimesh.Trimesh, stem: str) -> dict:
    validate_mesh(mesh)
    glb, obj = f"{stem}.glb", f"{stem}.obj"
    mesh.export(glb)
    mesh.export(obj)
    kinds = {}
    for p in (glb, obj):                      # prove both files re-open with geometry intact
        r = trimesh.load(p, force="mesh")
        if len(r.vertices) == 0 or len(r.faces) == 0:
            raise MeshValidationError(f"{p} re-opened empty.")
        kinds[Path(p).suffix] = r.visual.kind
    return {"glb": glb, "obj": obj, "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
            "extents": [round(float(e), 4) for e in mesh.extents], "watertight": bool(mesh.is_watertight),
            "visual_after_reload": kinds}
