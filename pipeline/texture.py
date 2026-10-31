"""REQ 8 - Texture generation: bake per-vertex colours into a real UV texture map (xatlas)."""
import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree


def bake_uv_texture(mesh: trimesh.Trimesh, tex_size: int = 1024, max_faces: int = 60000):
    """Returns (textured trimesh, texture PIL image). Raises on failure; caller keeps vertex colours then."""
    import xatlas
    colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32)
    if len(mesh.faces) > max_faces:                              # xatlas is slow on huge meshes
        try:
            small = mesh.simplify_quadric_decimation(face_count=max_faces)
        except TypeError:
            small = mesh.simplify_quadric_decimation(max_faces)
        nn = cKDTree(mesh.vertices).query(small.vertices)[1]     # re-transfer colours to decimated vertices
        mesh = trimesh.Trimesh(small.vertices, small.faces, process=False)
        colors = colors[nn]
    vmap, idx, uvs = xatlas.parametrize(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    new_v, new_c = mesh.vertices[vmap], colors[vmap]
    px = uvs * (tex_size - 1)
    px[:, 1] = (tex_size - 1) - px[:, 1]                         # v points up, image rows point down
    tex = np.zeros((tex_size, tex_size, 3), np.uint8)
    filled = np.zeros((tex_size, tex_size), np.uint8)
    face_col = new_c[idx].mean(axis=1)
    for f, col in zip(idx, face_col):
        pts = px[f].round().astype(np.int32)
        cv2.fillConvexPoly(tex, pts, tuple(int(c) for c in col))
        cv2.fillConvexPoly(filled, pts, 255)
    tex = cv2.inpaint(tex, (filled == 0).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)   # pad seams
    img = Image.fromarray(tex)
    textured = trimesh.Trimesh(vertices=new_v, faces=idx.astype(np.int64), process=False,
                               visual=trimesh.visual.TextureVisuals(uv=uvs, image=img))
    return textured, img
