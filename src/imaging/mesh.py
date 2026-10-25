"""3D meshes of heart structures from segmentation masks, for the digital twin.

Marching cubes turns each labelled structure (LV cavity, myocardium, RV,
coronary tree) into a triangle surface in millimetres. The result is sent
to heart_digital_twin_3d/ as plain JSON (vertex + face arrays a Three.js
BufferGeometry takes directly), or saved as .obj for any 3D viewer.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure


def mask_to_mesh(mask: np.ndarray, spacing_mm, step: int = 2, smooth_sigma: float = 1.0) -> dict | None:
    """Binary mask (H, W, D) -> {"vertices": [[x,y,z]...], "faces": [[i,j,k]...]} in mm."""
    if mask.sum() == 0:
        return None
    # Pad so surfaces touching the volume edge still close, then smooth the
    # blocky voxel edges (MRI slices are 5-10 mm thick).
    padded = np.pad(mask.astype(np.float32), 1)
    if smooth_sigma > 0:
        sigma = [smooth_sigma / s for s in spacing_mm]
        smoothed = ndimage.gaussian_filter(padded, sigma=sigma)
        if smoothed.max() > 0.5:  # a structure only a few voxels across can blur away entirely
            padded = smoothed
    try:
        verts, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=tuple(spacing_mm), step_size=step)
    except (RuntimeError, ValueError):  # too small or thin to form a surface
        return None
    verts -= np.array(spacing_mm)  # undo the 1-voxel pad
    return {"vertices": np.round(verts, 2).tolist(), "faces": faces.astype(int).tolist()}


def label_meshes(mask: np.ndarray, spacing_mm, labels: dict[int, str], **kwargs) -> dict[str, dict]:
    meshes = {}
    for value, name in labels.items():
        if value == 0:
            continue
        mesh = mask_to_mesh(mask == value, spacing_mm, **kwargs)
        if mesh is not None:
            meshes[name] = mesh
    return meshes


def mesh_to_obj(mesh: dict) -> str:
    lines = [f"v {x} {y} {z}" for x, y, z in mesh["vertices"]]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh["faces"]]
    return "\n".join(lines) + "\n"
