"""Image preprocessing shared by training notebooks and the app.

Keeping one copy of each conversion matters for the same reason it did for
the ECG spectrogram (see gradcam_ecg.py): the app must feed a model exactly
what it was trained on, or predictions silently degrade.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

VIT_SIZE = 224


def _vit_normalise(x: np.ndarray) -> np.ndarray:
    # google/vit-base-patch16-224-in21k's preprocessor: scale to [0,1], then mean 0.5 / std 0.5
    return (x.astype(np.float32) / 255.0 - 0.5) / 0.5


# ---------------------------------------------------------------- X-ray
def xray_to_tensor(image: Image.Image) -> torch.Tensor:
    """Any chest X-ray image (PNG/JPG, grey or colour) -> (3, 224, 224)."""
    grey = image.convert("L").resize((VIT_SIZE, VIT_SIZE), Image.BILINEAR)
    arr = np.stack([np.array(grey)] * 3, axis=-1)
    return torch.from_numpy(_vit_normalise(arr)).permute(2, 0, 1)


# ---------------------------------------------------------------- Echo
def read_video_frames(path: str | Path) -> np.ndarray:
    """All frames of a video file as (frames, H, W) uint8 grey."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise ValueError(f"No frames could be read from {path}")
    return np.stack(frames)


def read_video_bytes(data: bytes, suffix: str = ".avi") -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        return read_video_frames(path)


def sample_clip(frames: np.ndarray, n_frames: int, stride: int = 2, start: int | None = None) -> np.ndarray:
    """n_frames frames, `stride` apart, starting at `start` (default: centred).
    Short videos are looped, as in the EchoNet-Dynamic reference code."""
    span = n_frames * stride
    if start is None:
        start = max(0, (len(frames) - span) // 2)
    idx = (start + np.arange(n_frames) * stride) % len(frames)
    return frames[idx]


def clip_to_tensor(clip: np.ndarray) -> torch.Tensor:
    """(frames, H, W) uint8 -> (frames, 3, 224, 224) normalised for the ViT."""
    t = torch.from_numpy(clip.astype(np.float32)).unsqueeze(1)
    t = F.interpolate(t, size=(VIT_SIZE, VIT_SIZE), mode="bilinear", align_corners=False)
    t = (t / 255.0 - 0.5) / 0.5
    return t.expand(-1, 3, -1, -1).contiguous()


# ---------------------------------------------------------------- MRI
# (CT has its own resampling pipeline in src/imaging/ct.py)
def normalise_volume(volume: np.ndarray, lo_pct: float = 1, hi_pct: float = 99) -> np.ndarray:
    """Percentile clip then scale to [0, 1] -- MRI intensities have no fixed unit."""
    lo, hi = np.percentile(volume, [lo_pct, hi_pct])
    return np.clip((volume.astype(np.float32) - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def slices_to_tensor(volume01: np.ndarray, size: int = VIT_SIZE) -> torch.Tensor:
    """(H, W, slices) in [0,1] -> (slices, 1, size, size)."""
    t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(volume01, -1, 0))).unsqueeze(1)
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)


@torch.no_grad()
def segment_mri_volume(model: torch.nn.Module, volume: np.ndarray, batch: int = 8) -> np.ndarray:
    """Short-axis MRI volume (H, W, slices) -> label mask of the same shape."""
    h, w, _ = volume.shape
    x = slices_to_tensor(normalise_volume(volume))
    labels = []
    for i in range(0, len(x), batch):
        logits = model(x[i : i + batch])
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        labels.append(logits.argmax(1))
    return np.moveaxis(torch.cat(labels).numpy().astype(np.uint8), 0, -1)


def mask_slices_to_tensor(mask: np.ndarray, size: int = VIT_SIZE) -> torch.Tensor:
    """(H, W, slices) integer labels -> (slices, size, size), nearest-neighbour so labels stay whole."""
    t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(mask, -1, 0))).unsqueeze(1).float()
    return F.interpolate(t, size=(size, size), mode="nearest")[:, 0].long()
