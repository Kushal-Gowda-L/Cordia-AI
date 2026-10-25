"""Coronary CT angiography pipeline (ImageCAS), shared by training and the app.

CT scans differ in voxel size from scanner to scanner, so every scan is
resampled to one isotropic spacing before the model sees it, and the
intensity window is fixed in Hounsfield units (contrast-filled coronary
arteries sit around 300-500 HU). The same transforms run in
notebooks/step18_ct_imagecas_colab.ipynb and in the app.
"""
from __future__ import annotations

import numpy as np
import torch
from monai import transforms as T

CT_SPACING = (0.8, 0.8, 0.8)  # mm
HU_MIN, HU_MAX = -200.0, 800.0
PATCH = (96, 96, 96)


def _base(keys, label: bool):
    modes = ("bilinear", "nearest") if label else ("bilinear",)
    return [
        T.LoadImaged(keys=keys),
        T.EnsureChannelFirstd(keys=keys),
        T.Orientationd(keys=keys, axcodes="RAS"),
        T.Spacingd(keys=keys, pixdim=CT_SPACING, mode=modes),
        T.ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
    ]


def train_transforms(samples_per_scan: int = 2):
    keys = ["image", "label"]
    return T.Compose(_base(keys, True) + [
        T.CropForegroundd(keys=keys, source_key="image"),
        # Coronary arteries are <1% of the volume: sample patches centred on vessel twice as often as background
        T.RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=PATCH, pos=2, neg=1,
                                 num_samples=samples_per_scan),
        T.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        T.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    ])


def val_transforms():
    return T.Compose(_base(["image", "label"], True))


def inference_transforms():
    return T.Compose(_base(["image"], False))


@torch.no_grad()
def segment_ct_file(model: torch.nn.Module, path: str) -> tuple[np.ndarray, tuple]:
    """NIfTI CT file -> (binary coronary mask at CT_SPACING, spacing)."""
    from monai.inferers import sliding_window_inference

    data = inference_transforms()({"image": path})
    x = data["image"].unsqueeze(0).float()
    logits = sliding_window_inference(x, PATCH, sw_batch_size=2, predictor=model, overlap=0.25)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8), CT_SPACING
