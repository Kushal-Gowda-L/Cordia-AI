"""MRI and CT branches: Swin-UNETR segmentation (transformer encoder + U-Net decoder).

A plain ViT gives one label per image; the digital twin needs the outline of
each heart chamber, pixel by pixel. Swin-UNETR (Hatamizadeh et al., 2022)
keeps a transformer as the encoder and adds a convolutional decoder that
produces a full-resolution mask, so it is still a vision-transformer model.
Implementation from MONAI.

- MRI (ACDC): 2D, one short-axis slice at a time. ACDC slices are 5-10 mm
  apart, so per-slice 2D segmentation is the standard choice. Labels follow
  ACDC: 1 = RV cavity, 2 = LV myocardium, 3 = LV cavity.
  Trained in `notebooks/step17_mri_acdc_colab.ipynb`.
- CT (ImageCAS): 3D patches, binary coronary artery mask.
  Trained in `notebooks/step18_ct_imagecas_colab.ipynb`.
"""
from __future__ import annotations

import inspect

from monai.networks.nets import SwinUNETR

ACDC_LABELS = {0: "background", 1: "RV cavity", 2: "LV myocardium", 3: "LV cavity"}
ACDC_GROUPS = ["NOR", "MINF", "DCM", "HCM", "RV"]
ACDC_GROUP_NAMES = {
    "NOR": "Normal",
    "MINF": "Previous myocardial infarction",
    "DCM": "Dilated cardiomyopathy",
    "HCM": "Hypertrophic cardiomyopathy",
    "RV": "Abnormal right ventricle",
}

MRI_SLICE_SIZE = 224  # divisible by 32, as Swin-UNETR requires
CT_PATCH_SIZE = (96, 96, 96)


def _swin_unetr(img_size, **kwargs) -> SwinUNETR:
    # MONAI deprecated and then removed the img_size argument; pass it only
    # to versions that still take it.
    if "img_size" in inspect.signature(SwinUNETR.__init__).parameters:
        kwargs["img_size"] = img_size
    return SwinUNETR(**kwargs)


def build_mri_segmenter(feature_size: int = 24) -> SwinUNETR:
    return _swin_unetr(
        (MRI_SLICE_SIZE, MRI_SLICE_SIZE), in_channels=1, out_channels=len(ACDC_LABELS),
        feature_size=feature_size, spatial_dims=2,
    )


def build_ct_segmenter(feature_size: int = 48, use_checkpoint: bool = True) -> SwinUNETR:
    return _swin_unetr(
        CT_PATCH_SIZE, in_channels=1, out_channels=2, feature_size=feature_size,
        spatial_dims=3, use_checkpoint=use_checkpoint,
    )
