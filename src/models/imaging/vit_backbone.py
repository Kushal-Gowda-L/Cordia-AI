"""Shared ViT-B/16 backbone for the imaging branches (X-ray, echo).

Same pretrained model the ECG branch uses (google/vit-base-patch16-224-in21k),
so every image-like branch in CordiaAI is a Vision Transformer.

Two details worth knowing:
- `pretrained=False` builds the same architecture with random weights. Local
  inference uses this, since the trained checkpoint overwrites every weight
  anyway -- it avoids downloading ~350MB of ImageNet weights at app startup.
- Layer freezing matches parameter names with a regex that accepts both
  "encoder.layer.N." and "layers.N.", since transformers versions differ
  (see the note in notebooks/step5_ecg_vit_colab.ipynb).
"""
from __future__ import annotations

import re

from torch import nn
from transformers import ViTConfig, ViTModel

VIT_NAME = "google/vit-base-patch16-224-in21k"
_LAYER_RE = re.compile(r"\blayers?\.(\d+)\.")


def load_vit(pretrained: bool = True, config: ViTConfig | None = None) -> ViTModel:
    if config is not None:
        return ViTModel(config)
    return ViTModel.from_pretrained(VIT_NAME) if pretrained else ViTModel(ViTConfig())


def freeze_below(vit: ViTModel, train_from_layer: int) -> None:
    """Train only encoder layers >= train_from_layer (plus nothing else in the ViT)."""
    for name, p in vit.named_parameters():
        match = _LAYER_RE.search(name)
        p.requires_grad = bool(match) and int(match.group(1)) >= train_from_layer


class Projection(nn.Sequential):
    def __init__(self, d_in: int, d_embed: int):
        super().__init__(nn.Linear(d_in, d_embed), nn.LayerNorm(d_embed))
