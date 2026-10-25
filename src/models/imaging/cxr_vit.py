"""Chest X-ray branch: ViT-B/16 fine-tuned for heart-related findings.

Trained on NIH ChestX-ray14 (or CheXpert) in
`notebooks/step15_xray_vit_kaggle.ipynb`. Multi-label: an X-ray can show
several findings at once, so each finding has its own sigmoid output rather
than one softmax over classes.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import ViTConfig

from src.models.imaging.vit_backbone import Projection, freeze_below, load_vit

# The three ChestX-ray14 / CheXpert findings that are about the heart:
# an enlarged heart, fluid in the lungs (often from heart failure) and fluid
# around the lungs.
CXR_FINDINGS = ["Cardiomegaly", "Edema", "Effusion"]


class CXRViT(nn.Module):
    def __init__(
        self,
        n_findings: int = len(CXR_FINDINGS),
        d_embed: int = 256,
        train_from_layer: int = 8,
        pretrained: bool = True,
        config: ViTConfig | None = None,
    ):
        super().__init__()
        self.vit = load_vit(pretrained, config)
        freeze_below(self.vit, train_from_layer)
        self.project = Projection(self.vit.config.hidden_size, d_embed)
        self.classifier = nn.Linear(d_embed, n_findings)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3, 224, 224) -> E_x, the 256-dim X-ray embedding."""
        return self.project(self.vit(pixel_values=x).last_hidden_state[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns one logit per finding; apply sigmoid for probabilities."""
        return self.classifier(self.embed(x))
