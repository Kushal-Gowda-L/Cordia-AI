"""Echocardiogram branch: ejection fraction from an echo video.

Each frame goes through the shared ViT-B/16 and becomes one 256-dim token;
a small temporal transformer then reads the sequence of frame tokens (the
heart contracting and relaxing) and regresses ejection fraction (EF, %).
This "image transformer per frame + transformer over time" design is the
factorised form of a video transformer, and keeps memory low enough for a
free Colab T4 by freezing the lower ViT layers.

Trained on EchoNet-Dynamic in `notebooks/step16_echo_ef_colab.ipynb`.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import ViTConfig

from src.models.imaging.vit_backbone import Projection, freeze_below, load_vit

N_FRAMES = 16  # frames sampled per clip; EchoNet-Dynamic clips are ~50 fps, so 16 frames at stride 2 ~ one beat


class EchoViT(nn.Module):
    def __init__(
        self,
        n_frames: int = N_FRAMES,
        d_embed: int = 256,
        train_from_layer: int = 10,
        temporal_depth: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
        pretrained: bool = True,
        config: ViTConfig | None = None,
    ):
        super().__init__()
        self.vit = load_vit(pretrained, config)
        freeze_below(self.vit, train_from_layer)
        self.frame_project = Projection(self.vit.config.hidden_size, d_embed)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_embed))
        self.time_embed = nn.Parameter(torch.zeros(1, n_frames + 1, d_embed))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_embed, nhead=heads, dim_feedforward=d_embed * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_depth)
        self.norm = nn.LayerNorm(d_embed)
        self.ef_head = nn.Linear(d_embed, 1)

    def embed(self, clips: torch.Tensor) -> torch.Tensor:
        """clips: (batch, frames, 3, 224, 224) -> E_e, the 256-dim echo embedding."""
        b, t = clips.shape[:2]
        frames = self.vit(pixel_values=clips.flatten(0, 1)).last_hidden_state[:, 0]
        tokens = self.frame_project(frames).view(b, t, -1)
        seq = torch.cat([self.cls_token.expand(b, -1, -1), tokens], dim=1) + self.time_embed[:, : t + 1]
        return self.norm(self.temporal(seq)[:, 0])

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Returns predicted EF in percent, shape (batch,)."""
        return self.ef_head(self.embed(clips)).squeeze(-1)
