"""FT-Transformer (Feature Tokenizer Transformer) for the SF-2 tabular
branch, matching the methodology spec: depth=3, heads=8, d=256, dropout=0.2.

Built directly in PyTorch rather than via the `rtdl` package — its installer
fails on this machine's non-ASCII OneDrive path, and a from-scratch
implementation keeps the architecture exactly matched to spec and dependency
risk low, consistent with the diffusion generator in Step 3.

Numerical feature tokenization follows Gorishniy et al. (2021): each scalar
feature x_i is projected to a d-dimensional token via its own learned
weight vector and bias, i.e. token_i = x_i * W_i + b_i. A learnable [CLS]
token is prepended; its representation after the transformer encoder is the
patient embedding E_t used both for classification here and for fusion with
other branches later.
"""
from __future__ import annotations

import torch
from torch import nn


class FeatureTokenizer(nn.Module):
    """Projects each of the n numerical features into its own d-dim token."""

    def __init__(self, n_features: int, d_token: int = 256):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.uniform_(self.weight, -1 / n_features**0.5, 1 / n_features**0.5)
        nn.init.uniform_(self.bias, -1 / n_features**0.5, 1 / n_features**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> tokens: (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_token: int = 256,
        depth: int = 3,
        heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_token)
        self.classifier = nn.Linear(d_token, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Returns E_t, the CLS token's representation after the encoder."""
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(sequence)
        return self.norm(encoded[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.embed(x)
        return self.classifier(embedding)
