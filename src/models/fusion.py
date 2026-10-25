"""Step 6: multimodal fusion — cross-attention over E_t (tabular) and E_v (ECG),
followed by a stacking MLP meta-learner ensemble (GAN-Bagging), per the
CordiaAI methodology (Section 6.2, items 4 and 5).
"""
import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    """E_t (tabular) as Query, E_v (ECG) as Key+Value, d_model=256, 8 heads."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, e_t: torch.Tensor, e_v: torch.Tensor) -> torch.Tensor:
        query = e_t.unsqueeze(1)
        key_value = e_v.unsqueeze(1)
        fused, _ = self.attn(query, key_value, key_value)
        return self.norm(fused.squeeze(1) + e_t)


class StackingMetaLearner(nn.Module):
    """A single MLP head. The ensemble trains several of these on bootstrap
    samples of the fused embedding and averages their softmax outputs."""

    def __init__(self, d_model: int = 256, n_classes: int = 5, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionEnsemble(nn.Module):
    """CrossAttentionFusion + N stacking MLP meta-learners, averaged."""

    def __init__(
        self,
        d_model: int = 256,
        n_classes: int = 5,
        n_heads: int = 8,
        n_learners: int = 5,
        hidden: int = 128,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.3,
    ):
        super().__init__()
        self.fusion = CrossAttentionFusion(d_model, n_heads, attn_dropout)
        self.learners = nn.ModuleList(
            [StackingMetaLearner(d_model, n_classes, hidden, mlp_dropout) for _ in range(n_learners)]
        )

    def forward(self, e_t: torch.Tensor, e_v: torch.Tensor):
        fused = self.fusion(e_t, e_v)
        logits = torch.stack([learner(fused) for learner in self.learners], dim=0)
        probs = torch.softmax(logits, dim=-1).mean(dim=0)
        return probs, fused
