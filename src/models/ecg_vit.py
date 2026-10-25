"""ECG branch architecture: ViT-B/16 fine-tuned on PTB-XL STFT spectrograms.

Trained on Colab (`notebooks/step5_ecg_vit_colab.ipynb`); this module exists
locally so the saved weights (`checkpoints/ecg_vit.pt`) can be loaded for
local inference and explainability (Grad-CAM) without needing a GPU.
"""
from torch import nn
from transformers import ViTModel

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


class ECGViT(nn.Module):
    def __init__(self, n_classes: int = 5, d_embed: int = 256, freeze_below_layer: int = 8, total_layers: int = 12):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        for name, p in self.vit.named_parameters():
            p.requires_grad = any(f"layers.{i}." in name for i in range(freeze_below_layer, total_layers))

        self.project = nn.Sequential(nn.Linear(768, d_embed), nn.LayerNorm(d_embed))
        self.classifier = nn.Linear(d_embed, n_classes)

    def embed(self, x):
        cls_token = self.vit(pixel_values=x).last_hidden_state[:, 0]
        return self.project(cls_token)

    def forward(self, x):
        return self.classifier(self.embed(x))
