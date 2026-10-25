"""A compact Gaussian-diffusion generator for tabular data, standing in for
TabDDPM.

The published TabDDPM (Kotelnikov et al.) uses multinomial diffusion for
categorical columns and Gaussian diffusion for numeric ones, implemented in
a research codebase that isn't pip-installable. Since our SF-2 features
arrive already standardized to continuous z-scores, a pure Gaussian
diffusion process over that space captures the same core idea — learning to
denoise a corrupted feature vector back to the real data distribution — at a
fraction of the implementation cost. This is a deliberate simplification,
not a claim of reproducing TabDDPM exactly.
"""
from __future__ import annotations

import pandas as pd
import torch
from torch import nn


class _Denoiser(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128):
        super().__init__()
        self.time_embed = nn.Embedding(1000, hidden)
        self.net = nn.Sequential(
            nn.Linear(n_features + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=1))


class TabularDiffusion:
    def __init__(
        self,
        n_features: int,
        timesteps: int = 200,
        random_state: int = 42,
        device: str | None = None,
    ):
        torch.manual_seed(random_state)
        self.timesteps = timesteps
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _Denoiser(n_features).to(self.device)

        self.betas = torch.linspace(1e-4, 0.02, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def fit(self, X: pd.DataFrame, epochs: int = 300, lr: float = 1e-3) -> "TabularDiffusion":
        x0 = torch.tensor(X.values, dtype=torch.float32, device=self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        for _ in range(epochs):
            t = torch.randint(0, self.timesteps, (len(x0),), device=self.device)
            noise = torch.randn_like(x0)
            alpha_bar_t = self.alpha_bars[t].unsqueeze(1)
            x_t = alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise

            predicted_noise = self.model(x_t, t)
            loss = nn.functional.mse_loss(predicted_noise, noise)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        return self

    @torch.no_grad()
    def sample(self, n: int, n_features: int) -> torch.Tensor:
        x = torch.randn(n, n_features, device=self.device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((n,), t, dtype=torch.long, device=self.device)
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            beta_t = self.betas[t]
            predicted_noise = self.model(x, t_batch)

            mean = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * predicted_noise)
            if t > 0:
                x = mean + beta_t.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.cpu()


def diffusion_augment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_counts: dict[int, int],
    epochs: int = 300,
    random_state: int = 42,
    device: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    synthetic_X_parts, synthetic_y_parts = [], []
    n_features = X_train.shape[1]

    for cls, target in target_counts.items():
        class_rows = X_train[y_train == cls]
        deficit = target - len(class_rows)
        if deficit <= 0:
            continue

        diffusion = TabularDiffusion(
            n_features, random_state=random_state, device=device
        ).fit(class_rows, epochs=epochs)
        samples = diffusion.sample(deficit, n_features).numpy()

        synthetic_X_parts.append(pd.DataFrame(samples, columns=X_train.columns))
        synthetic_y_parts.append(pd.Series([cls] * deficit))

    if not synthetic_X_parts:
        return X_train.iloc[0:0], y_train.iloc[0:0]

    X_synth = pd.concat(synthetic_X_parts, ignore_index=True)
    y_synth = pd.concat(synthetic_y_parts, ignore_index=True)
    return X_synth, y_synth
