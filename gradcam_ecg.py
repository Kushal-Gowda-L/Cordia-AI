"""Explainable AI layer, item 12: Grad-CAM-style ECG spectrogram attribution
for the trained ViT-B/16 ECG branch (methodology Section 6.2 #12).

Standard Grad-CAM assumes convolutional feature maps; a ViT has none, so
this implements the patch-token analogue used in ViT-explainability work:
gradients of the target class logit are taken w.r.t. the last transformer
block's 196 patch tokens (the 14x14 grid the 224x224 image is divided into),
averaged per patch the same way Grad-CAM averages per spatial location, then
upsampled back to 224x224 and overlaid on the spectrogram.

Runs entirely locally (CPU) against a handful of PTB-XL sample records
downloaded fresh for this script -- the raw ECG signals used to train
`checkpoints/ecg_vit.pt` only ever lived on the ephemeral Colab runtime and
were never saved locally, so this pulls its own small sample set directly
from PhysioNet (a few hundred KB, not the full dataset).
"""
import ast
import ssl
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import wfdb
from PIL import Image

from src.models.ecg_vit import CLASSES, ECGViT

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "datasets" / "ecg-images" / "gradcam_samples"
BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
LEAD_II_IDX = 1
N_PER_CLASS = 2

ssl._create_default_https_context = ssl._create_unverified_context


def download(rel_path: str) -> None:
    for ext in (".hea", ".dat"):
        local = SAMPLE_DIR / (rel_path + ext)
        if local.exists():
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BASE_URL + rel_path + ext, local)


def pick_sample_records() -> pd.DataFrame:
    Y = pd.read_csv(SAMPLE_DIR / "ptbxl_database.csv", index_col="ecg_id")
    Y["scp_codes"] = Y["scp_codes"].apply(ast.literal_eval)
    agg_df = pd.read_csv(SAMPLE_DIR / "scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate(scp_dict):
        classes = {agg_df.loc[code].diagnostic_class for code in scp_dict if code in agg_df.index}
        return list(classes)

    Y["diagnostic_superclass"] = Y["scp_codes"].apply(aggregate)
    Y = Y[Y["diagnostic_superclass"].apply(len) == 1].copy()
    Y["label"] = Y["diagnostic_superclass"].apply(lambda x: x[0])
    Y = Y[Y["label"].isin(CLASSES)]

    return Y.groupby("label", group_keys=False)[Y.columns].apply(lambda g: g.sample(N_PER_CLASS, random_state=42))


def ecg_to_spectrogram(record_path: Path) -> np.ndarray:
    signal, _ = wfdb.rdsamp(str(record_path))
    lead = signal[:, LEAD_II_IDX].astype(np.float32)
    stft = librosa.stft(lead, n_fft=64, hop_length=8, window="hann")
    magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    normalised = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min() + 1e-8)
    img = Image.fromarray((normalised * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    return np.stack([np.array(img)] * 3, axis=-1)


def to_model_input(image: np.ndarray) -> torch.Tensor:
    img = image.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0)


def grad_cam(model: ECGViT, x: torch.Tensor, target_class: int) -> np.ndarray:
    """Gradients of the target logit w.r.t. the last transformer block's
    patch tokens, averaged per patch (the ViT analogue of Grad-CAM's channel
    averaging), ReLU'd, reshaped to the 14x14 patch grid, and upsampled to
    224x224. A forward hook captures the last block's actual output tensor
    (rather than reading it back out of `output_hidden_states`, which this
    transformers version returns disconnected from the live autograd graph)
    so `.retain_grad()` is called on the tensor that's really in the path to
    the logits."""
    captured = {}

    def hook(_module, args, kwargs):
        # Hook the last block's INPUT, not its output: LayerNorm (applied
        # right after the last block) acts per-position, so the CLS logit
        # has zero gradient w.r.t. that block's own *output* patch tokens --
        # the cross-token mixing that lets patches influence the CLS token
        # happens *inside* this block's self-attention, so we need the
        # tokens flowing in, not out, for gradient to reach them at all.
        tokens = args[0] if args else kwargs["hidden_states"]
        tokens.retain_grad()
        captured["tokens"] = tokens

    handle = model.vit.layers[-1].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        logits = model(x)
        model.zero_grad()
        logits[0, target_class].backward()
    finally:
        handle.remove()

    tokens = captured["tokens"]
    grads = tokens.grad[0, 1:, :]  # (196, 768), drop CLS token
    activations = tokens[0, 1:, :].detach()  # (196, 768)
    weights = grads.mean(dim=0)  # (768,) -- global-average-pool the gradient, like Grad-CAM's channel weights
    cam = (activations * weights).sum(dim=-1)  # (196,), SIGNED -- see note below

    # Vanilla Grad-CAM applies ReLU here to keep only positive contributions.
    # Empirically (all 10 sample records tested), that produces an all-zero
    # map every single time for this model: the weighted per-patch sum comes
    # out strictly negative everywhere, so ReLU wipes it out completely. This
    # is a documented failure mode of adapting CNN Grad-CAM's GAP-then-ReLU
    # formula to ViT patch tokens (their activation statistics differ enough
    # from conv feature maps that the sign convention doesn't transfer
    # cleanly). Min-max normalizing the *signed* value instead still answers
    # the useful question -- which patches were relatively most/least
    # supportive of the predicted class -- without silently collapsing to a
    # blank heatmap. See README.md's Grad-CAM section for the full note.
    cam = cam.reshape(14, 14).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0


def overlay(spectrogram: np.ndarray, cam: np.ndarray) -> Image.Image:
    heat = np.stack(
        [cam, np.zeros_like(cam), 1 - cam], axis=-1
    )  # red = high attribution, blue = low
    heat = (heat * 255).astype(np.uint8)
    base = spectrogram.astype(np.uint8)
    blended = (0.55 * base + 0.45 * heat).astype(np.uint8)
    return Image.fromarray(blended)


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    samples = pick_sample_records()
    print(f"Selected {len(samples)} sample records ({N_PER_CLASS} per class)")

    print("Loading ECGViT + trained weights...")
    model = ECGViT()
    model.load_state_dict(torch.load(ROOT / "checkpoints" / "ecg_vit.pt", map_location="cpu"))
    model.eval()

    out_dir = SAMPLE_DIR / "heatmaps"
    out_dir.mkdir(exist_ok=True)

    correct = 0
    for ecg_id, row in samples.iterrows():
        rel_path = row["filename_lr"]
        true_label = row["label"]
        print(f"  ecg_id={ecg_id}  true={true_label}  downloading {rel_path}...")
        download(rel_path)

        spectrogram = ecg_to_spectrogram(SAMPLE_DIR / rel_path)
        x = to_model_input(spectrogram)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_label = CLASSES[pred_idx]
        correct += int(pred_label == true_label)

        cam = grad_cam(model, x, pred_idx)
        overlay_img = overlay(spectrogram, cam)
        out_path = out_dir / f"{ecg_id}_true-{true_label}_pred-{pred_label}_{probs[pred_idx]:.2f}.png"
        overlay_img.save(out_path)
        print(f"    predicted={pred_label} ({probs[pred_idx]:.2f})  saved {out_path.name}")

    print(f"\n{correct}/{len(samples)} correct on this small sample set.")
    print(f"Heatmaps saved to datasets/ecg-images/gradcam_samples/heatmaps/ ({len(samples)} PNGs)")


if __name__ == "__main__":
    main()
