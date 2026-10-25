"""Build a REAL paired multimodal cohort from MIMIC-IV, replacing random pairing.

train_fusion.py pairs UCI tabular rows with PTB-XL ECGs at random, because
those datasets share no patients. MIMIC-IV's modules do: one hospital, one
subject_id. This script finds hospital admissions that have, for the SAME
patient during the SAME admission:

  - a 12-lead ECG              (MIMIC-IV-ECG, open access)
  - a frontal chest X-ray      (MIMIC-CXR-JPG, credentialed)
  - blood labs                 (MIMIC-IV hosp/labevents, credentialed)
  - echo LV ejection fraction  (MIMIC-IV-ECHO, credentialed; optional)

runs CordiaAI's own trained ECG and X-ray branches on that patient's files,
and labels each admission from its discharge diagnosis codes. The output CSV
is what train_late_fusion.py fits the combiner on.

Labels (ICD-10 / ICD-9 discharge codes of the admission):
  ischemic heart disease  I20-I25 / 410-414
  heart failure           I50 / 428
  cardiomyopathy          I42 / 425
  valve disease           I05-I08, I34-I37 / 394-397, 424
`heart_disease` = any of the four. Discharge codes include known history,
not only what the admission found -- state this when reporting results.

Run on Colab with a GPU after PhysioNet credentialing (see
download_imaging_datasets.py: mimic_iv_ecg, mimic_cxr_jpg, mimic_iv, and
optionally mimic_iv_echo). Example:

    python build_mimic_paired_cohort.py \
        --ecg-root /content/mimic/mimic-iv-ecg/1.0 \
        --cxr-root /content/mimic/mimic-cxr-jpg/2.1.0 \
        --mimic-root /content/mimic/mimiciv/3.1 \
        --max-admissions 5000 --device cuda

MIMIC-IV-ECHO's measurement table is wide-ranging and this script does not
assume its column names: pass --echo-csv plus the four --echo-* column
arguments after looking at the file's header (the script prints it).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent

ICD_GROUPS = {
    "ischemic": (["I20", "I21", "I22", "I23", "I24", "I25"], ["410", "411", "412", "413", "414"]),
    "heart_failure": (["I50"], ["428"]),
    "cardiomyopathy": (["I42"], ["425"]),
    "valve": (["I05", "I06", "I07", "I08", "I34", "I35", "I36", "I37"], ["394", "395", "396", "397", "424"]),
}
LAB_LABELS = {  # d_labitems.label -> combiner column
    "Troponin T": "lab_troponin_t",
    "NTproBNP": "lab_ntprobnp",
    "Creatinine": "lab_creatinine",
    "Hemoglobin": "lab_hemoglobin",
    "Glucose": "lab_glucose",
    "Cholesterol, Total": "lab_cholesterol",
}


def read(path: Path, **kw) -> pd.DataFrame:
    for candidate in (path, path.with_name(path.name + ".gz")):
        if candidate.exists():
            return pd.read_csv(candidate, **kw)
    raise FileNotFoundError(f"{path}(.gz) not found -- check the --*-root arguments")


def within_admission(events: pd.DataFrame, adm: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Keep events during an admission; earliest event per admission."""
    m = events.merge(adm[["subject_id", "hadm_id", "admittime", "dischtime"]], on="subject_id")
    m = m[(m[time_col] >= m["admittime"]) & (m[time_col] <= m["dischtime"])]
    return m.sort_values(time_col).drop_duplicates("hadm_id")


def labels_for(hadm_ids: set, mimic_root: Path) -> pd.DataFrame:
    dx = read(mimic_root / "hosp" / "diagnoses_icd.csv", dtype={"icd_code": str})
    dx = dx[dx["hadm_id"].isin(hadm_ids)]
    out = pd.DataFrame({"hadm_id": sorted(hadm_ids)})
    for group, (icd10, icd9) in ICD_GROUPS.items():
        hit = ((dx["icd_version"] == 10) & dx["icd_code"].str[:3].isin(icd10)) | (
            (dx["icd_version"] == 9) & dx["icd_code"].str[:3].isin(icd9))
        out[group] = out["hadm_id"].isin(dx.loc[hit, "hadm_id"]).astype(int)
    out["heart_disease"] = out[list(ICD_GROUPS)].max(axis=1)
    return out


def labs_for(hadm_ids: set, mimic_root: Path, chunksize: int = 5_000_000) -> pd.DataFrame:
    items = read(mimic_root / "hosp" / "d_labitems.csv")
    items = items[items["label"].isin(LAB_LABELS)]
    item_to_col = dict(zip(items["itemid"], items["label"].map(LAB_LABELS)))
    parts = []
    path = mimic_root / "hosp" / "labevents.csv"
    path = path if path.exists() else path.with_name("labevents.csv.gz")
    cols = ["hadm_id", "itemid", "charttime", "valuenum"]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):  # labevents is too big to load at once
        parts.append(chunk[chunk["hadm_id"].isin(hadm_ids) & chunk["itemid"].isin(item_to_col)])
    labs = pd.concat(parts).dropna(subset=["valuenum"]).sort_values("charttime")
    labs["col"] = labs["itemid"].map(item_to_col)
    first = labs.drop_duplicates(["hadm_id", "col"])  # first value in the admission
    return first.pivot(index="hadm_id", columns="col", values="valuenum").reset_index()


def ecg_spectrogram(record_path: Path) -> np.ndarray:
    """MIMIC-IV-ECG is 500 Hz; the ECG branch was trained on PTB-XL's 100 Hz Lead II, so
    downsample 5:1 and apply the same STFT as gradcam_ecg.py / main.py."""
    import librosa
    import wfdb
    from PIL import Image
    from scipy.signal import resample_poly

    signal, fields = wfdb.rdsamp(str(record_path))
    lead_ii = signal[:, [n.upper() for n in fields["sig_name"]].index("II")].astype(np.float32)
    lead = resample_poly(np.nan_to_num(lead_ii), up=100, down=int(fields["fs"])).astype(np.float32)
    stft = librosa.stft(lead, n_fft=64, hop_length=8, window="hann")
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    norm = (db - db.min()) / (db.max() - db.min() + 1e-8)
    img = Image.fromarray((norm * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    return np.stack([np.array(img)] * 3, axis=-1)


@torch.no_grad()
def run_ecg_branch(paths: list[Path], device: str) -> list[dict | None]:
    from src.models.ecg_vit import CLASSES, ECGViT

    model = ECGViT()
    model.load_state_dict(torch.load(ROOT / "checkpoints" / "ecg_vit.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    out = []
    for i, p in enumerate(paths):
        try:
            x = (ecg_spectrogram(p).astype(np.float32) / 255.0 - 0.5) / 0.5
            probs = torch.softmax(model(torch.tensor(x).permute(2, 0, 1)[None].to(device)), 1)[0].cpu().numpy()
            out.append({f"ecg_{c}": float(v) for c, v in zip(CLASSES, probs)})
        except Exception as e:  # unreadable or lead II missing: leave the branch empty for this admission
            print(f"  ECG skipped ({p.name}): {e}")
            out.append(None)
        if (i + 1) % 500 == 0:
            print(f"  ECG {i + 1}/{len(paths)}")
    return out


@torch.no_grad()
def run_cxr_branch(paths: list[Path], device: str) -> list[dict | None]:
    from PIL import Image

    from src.imaging.preprocess import xray_to_tensor
    from src.models.imaging.cxr_vit import CXR_FINDINGS, CXRViT

    model = CXRViT(pretrained=False)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / "cxr_vit.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    out = []
    for i, p in enumerate(paths):
        try:
            probs = torch.sigmoid(model(xray_to_tensor(Image.open(p))[None].to(device)))[0].cpu().numpy()
            out.append({f"cxr_{f}": float(v) for f, v in zip(CXR_FINDINGS, probs)})
        except Exception as e:
            print(f"  X-ray skipped ({p.name}): {e}")
            out.append(None)
        if (i + 1) % 500 == 0:
            print(f"  X-ray {i + 1}/{len(paths)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ecg-root", type=Path, required=True)
    ap.add_argument("--cxr-root", type=Path, required=True)
    ap.add_argument("--mimic-root", type=Path, required=True)
    ap.add_argument("--echo-csv", type=Path)
    ap.add_argument("--echo-subject-col"), ap.add_argument("--echo-time-col")
    ap.add_argument("--echo-name-col"), ap.add_argument("--echo-value-col")
    ap.add_argument("--echo-ef-name", help="value of --echo-name-col that means LV ejection fraction")
    ap.add_argument("--max-admissions", type=int, default=5000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "paired" / "mimic_paired_cohort.csv")
    args = ap.parse_args()

    adm = read(args.mimic_root / "hosp" / "admissions.csv", parse_dates=["admittime", "dischtime"])
    pts = read(args.mimic_root / "hosp" / "patients.csv")[["subject_id", "gender", "anchor_age"]]

    ecg = read(args.ecg_root / "record_list.csv", parse_dates=["ecg_time"])
    ecg = within_admission(ecg, adm, "ecg_time")[["subject_id", "hadm_id", "ecg_time", "path"]]
    print(f"Admissions with an ECG: {len(ecg)}")

    meta = read(args.cxr_root / "mimic-cxr-2.0.0-metadata.csv")
    meta = meta[meta["ViewPosition"].isin(["PA", "AP"])].copy()
    meta["cxr_time"] = pd.to_datetime(
        meta["StudyDate"].astype(str) + meta["StudyTime"].astype(float).astype(int).astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S", errors="coerce")
    meta["cxr_path"] = meta.apply(
        lambda r: f"files/p{str(r.subject_id)[:2]}/p{r.subject_id}/s{r.study_id}/{r.dicom_id}.jpg", axis=1)
    cxr = within_admission(meta.dropna(subset=["cxr_time"]), adm, "cxr_time")[["hadm_id", "cxr_time", "cxr_path"]]

    cohort = ecg.merge(cxr, on="hadm_id")
    print(f"Admissions with ECG + chest X-ray for the same patient: {len(cohort)}")
    if len(cohort) > args.max_admissions:
        cohort = cohort.sample(n=args.max_admissions, random_state=42)

    cohort = cohort.merge(pts, on="subject_id", how="left")
    cohort["age"] = cohort["anchor_age"]
    cohort["sex"] = (cohort["gender"] == "M").astype(int)  # same coding as UCI: 1 = male

    hadm_ids = set(cohort["hadm_id"])
    cohort = cohort.merge(labs_for(hadm_ids, args.mimic_root), on="hadm_id", how="left")
    cohort = cohort.merge(labels_for(hadm_ids, args.mimic_root), on="hadm_id", how="left")

    if args.echo_csv:
        echo = pd.read_csv(args.echo_csv, nrows=5)
        needed = [args.echo_subject_col, args.echo_time_col, args.echo_name_col, args.echo_value_col, args.echo_ef_name]
        if not all(needed):
            print(f"Echo CSV columns: {list(echo.columns)}\nRe-run with --echo-subject-col/--echo-time-col/"
                  "--echo-name-col/--echo-value-col/--echo-ef-name set from these. Skipping echo for now.")
        else:
            echo = pd.read_csv(args.echo_csv, parse_dates=[args.echo_time_col])
            echo = echo[echo[args.echo_name_col] == args.echo_ef_name].rename(
                columns={args.echo_subject_col: "subject_id", args.echo_time_col: "echo_time",
                         args.echo_value_col: "echo_lv_ef_percent"})
            echo = within_admission(echo[["subject_id", "echo_time", "echo_lv_ef_percent"]], adm, "echo_time")
            cohort = cohort.merge(echo[["hadm_id", "echo_lv_ef_percent"]], on="hadm_id", how="left")
            print(f"  of which with an echo EF: {cohort['echo_lv_ef_percent'].notna().sum()}")

    print(f"Running ECG branch on {len(cohort)} ECGs ({args.device})...")
    ecg_out = run_ecg_branch([args.ecg_root / p for p in cohort["path"]], args.device)
    print("Running X-ray branch...")
    cxr_out = run_cxr_branch([args.cxr_root / p for p in cohort["cxr_path"]], args.device)
    cohort = pd.concat([cohort.reset_index(drop=True),
                        pd.DataFrame([o or {} for o in ecg_out]), pd.DataFrame([o or {} for o in cxr_out])], axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cohort.drop(columns=["anchor_age", "gender"]).to_csv(args.out, index=False)
    print(f"\nSaved {len(cohort)} paired admissions ({cohort['subject_id'].nunique()} patients) to {args.out}")
    print("Label balance:", cohort[["heart_disease", *ICD_GROUPS]].mean().round(3).to_dict())
    print("This file holds MIMIC-derived data: keep it private, as the PhysioNet DUA requires.")


if __name__ == "__main__":
    main()
