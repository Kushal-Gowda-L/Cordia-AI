"""Download and check the imaging datasets for the X-ray, echo, MRI and CT branches.

Usage:
    python download_imaging_datasets.py list
    python download_imaging_datasets.py download acdc --dest datasets/mri/acdc
    python download_imaging_datasets.py download mimic_cxr_jpg --dest /content/mimic --physionet-user YOUR_NAME
    python download_imaging_datasets.py verify echonet_dynamic --root datasets/echo/EchoNet-Dynamic

Two kinds of dataset, and why:
- "auto": a public download that needs no agreement beyond its licence
  (or only your own free Kaggle / PhysioNet login), so this script fetches it.
- "manual": the dataset owner requires YOU to register and accept a data
  use agreement (Stanford AIMI, Human Heart Project, PhysioNet credentialed
  access, emailing the authors). That acceptance is a legal step in your
  name, so this script prints exactly where to do it, and then `verify`
  checks the folder you downloaded into has the layout the notebooks expect.
  PhysioNet credentialed sets can still be fetched here once you are
  approved, with your own `--physionet-user` (wget asks for your password).

The datasets are large (NIH X-ray ~45 GB, MIMIC-IV-ECG ~34 GB zipped), and
training needs a GPU, so the intended place to run downloads is Colab or a
Kaggle notebook (see notebooks/step15-18), not this laptop's OneDrive folder.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Dataset:
    name: str
    modality: str
    access: str  # "auto" or "manual"
    licence: str
    page: str
    size: str
    steps: list[str] = field(default_factory=list)
    expect: list[str] = field(default_factory=list)  # glob patterns `verify` must find under --root
    physionet: str | None = None  # "<slug>/<version>" for PhysioNet-hosted sets


DATASETS = {d.name: d for d in [
    Dataset(
        "nih_cxr", "Chest X-ray", "auto", "No restrictions on use; attribution required (NIH Clinical Center)",
        "https://nihcc.app.box.com/v/ChestXray-NIHCC", "112,120 images, ~45 GB",
        ["Easiest: in a Kaggle notebook, Add Data -> 'NIH Chest X-rays' (nih-chest-xrays/data); no download needed.",
         "Elsewhere: this script uses the Kaggle CLI, which needs your kaggle.json API token in ~/.kaggle/."],
        ["**/Data_Entry_2017*.csv", "**/images*/**/*.png"],
    ),
    Dataset(
        "chexpert", "Chest X-ray", "manual", "Stanford AIMI Research Use Agreement (non-commercial)",
        "https://aimi.stanford.edu/datasets/chexpert-chest-x-rays", "224,316 images",
        ["Register and accept the Research Use Agreement on the page above, then download.",
         "Optional alternative to nih_cxr; the X-ray notebook accepts either."],
        ["**/train.csv"],
    ),
    Dataset(
        "acdc", "Cardiac MRI", "auto", "CC BY-NC-SA 4.0; cite Bernard et al., IEEE TMI 2018",
        "https://www.creatis.insa-lyon.fr/Challenge/acdc/", "150 patients, ~2 GB",
        ["Fetched from the Hugging Face mirror mathpluscode/ACDC (patientXXX_sax_ed / _sax_es [_gt].nii.gz per patient).",
         "Official source, if you prefer it: free account on the Human Heart Project database, "
         "https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb"],
        ["**/patient*_gt.nii.gz"],
    ),
    Dataset(
        "echonet_dynamic", "Echocardiogram video", "manual",
        "Stanford AIMI Research Use Agreement: personal non-commercial research, no redistribution",
        "https://echonet.github.io/dynamic/index.html", "10,030 videos, ~7 GB",
        ["Register at https://stanford.redivis.com/datasets/66s1-2hsmzj5rn and accept the agreement.",
         "Download EchoNet-Dynamic.zip into your Google Drive (e.g. MyDrive/cordia/EchoNet-Dynamic.zip).",
         "Do not upload it anywhere public, including GitHub: the agreement forbids redistribution."],
        ["**/FileList.csv", "**/Videos/*.avi"],
    ),
    Dataset(
        "camus", "Echocardiogram (2- and 4-chamber)", "manual", "Licence not stated on site; cite Leclerc et al., IEEE TMI 2019",
        "https://creatis.insa-lyon.fr/Challenge/camus", "500 patients",
        ["Free account on https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8",
         "Download the NIfTI release into Drive (MyDrive/cordia/CAMUS)."],
        ["**/*_gt.nii.gz"],
    ),
    Dataset(
        "imagecas", "Coronary CT angiography", "manual", "Not stated; email the authors for the download password",
        "https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT",
        "1,000 CT scans, large",
        ["Email xiao.wei.xu@foxmail.com (per the repository README) asking for the Kaggle link and password.",
         "Put the extracted NIfTI files in Drive (MyDrive/cordia/ImageCAS)."],
        ["**/*.nii.gz"],
    ),
    Dataset(
        "asoca", "Coronary CT angiography", "manual", "Terms on the UK Data Service listing",
        "https://asoca.grand-challenge.org/", "40 training (20 normal, 20 CAD) + 20 test",
        ["Request access via the UK Data Service link on the ASOCA page; gives real normal-vs-CAD labels."],
        ["**/*.nrrd"],
    ),
    Dataset(
        "mimic_iv_ecg", "12-lead ECG (paired with MIMIC-IV)", "auto", "ODbL 1.0 (open access)",
        "https://physionet.org/content/mimic-iv-ecg/", "~800,000 ECGs, 33.8 GB zip",
        ["Open access: no credentialing needed for this one."],
        ["**/record_list.csv"], physionet="mimic-iv-ecg/1.0",
    ),
    Dataset(
        "mimic_cxr_jpg", "Chest X-ray (paired with MIMIC-IV)", "manual", "PhysioNet Credentialed Health Data License 1.5.0",
        "https://www.physionet.org/content/mimic-cxr-jpg/", "377,110 images, very large",
        ["Become a credentialed PhysioNet user (free CITI 'Data or Specimens Only Research' course), sign the DUA.",
         "Then: download mimic_cxr_jpg --physionet-user YOU --metadata-only first (labels + metadata, small)."],
        ["**/mimic-cxr-2.0.0-chexpert.csv*"], physionet="mimic-cxr-jpg/2.1.0",
    ),
    Dataset(
        "mimic_iv_echo", "Echo measurements + DICOM (paired with MIMIC-IV)", "manual",
        "PhysioNet Credentialed Health Data License 1.5.0", "https://physionet.org/content/mimic-iv-echo/",
        "206,488 studies of measurements; 7,228 with video",
        ["Credentialed access, as for mimic_cxr_jpg. --metadata-only fetches the CSVs (structured_measurement.csv)."],
        ["**/structured_measurement.csv*"], physionet="mimic-iv-echo/1.0.1",
    ),
    Dataset(
        "mimic_iv", "Clinical: admissions, diagnoses, labs", "manual", "PhysioNet Credentialed Health Data License 1.5.0",
        "https://physionet.org/content/mimiciv/", "~10 GB of CSVs",
        ["Credentialed access. Needed for labels (diagnoses_icd) and labs when building the paired cohort."],
        ["**/hosp/admissions.csv*"], physionet="mimiciv/3.1",
    ),
    Dataset(
        "echonext", "ECG + echo-derived structural heart disease labels", "manual",
        "PhysioNet Restricted Health Data License 1.5.0", "https://physionet.org/content/echonext/",
        "100,000 ECGs",
        ["Sign the PhysioNet Restricted DUA on the page above (no CITI course needed for 'restricted')."],
        ["**/*.csv"], physionet="echonext/1.1.1",
    ),
]}

METADATA_SUFFIXES = (".csv", ".csv.gz", ".txt", ".html")


def print_list() -> None:
    for d in DATASETS.values():
        print(f"{d.name:16s} {d.access:6s} {d.modality:45s} {d.size}")
    print("\n'auto' = this script downloads it; 'manual' = you register/accept terms first (see `info <name>`).")


def print_info(d: Dataset) -> None:
    print(f"{d.name} -- {d.modality}\n  Page:    {d.page}\n  Licence: {d.licence}\n  Size:    {d.size}")
    for i, s in enumerate(d.steps, 1):
        print(f"  {i}. {s}")


def _need(tool: str, hint: str) -> None:
    if shutil.which(tool) is None:
        sys.exit(f"'{tool}' is not installed. {hint}")


def download(d: Dataset, dest: Path, physionet_user: str | None, metadata_only: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if d.name == "nih_cxr":
        _need("kaggle", "pip install kaggle, and put your API token in ~/.kaggle/kaggle.json")
        subprocess.run(["kaggle", "datasets", "download", "-d", "nih-chest-xrays/data", "-p", str(dest), "--unzip"], check=True)
    elif d.name == "acdc":
        from huggingface_hub import snapshot_download

        snapshot_download("mathpluscode/ACDC", repo_type="dataset", local_dir=str(dest))
    elif d.physionet:
        if d.access == "manual" and not physionet_user:
            print_info(d)
            sys.exit("\nThis is credentialed/restricted PhysioNet data: pass --physionet-user once you are approved.")
        _need("wget", "Install wget (Colab has it).")
        cmd = ["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=1", "-P", str(dest)]
        if physionet_user:
            cmd += ["--user", physionet_user, "--ask-password"]
        if metadata_only:
            # CSVs only (labels, record lists, measurements), not the images/waveforms
            cmd += ["-A", ",".join(f"*{s}" for s in METADATA_SUFFIXES)]
        subprocess.run(cmd + [f"https://physionet.org/files/{d.physionet}/"], check=True)
    else:
        print_info(d)
        sys.exit("\nThis dataset needs your own registration first; see the steps above, then run `verify`.")
    print(f"\nDownloaded to {dest}. Checking layout...")
    verify(d, dest)


def verify(d: Dataset, root: Path) -> bool:
    ok = True
    for pattern in d.expect:
        hits = list(root.glob(pattern))
        print(f"  {'OK ' if hits else 'MISSING'} {pattern}: {len(hits)} file(s)")
        ok &= bool(hits)
    print("Layout looks right." if ok else "Some expected files are missing -- check the folder you extracted into.")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("info", "download", "verify"):
        s = sub.add_parser(name)
        s.add_argument("dataset", choices=sorted(DATASETS))
        if name == "download":
            s.add_argument("--dest", type=Path, required=True)
            s.add_argument("--physionet-user")
            s.add_argument("--metadata-only", action="store_true")
        if name == "verify":
            s.add_argument("--root", type=Path, required=True)
    args = p.parse_args()

    if args.cmd == "list":
        print_list()
    elif args.cmd == "info":
        print_info(DATASETS[args.dataset])
    elif args.cmd == "download":
        download(DATASETS[args.dataset], args.dest, args.physionet_user, args.metadata_only)
    else:
        sys.exit(0 if verify(DATASETS[args.dataset], args.root) else 1)


if __name__ == "__main__":
    main()
