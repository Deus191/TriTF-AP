# TriTF-AP research code

This is the public-code layout for the TriTF-AP motor-imagery EEG project. It contains the reconstructed paper model, preprocessing and training utilities, baseline model implementations, tests, and example dataset configuration.

No EEG data, generated time-frequency images, experiment logs, predictions, checkpoints, result tables, or manuscript files are included. All run outputs must be written below `outputs/`, which is ignored by Git.

## Repository layout

```text
src/                 TriTF-AP preprocessing, augmentation, model, and training code
baselines/           DMSA-CNN, EEG-Inception, and CTNet model code plus a LOSO runner
config/              Dataset configuration template without local computer paths
scripts/             Explicit dataset inspection/download helper
tests/               Offline regression and smoke tests
third_party/         Third-party notices and vendored dependency licenses
```

The code in `src/` is a repaired/reconstructed implementation. It is not a claim that historical checkpoints or manuscript accuracy values have been reproduced.

## Installation

Python 3.12 was used for the local repair tests. Create an isolated environment and install:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyTorch baseline dependencies are listed separately:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-baselines.txt
```

## Dataset configuration

Copy `config/datasets.example.json` to `config/datasets.json` and fill in paths on the machine that will run the experiments. The populated file is intentionally ignored because it may contain private local paths.

```powershell
Copy-Item config/datasets.example.json config/datasets.json
.\.venv\Scripts\python.exe scripts/datasets.py --check-local
```

Dataset files must be obtained from their official providers and remain outside Git.

## Running

Generate IV-2b time-frequency images into an ignored output directory after configuring the raw dataset path:

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from preprocessing_2b import GetPrecossedData; GetPrecossedData(1)"
```

Run TriTF-AP on IV-2b images:

```powershell
.\.venv\Scripts\python.exe src/VAE.py --dataset 2b --protocol LOSO --output outputs/tritf_ap_2b_seed42
```

Run a raw-signal baseline on MAT files containing `data` and `label` arrays:

```powershell
.\.venv\Scripts\python.exe baselines/train_loso.py --model eeg_inception --data-root D:/path/to/mat --output outputs/eeg_inception_seed42
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Publication safety

Before pushing, run `git status --short --ignored` and `git ls-files`. Files matching data, result, model-weight, archive, image, and local-environment patterns are excluded by the root `.gitignore`.

The project-specific source is currently distributed with all rights reserved. Review `LICENSE` and the provenance notes in `THIRD_PARTY_NOTICES.md` before changing the repository to an open-source license.

