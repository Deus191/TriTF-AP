# TriTF-AP research code

This is the public-code layout for the TriTF-AP motor-imagery EEG project. It contains an executable implementation of the method specified in the manuscript, preprocessing and training utilities, baseline model implementations, tests, and example dataset configuration.

No EEG data, generated time-frequency images, experiment logs, predictions, checkpoints, result tables, or manuscript files are included. All run outputs must be written below `outputs/`, which is ignored by Git.

## Repository layout

```text
src/                 TriTF-AP preprocessing, augmentation, model, and training code
baselines/           Raw-signal baselines and leakage-safe LOSO runners
config/              Dataset configuration template without local computer paths
scripts/             Dataset helpers, parallel launchers, and paired statistics
tests/               Offline regression and smoke tests
third_party/         Third-party notices and vendored dependency licenses
```

The code in `src/` is the implementation of the manuscript method.

## Manuscript-method contract

The public entry path is `VAE.py -> Hope.Train -> paper_model.build_classifier`. It implements:

- C3/Cz/C4 input interpreted uniformly in that order;
- `C3-Cz` and `C4-Cz` derivations, with one 8--30 Hz Morlet-CWT log-power map per derivation;
- vertical concatenation of the two maps, joint per-trial normalization, fixed `jet` mapping, and resize to `64x64x3` RGB;
- training-split-only convolutional-VAE augmentation;
- the manuscript's DW8/PW24 stem, four-branch RepSeparable block, PW32, SimAM, GAP, Dense72 and softmax head;
- validation-loss early stopping and analytical deployment fusion.

The TF writer has no switch for a three-map or non-bipolar representation. Each generated split contains `preprocessing.json` recording the representation contract. See [METHOD_IMPLEMENTATION.md](METHOD_IMPLEMENTATION.md) for the exact paper-to-code mapping.

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

Do not mix images made by an older preprocessing implementation with this version. The writer refuses to overwrite an existing subject/split directory; use a new output root when regenerating.

Run TriTF-AP on IV-2b images:

```powershell
.\.venv\Scripts\python.exe src/VAE.py --dataset 2b --protocol LOSO --output outputs/tritf_ap_2b_seed42
```

Run a non-MASSANet baseline on the same three-channel IV-2b protocol:

```powershell
.\.venv\Scripts\python.exe baselines/train_loso.py --model eegnet --dataset bci_iv_2b --data-root D:/path/to/mat --output outputs/eegnet_seed42
```

Available choices are `dmsa`, `eeg_inception`, `ctnet`, `eegnet`,
`deepconvnet`, and `fbcsp_svm`. The same runner supports OpenBMI HO and LOSO;
see [baselines/README.md](baselines/README.md) for commands and the four-worker
AutoDL launcher.

The MASSANet adapter supports the manuscript's three-channel LOSO protocol on
both IV-2b and OpenBMI while loading the upstream MASSANet model from a separate
clone. See [baselines/README.md](baselines/README.md) for AutoDL commands and
usage details.

For a full OpenBMI LOSO run on a GPU host, the public
launcher can split the 54 folds across four independent processes, resume
completed subjects, merge fold directories, and verify the final summary:

```bash
python scripts/run_massanet_openbmi_parallel.py \
  --data-root /path/to/OpenBMI \
  --cache-dir /path/to/openbmi_massanet_cache \
  --massanet-root /path/to/MASSANet \
  --output outputs/massanet_openbmi_loso_seed42 \
  --parallelism 4
```

Subject-wise paired Wilcoxon tests, Holm correction, rank-biserial effect sizes,
exact sign tests, and paired-bootstrap confidence intervals can be regenerated
without committing result files:

```bash
python scripts/paired_statistics.py \
  --ours /path/to/ours.csv Accuracy \
  --baseline MASSANet /path/to/massanet.csv Accuracy \
  --family-name "OpenBMI LOSO confirmatory comparison" \
  --output outputs/openbmi_statistics
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Publication safety

Before pushing, run `git status --short --ignored` and `git ls-files`. Files matching data, result, model-weight, archive, image, and local-environment patterns are excluded by the root `.gitignore`.

The project's own source is released under the MIT License (see `LICENSE`). Third-party components remain under their own licenses; see `THIRD_PARTY_NOTICES.md`.
