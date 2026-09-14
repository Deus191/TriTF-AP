# Baselines

This directory contains code-only baseline implementations. It intentionally contains no historical logs, checkpoints, predictions, or aggregate result files.

- `dmsa.py`: DMSA-CNN-style multiscale attentional CNN implementation from the supplied comparison material.
- `eeg_inception.py`: EEG-Inception-style model implementation from the supplied comparison material.
- `ctnet.py`: clean model-only extraction of the CTNet architecture described by Zhao et al., *Scientific Reports* 14, 20237 (2024), DOI `10.1038/s41598-024-71118-7`.
- `train_loso.py`: configurable raw-MAT LOSO runner. Outputs are always written below a user-selected output directory and are ignored by Git.
- `run_massanet.py`: leakage-safe adapter for the 2026 MASSANet model. It imports a separately cloned upstream repository and runs the paper's three-channel protocols (IV-2b LOSO; OpenBMI HO or LOSO) without copying MASSANet source into this repository.

## MASSANet adapter

Install the optional dependencies and clone the upstream implementation:

```bash
python -m pip install -r requirements-massanet.txt
git clone https://github.com/Taowelll/MASSANet.git /root/autodl-tmp/MASSANet
```

Validate the uploaded IV-2b MAT data and model before training:

```bash
python baselines/run_massanet.py \
  --dataset bci_iv_2b \
  --data-root /root/autodl-fs/datasets/BCI_IV_2b/mymat_withoutFilter \
  --massanet-root /root/autodl-tmp/MASSANet \
  --output outputs/massanet_iv2b_seed42 \
  --check-only
```

Run one held-out subject as a smoke experiment, then resume with all subjects:

```bash
python baselines/run_massanet.py \
  --dataset bci_iv_2b \
  --data-root /root/autodl-fs/datasets/BCI_IV_2b/mymat_withoutFilter \
  --massanet-root /root/autodl-tmp/MASSANet \
  --output outputs/massanet_iv2b_seed42 \
  --subjects 1

python baselines/run_massanet.py \
  --dataset bci_iv_2b \
  --data-root /root/autodl-fs/datasets/BCI_IV_2b/mymat_withoutFilter \
  --massanet-root /root/autodl-tmp/MASSANet \
  --output outputs/massanet_iv2b_seed42 \
  --resume
```

OpenBMI uses MOABB's labelled offline runs from both sessions, selects C3/Cz/C4,
resamples to 250 Hz, and caches fixed 8--30 Hz four-second trials. The adapter
forces MOABB's `upstream` provider so an existing GigaDB/OpenBMI download is reused
instead of starting a second NEMAR download.

Run the within-subject HO protocol (session 1 train/validation, session 2 test):

```bash
python baselines/run_massanet.py \
  --dataset openbmi \
  --protocol ho \
  --data-root /root/autodl-fs/OpenBMI \
  --cache-dir /root/autodl-tmp/openbmi_massanet_cache \
  --massanet-root /root/autodl-tmp/MASSANet \
  --output outputs/massanet_openbmi_ho_seed42
```

Run 54-fold cross-subject LOSO (both labelled sessions of the held-out subject
form the test set):

```bash
python baselines/run_massanet.py \
  --dataset openbmi \
  --protocol loso \
  --data-root /root/autodl-fs/OpenBMI \
  --cache-dir /root/autodl-tmp/openbmi_massanet_cache \
  --massanet-root /root/autodl-tmp/MASSANet \
  --output outputs/massanet_openbmi_loso_seed42
```

Defaults retain MASSANet's published training settings where applicable: batch
size 16, Adam with learning rate `1e-3` and weight decay `0.075`, label smoothing
`0.05`, CutCat, cosine annealing, and dropout `0.1`. CutCat is restricted to the
fit partition. Model selection uses validation loss and early stopping with
patience 30; pass `--patience 0 --selection-metric validation_accuracy` for the
upstream checkpoint-selection behavior. Each fold records the upstream commit
and source hashes, exact trial IDs, checkpoint, predictions, metrics, and model
parameter count. Per-channel mean and standard deviation are fitted only on the
fold's fit partition and then applied unchanged to validation and test data.
`summary.json` reports subject-wise mean and sample standard deviation and marks
whether the full selected protocol is complete.

The supplied historical CTNet notebook depended on unavailable helper modules and contained embedded outputs, so it is not included. The model-only extraction contains no notebook output or reported accuracy.

Before public redistribution under an open-source license, verify the provenance and licensing of each implementation. The repository currently uses an all-rights-reserved project license.

