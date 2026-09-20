# Baselines

This directory contains code-only baseline implementations. Training outputs, checkpoints, predictions and result files are written under `outputs/`, which is ignored by Git.

- `dmsa.py`: DMSA-CNN-style multiscale attentional CNN implementation from the supplied comparison material.
- `eeg_inception.py`: EEG-Inception-style model implementation from the supplied comparison material.
- `ctnet.py`: clean model-only extraction of the CTNet architecture described by Zhao et al., *Scientific Reports* 14, 20237 (2024), DOI `10.1038/s41598-024-71118-7`.
- `eegnet.py`: compact EEGNet-8,2 implementation for three-channel raw EEG (Lawhern et al., 2018).
- `deepconvnet.py`: four-block Deep ConvNet implementation (Schirrmeister et al., 2017).
- `fbcsp.py`: filter-bank CSP, mutual-information feature selection, and SVM baseline (Ang et al., 2008).
- `train_loso.py`: unified runner for all non-MASSANet baselines on IV-2b and OpenBMI.
- `run_massanet.py`: leakage-safe adapter for the 2026 MASSANet model. It imports a separately cloned upstream repository and runs the paper's three-channel protocols (IV-2b LOSO; OpenBMI HO or LOSO) without copying MASSANet source into this repository.

## Other baselines

Install `requirements-baselines.txt`. The unified runner accepts `dmsa`,
`eeg_inception`, `ctnet`, `eegnet`, `deepconvnet`, and `fbcsp_svm`. It reuses
the MASSANet adapter's exact data loaders and split construction: fixed 8--30 Hz,
250 Hz, 4-second C3/Cz/C4 inputs; group-disjoint validation for LOSO; and
per-channel normalization fitted only on the fit partition.

Run IV-2b LOSO (training subjects use T+E; each held-out subject is evaluated on E):

```bash
python baselines/train_loso.py \
  --model eegnet \
  --dataset bci_iv_2b \
  --data-root /root/autodl-fs/datasets/BCI_IV_2b/mymat_withoutFilter \
  --output outputs/eegnet_iv2b_loso_seed42
```

Run OpenBMI HO (session 1 train/validation, session 2 test):

```bash
python baselines/train_loso.py \
  --model deepconvnet \
  --dataset openbmi \
  --protocol ho \
  --data-root /root/autodl-fs/OpenBMI \
  --cache-dir /root/autodl-tmp/openbmi_massanet_cache \
  --output outputs/deepconvnet_openbmi_ho_seed42
```

Run OpenBMI LOSO with four fold workers (replace `--model` and `--output` for
each baseline):

```bash
python scripts/run_openbmi_baselines_parallel.py \
  --model fbcsp_svm \
  --data-root /root/autodl-fs/OpenBMI \
  --cache-dir /root/autodl-tmp/openbmi_massanet_cache \
  --output outputs/fbcsp_svm_openbmi_loso_seed42 \
  --parallelism 4
```

Use `--subjects 1 --check-only` before a full run. A completed fold contains its
configuration, exact trial IDs, training history, test predictions, and metrics.
The neural baselines save `best.pth`; FBCSP-SVM saves `best.pkl`. Both file types,
all CSV logs, and every `outputs/` directory are ignored by Git. Parameter counts
are reported for neural networks; FBCSP-SVM reports `null` with an explanation
because its learned CSP filters and SVM support vectors are data-dependent.

These are transparent in-repository implementations of the published architectures.

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

The IV-2b commands in this section use the paper-aligned normalization implemented
by the adapter: per-channel mean and standard deviation are fitted on each fold's
fit partition and then applied unchanged to validation and test data. Results from
older adapter revisions that scaled every trial by its own maximum absolute value
must not be mixed with this protocol.

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

To run independent LOSO folds with four worker processes, use the parameterized
launcher below. It first materializes the cache in one process, skips subjects
already complete in the final output, keeps worker logs/results below the
selected output parent, merges non-conflicting fold directories, and requires a
complete 54-subject summary when all subjects are requested:

```bash
python scripts/run_massanet_openbmi_parallel.py \
  --data-root /path/to/OpenBMI \
  --cache-dir /path/to/openbmi_massanet_cache \
  --massanet-root /path/to/MASSANet \
  --output outputs/massanet_openbmi_loso_seed42 \
  --parallelism 4 \
  --device cuda
```

Each process uses the same GPU by default. Reduce `--parallelism` if GPU memory
is insufficient; use `--device cuda:N` when launching separate commands on
different GPUs. The launcher itself never copies data or generated outputs into
tracked repository paths.

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

Before public redistribution under an open-source license, verify the provenance and licensing of each implementation. The repository currently uses an all-rights-reserved project license.

