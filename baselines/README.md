# Baselines

This directory contains code-only baseline implementations. It intentionally contains no historical logs, checkpoints, predictions, or aggregate result files.

- `dmsa.py`: DMSA-CNN-style multiscale attentional CNN implementation from the supplied comparison material.
- `eeg_inception.py`: EEG-Inception-style model implementation from the supplied comparison material.
- `ctnet.py`: clean model-only extraction of the CTNet architecture described by Zhao et al., *Scientific Reports* 14, 20237 (2024), DOI `10.1038/s41598-024-71118-7`.
- `train_loso.py`: configurable raw-MAT LOSO runner. Outputs are always written below a user-selected output directory and are ignored by Git.

The supplied historical CTNet notebook depended on unavailable helper modules and contained embedded outputs, so it is not included. The model-only extraction contains no notebook output or reported accuracy.

Before public redistribution under an open-source license, verify the provenance and licensing of each implementation. The repository currently uses an all-rights-reserved project license.

