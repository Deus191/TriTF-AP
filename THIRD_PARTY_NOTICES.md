# Third-party notices

The project's own source is released under the MIT License (see `LICENSE`). The
components listed below contain or adapt code from other projects and remain
governed by their own licenses. The repository-level `LICENSE` does not
relicense them.

## Vendored and adapted code

| Path | Origin | License |
|---|---|---|
| `src/gumpy/` | [gumpy](https://github.com/gumpy-bci/gumpy) | MIT, Copyright (c) 2018 The gumpy developers |
| `baselines/dmsa.py` | EEGNet/DeepConvNet-style primitives from [ravikiran-mane/FBCNet](https://github.com/ravikiran-mane/FBCNet); DMSA layers original to this repository | MIT, Copyright (c) 2020 ravikiran-mane |
| `baselines/ctnet.py` | CTNet model extracted from [snailpt/CTNet](https://github.com/snailpt/CTNet) | MIT, Copyright (c) 2024 snailpt |

Full license texts are retained at `third_party/gumpy/LICENSE`,
`third_party/FBCNet-LICENSE` and `third_party/CTNet-LICENSE`. Each keeps its
original copyright notice, as the MIT License requires.

The remaining baseline implementations — `eegnet.py`, `deepconvnet.py`,
`fbcsp.py`, `train_loso.py` and `run_massanet.py` — are original to this
repository and are covered by the project's MIT License.

## External runtime dependencies

`baselines/run_massanet.py` is an interoperability/training adapter and does not
contain MASSANet implementation source. At runtime it loads a separately cloned
copy from `https://github.com/Taowelll/MASSANet`; users must review and comply
with that upstream repository's terms. The adapter records the external commit
and SHA-256 hashes used for each experiment.
