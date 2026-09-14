# Third-party notices

The vendored `gumpy` package is licensed under the MIT License. Its original license is retained at `third_party/gumpy/LICENSE`.

The baseline implementations under `baselines/` were organized from research materials supplied with this project, and the paper citations are retained in `baselines/README.md`. Before changing this repository to a permissive open-source license, confirm the provenance and redistribution terms of each baseline implementation. The repository-level `LICENSE` does not relicense third-party code.

`baselines/run_massanet.py` is an interoperability/training adapter and does not
contain MASSANet implementation source. At runtime it loads a separately cloned
copy from `https://github.com/Taowelll/MASSANet`; users must review and comply
with that upstream repository's terms. The adapter records the external commit
and SHA-256 hashes used for each experiment.
