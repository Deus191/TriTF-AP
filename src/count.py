"""Aggregate independent FINAL test_metrics.json files, never per-epoch maxima."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def summarize(root, output):
    records, conditions = [], set()
    for metrics_path in sorted(Path(root).glob("sub_*/test_metrics.json")):
        config = json.loads((metrics_path.parent / "config.json").read_text(encoding="utf-8"))
        if config.get("status") != "complete":
            raise ValueError("Incomplete run: " + str(metrics_path.parent))
        conditions.add(tuple(config.get(key) for key in ("dataset", "protocol", "model", "seed", "augment")))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        records.append((metrics_path.parent.name, metrics["accuracy"], metrics["kappa"], metrics["n_test"]))
    if not records or len(conditions) != 1:
        raise ValueError("No completed final tests, or mixed experiment conditions.")
    with Path(output).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "test_accuracy", "test_kappa", "n_test"])
        writer.writerows(records)
    summary = {"subjects": len(records), "macro_accuracy": float(np.mean([r[1] for r in records])),
               "macro_kappa": float(np.mean([r[2] for r in records])),
               "note": "Partial subject sets are not a complete LOSO benchmark."}
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)


if __name__ == "__main__":
    main()
