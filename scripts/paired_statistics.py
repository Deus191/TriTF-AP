"""Reproducible subject-wise paired tests for TriTF-AP comparisons.

Input CSV files remain external to Git.  Each file must contain a subject column
and one numeric metric column.  Values may be proportions or percentages.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, rankdata, wilcoxon


def subject_number(value):
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError("Cannot extract a subject number from {!r}".format(value))
    return int(match.group(1))


def read_metric(path, metric_column, subject_column="Subject"):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("CSV has no header: {}".format(path))
        lookup = {name.lower(): name for name in reader.fieldnames}
        subject_key = lookup.get(subject_column.lower())
        metric_key = lookup.get(metric_column.lower())
        if subject_key is None or metric_key is None:
            raise ValueError(
                "{} must contain subject {!r} and metric {!r}; found {}".format(
                    path, subject_column, metric_column, reader.fieldnames
                )
            )
        values = {}
        for row in reader:
            subject = subject_number(row[subject_key])
            if subject in values:
                raise ValueError("Duplicate subject {} in {}".format(subject, path))
            values[subject] = float(row[metric_key])
    if not values:
        raise ValueError("No rows in {}".format(path))
    scale = 100.0 if max(abs(value) for value in values.values()) > 1.5 else 1.0
    return {subject: value / scale for subject, value in values.items()}


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_ci(differences, resamples, seed):
    rng = np.random.default_rng(seed)
    means = []
    remaining = resamples
    while remaining:
        batch = min(10_000, remaining)
        indices = rng.integers(0, len(differences), size=(batch, len(differences)))
        means.append(differences[indices].mean(axis=1))
        remaining -= batch
    return np.quantile(np.concatenate(means), [0.025, 0.975]).tolist()


def compare(ours, baseline, resamples, seed):
    subjects = sorted(set(ours).intersection(baseline))
    if len(subjects) < 5:
        raise ValueError("At least five matched subjects are required; found {}".format(len(subjects)))
    ours_values = np.asarray([ours[subject] for subject in subjects], dtype=float)
    baseline_values = np.asarray([baseline[subject] for subject in subjects], dtype=float)
    differences = ours_values - baseline_values
    nonzero = differences[differences != 0]
    method = "exact" if len(nonzero) <= 25 and len(nonzero) == len(differences) else "auto"
    test = wilcoxon(
        differences,
        zero_method="wilcox",
        alternative="two-sided",
        method=method,
    )
    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    rank_biserial = (positive - negative) / (positive + negative) if positive + negative else 0.0
    wins = int((differences > 0).sum())
    losses = int((differences < 0).sum())
    ties = int((differences == 0).sum())
    sign_p = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
    ci_low, ci_high = bootstrap_ci(differences, resamples, seed)
    result = {
        "n_subjects": len(subjects),
        "ours_mean": float(ours_values.mean()),
        "ours_sd_sample": float(ours_values.std(ddof=1)),
        "baseline_mean": float(baseline_values.mean()),
        "baseline_sd_sample": float(baseline_values.std(ddof=1)),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "ci95_mean_difference_low": float(ci_low),
        "ci95_mean_difference_high": float(ci_high),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "wilcoxon_w": float(test.statistic),
        "wilcoxon_method": method,
        "p_raw_two_sided": float(test.pvalue),
        "rank_biserial": float(rank_biserial),
        "sign_test_p_two_sided": sign_p,
    }
    paired = [
        {
            "Subject": subject,
            "TriTF_AP": float(ours_values[index]),
            "Baseline": float(baseline_values[index]),
            "Difference": float(differences[index]),
        }
        for index, subject in enumerate(subjects)
    ]
    return result, paired


def safe_name(value):
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return result or "baseline"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", nargs=2, metavar=("CSV", "METRIC"), required=True)
    parser.add_argument(
        "--baseline",
        nargs=3,
        action="append",
        metavar=("NAME", "CSV", "METRIC"),
        required=True,
        help="Repeat for every baseline in the same correction family",
    )
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument("--family-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        parser.error("--bootstrap-resamples must be positive")
    ours = read_metric(Path(args.ours[0]), args.ours[1], args.subject_column)
    rows = []
    paired_outputs = []
    for name, csv_path, metric in args.baseline:
        baseline = read_metric(Path(csv_path), metric, args.subject_column)
        result, paired = compare(ours, baseline, args.bootstrap_resamples, args.seed)
        result.update({"comparison": "TriTF-AP vs {}".format(name), "baseline": name})
        rows.append(result)
        paired_outputs.append((name, paired))

    adjusted = holm_adjust([row["p_raw_two_sided"] for row in rows])
    for row, p_holm in zip(rows, adjusted):
        row["p_holm"] = float(p_holm)
        row["significant_at_alpha"] = bool(p_holm < args.alpha)
        row["alpha"] = args.alpha
        row["family_name"] = args.family_name

    args.output.mkdir(parents=True, exist_ok=True)
    summary_json = {
        "family_name": args.family_name,
        "family_size": len(rows),
        "test": "two-sided subject-wise paired Wilcoxon signed-rank",
        "multiplicity": "Holm",
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "comparisons": rows,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, paired in paired_outputs:
        with (args.output / ("paired_" + safe_name(name) + ".csv")).open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    print(json.dumps(summary_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
