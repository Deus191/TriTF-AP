"""Run OpenBMI MASSANet LOSO folds in parallel and merge verified outputs.

The launcher performs a single-process preflight first so the OpenBMI cache is
fully materialized before workers start.  Data, caches, logs, checkpoints, and
results remain below user-selected directories and are not written to Git.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "baselines" / "run_massanet.py"
EXPECTED_SUBJECTS = tuple(range(1, 55))


def split_subjects(subjects, worker_count):
    """Round-robin subjects into at most ``worker_count`` non-empty groups."""
    worker_count = min(max(1, worker_count), len(subjects))
    groups = [[] for _ in range(worker_count)]
    for index, subject in enumerate(subjects):
        groups[index % worker_count].append(subject)
    return [group for group in groups if group]


def completed_subjects(output):
    completed = set()
    if not output.is_dir():
        return completed
    for metrics_path in output.glob("sub_*/test_metrics.json"):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if metrics.get("protocol") == "loso" and metrics.get("subject") in EXPECTED_SUBJECTS:
            completed.add(int(metrics["subject"]))
    return completed


def common_command(args):
    command = [
        str(args.python),
        str(RUNNER),
        "--dataset", "openbmi",
        "--protocol", "loso",
        "--data-root", str(args.data_root),
        "--cache-dir", str(args.cache_dir),
        "--massanet-root", str(args.massanet_root),
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--patience", str(args.patience),
        "--workers", str(args.loader_workers),
        "--seed", str(args.seed),
    ]
    if args.no_augmentation:
        command.extend(["--augmentation", "none"])
    return command


def preflight(args, sample_subject):
    command = common_command(args) + [
        "--output", str(args.output),
        "--subjects", str(sample_subject),
        "--check-only",
    ]
    print("Preflight and cache materialization:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def launch_workers(args, groups, worker_root):
    processes = []
    worker_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(args.cpu_threads_per_worker))
    env.setdefault("MKL_NUM_THREADS", str(args.cpu_threads_per_worker))
    for index, subjects in enumerate(groups, start=1):
        output = worker_root / "worker_{:02d}".format(index)
        output.mkdir(parents=True, exist_ok=True)
        log_path = worker_root / "worker_{:02d}.log".format(index)
        command = common_command(args) + [
            "--output", str(output),
            "--subjects",
        ] + [str(subject) for subject in subjects] + ["--resume"]
        stream = log_path.open("a", encoding="utf-8")
        print("Starting worker {} for subjects {}".format(index, subjects), flush=True)
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
        )
        processes.append((index, process, stream, log_path, output))

    failures = []
    for index, process, stream, log_path, output in processes:
        code = process.wait()
        stream.close()
        if code:
            failures.append((index, code, log_path))
        else:
            print("Worker {} completed: {}".format(index, output), flush=True)
    if failures:
        details = "; ".join(
            "worker {} exit {} ({})".format(index, code, log)
            for index, code, log in failures
        )
        raise RuntimeError("Parallel MASSANet run failed: " + details)


def merge_outputs(worker_root, final_output):
    final_output.mkdir(parents=True, exist_ok=True)
    for fold in sorted(worker_root.glob("worker_*/sub_*")):
        metrics = fold / "test_metrics.json"
        if not metrics.is_file():
            continue
        destination = final_output / fold.name
        if destination.exists():
            existing = destination / "test_metrics.json"
            if existing.is_file() and existing.read_bytes() == metrics.read_bytes():
                continue
            raise RuntimeError("Conflicting fold output: {}".format(destination))
        shutil.copytree(str(fold), str(destination))


def write_final_summary(final_output):
    sys.path.insert(0, str(ROOT))
    from baselines.run_massanet import write_summary

    summary = write_summary(final_output, "openbmi", "loso")
    if summary is None:
        raise RuntimeError("No completed folds were found")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--massanet-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(EXPECTED_SUBJECTS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    selected = sorted(set(args.subjects))
    if selected != sorted(args.subjects) or any(subject not in EXPECTED_SUBJECTS for subject in selected):
        parser.error("--subjects must contain unique integers from 1 through 54")
    if args.parallelism < 1 or args.cpu_threads_per_worker < 1:
        parser.error("--parallelism and --cpu-threads-per-worker must be positive")
    if not RUNNER.is_file():
        parser.error("Missing MASSANet runner: {}".format(RUNNER))

    args.output = args.output.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.data_root = args.data_root.resolve()
    args.massanet_root = args.massanet_root.resolve()
    already_done = completed_subjects(args.output)
    remaining = [subject for subject in selected if subject not in already_done]
    print("Completed folds: {}; remaining: {}".format(len(already_done), len(remaining)), flush=True)

    if remaining and not args.skip_preflight:
        preflight(args, remaining[0])
    worker_root = args.output.parent / (args.output.name + "_parallel_workers")
    if remaining:
        groups = split_subjects(remaining, args.parallelism)
        launch_workers(args, groups, worker_root)
        merge_outputs(worker_root, args.output)

    summary = write_final_summary(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if set(selected) == set(EXPECTED_SUBJECTS) and not summary["complete_loso"]:
        raise RuntimeError(
            "Expected 54 completed folds, found {}".format(summary["subject_count"])
        )


if __name__ == "__main__":
    main()
