"""Run one non-MASSANet OpenBMI LOSO baseline with parallel fold workers."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "baselines" / "train_loso.py"
EXPECTED_SUBJECTS = tuple(range(1, 55))
MODELS = ("dmsa", "eeg_inception", "ctnet", "eegnet", "deepconvnet", "fbcsp_svm")


def split_subjects(subjects, worker_count):
    worker_count = min(max(1, worker_count), len(subjects))
    groups = [[] for _ in range(worker_count)]
    for index, subject in enumerate(subjects):
        groups[index % worker_count].append(subject)
    return [group for group in groups if group]


def completed_subjects(output, model):
    completed = set()
    if not output.is_dir():
        return completed
    for metrics_path in output.glob("sub_*/test_metrics.json"):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            metrics.get("dataset") == "openbmi"
            and metrics.get("protocol") == "loso"
            and metrics.get("model") == model
            and metrics.get("subject") in EXPECTED_SUBJECTS
        ):
            completed.add(int(metrics["subject"]))
    return completed


def common_command(args):
    command = [
        str(args.python), str(RUNNER),
        "--model", args.model,
        "--dataset", "openbmi",
        "--protocol", "loso",
        "--data-root", str(args.data_root),
        "--cache-dir", str(args.cache_dir),
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
    ]
    if args.dropout is not None:
        command.extend(["--dropout", str(args.dropout)])
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
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", str(args.cpu_threads_per_worker))
    environment.setdefault("MKL_NUM_THREADS", str(args.cpu_threads_per_worker))
    for index, subjects in enumerate(groups, start=1):
        output = worker_root / "worker_{:02d}".format(index)
        output.mkdir(parents=True, exist_ok=True)
        log_path = worker_root / "worker_{:02d}.log".format(index)
        command = common_command(args) + ["--output", str(output), "--subjects"]
        command.extend(str(subject) for subject in subjects)
        command.append("--resume")
        stream = log_path.open("a", encoding="utf-8")
        print("Starting worker {} for subjects {}".format(index, subjects), flush=True)
        process = subprocess.Popen(
            command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT,
            env=environment
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
        raise RuntimeError(
            "; ".join(
                "worker {} exit {} ({})".format(index, code, log)
                for index, code, log in failures
            )
        )


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


def write_final_summary(output, model):
    sys.path.insert(0, str(ROOT))
    from baselines.train_loso import write_summary

    summary = write_summary(output, "openbmi", "loso", model)
    if summary is None:
        raise RuntimeError("No completed folds were found")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(EXPECTED_SUBJECTS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-preflight", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    selected = sorted(set(args.subjects))
    if selected != sorted(args.subjects) or any(s not in EXPECTED_SUBJECTS for s in selected):
        parser.error("--subjects must contain unique integers from 1 through 54")
    if args.parallelism < 1 or args.cpu_threads_per_worker < 1:
        parser.error("Parallelism and CPU threads must be positive")
    if args.model == "fbcsp_svm" and args.device != "cpu":
        print("FBCSP-SVM runs on CPU; overriding --device to cpu", flush=True)
        args.device = "cpu"
    args.output = args.output.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.data_root = args.data_root.resolve()
    already_done = completed_subjects(args.output, args.model)
    remaining = [subject for subject in selected if subject not in already_done]
    print("Completed folds: {}; remaining: {}".format(len(already_done), len(remaining)), flush=True)
    if remaining and not args.skip_preflight:
        preflight(args, remaining[0])
    worker_root = args.output.parent / (args.output.name + "_parallel_workers")
    if remaining:
        launch_workers(args, split_subjects(remaining, args.parallelism), worker_root)
        merge_outputs(worker_root, args.output)
    summary = write_final_summary(args.output, args.model)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if set(selected) == set(EXPECTED_SUBJECTS) and not summary["complete_protocol"]:
        raise RuntimeError("Expected 54 completed folds, found {}".format(summary["subject_count"]))


if __name__ == "__main__":
    main()
