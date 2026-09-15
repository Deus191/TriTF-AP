"""Unified, leakage-safe runner for the repository's non-MASSANet baselines.

The runner uses the same IV-2b/OpenBMI loading and evaluation partitions as
``run_massanet.py``. Generated logs, checkpoints, predictions, and summaries
belong below the user-selected output directory and are excluded from Git.
"""
import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASELINE_DIR = Path(__file__).resolve().parent
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from ctnet import CTNet  # noqa: E402
from deepconvnet import DeepConvNet  # noqa: E402
from dmsa import DMSANet  # noqa: E402
from eeg_inception import EEGInception  # noqa: E402
from eegnet import EEGNet  # noqa: E402
from fbcsp import FBCSPSVM  # noqa: E402
from run_massanet import (  # noqa: E402
    CHANNELS,
    DATASET_SUBJECTS,
    SAMPLING_RATE,
    TIME_POINTS,
    build_fold,
    load_all_subjects,
)


NEURAL_MODELS = ("dmsa", "eeg_inception", "ctnet", "eegnet", "deepconvnet")
ALL_MODELS = NEURAL_MODELS + ("fbcsp_svm",)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(name, channels, time_points, classes, sampling_rate, dropout=None):
    """Construct one neural baseline using its default dropout unless overridden."""
    if name == "dmsa":
        kwargs = {"dropoutP": dropout} if dropout is not None else {}
        return DMSANet(nChan=channels, nTime=time_points, nClass=classes, **kwargs)
    if name == "eeg_inception":
        duration_ms = round(time_points / sampling_rate * 1000)
        kwargs = {"dropout_rate": dropout} if dropout is not None else {}
        return EEGInception(
            input_time=duration_ms, fs=sampling_rate, ncha=channels,
            n_classes=classes, **kwargs
        )
    if name == "ctnet":
        kwargs = {"dropout": dropout} if dropout is not None else {}
        return CTNet(n_channels=channels, n_time=time_points, n_classes=classes, **kwargs)
    if name == "eegnet":
        kwargs = {"dropout": dropout} if dropout is not None else {}
        return EEGNet(
            n_channels=channels, n_time=time_points, n_classes=classes,
            sampling_rate=sampling_rate, **kwargs
        )
    if name == "deepconvnet":
        kwargs = {"dropout": dropout} if dropout is not None else {}
        return DeepConvNet(
            n_channels=channels, n_time=time_points, n_classes=classes, **kwargs
        )
    raise ValueError("Not a neural model: {}".format(name))


def make_loader(data, labels, batch_size, shuffle, seed, pin_memory):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(data), torch.from_numpy(labels)),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        generator=generator,
    )


def evaluate_neural(model, loader, device, loss_function=None):
    model.eval()
    predictions, targets = [], []
    loss_total, item_total = 0.0, 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True).unsqueeze(1)
            labels = labels.to(device, non_blocking=True)
            output = model(features)
            if loss_function is not None:
                loss_total += float(loss_function(output, labels).item()) * len(labels)
                item_total += len(labels)
            predictions.extend(output.argmax(1).cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
    loss = loss_total / item_total if item_total else None
    return np.asarray(targets), np.asarray(predictions), loss


def metric_values(truth, predicted):
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "kappa": float(cohen_kappa_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "n_test": int(len(truth)),
    }


def prepare_fold_directory(args, fold, held_out):
    fold_dir = args.output / "sub_{:02d}".format(held_out)
    if fold_dir.exists():
        metrics_path = fold_dir / "test_metrics.json"
        config_path = fold_dir / "config.json"
        if args.resume and metrics_path.is_file() and config_path.is_file():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if (
                previous.get("status") == "complete"
                and previous.get("dataset") == args.dataset
                and previous.get("protocol") == args.protocol
                and previous.get("model") == args.model
            ):
                print("Skipping completed subject {}".format(held_out), flush=True)
                return None, None
        raise FileExistsError("Incomplete or incompatible fold directory: {}".format(fold_dir))
    fold_dir.mkdir(parents=True)
    config = {
        "model": args.model,
        "dataset": args.dataset,
        "protocol": args.protocol,
        "data_root": str(args.data_root.resolve()),
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else None,
        "output": str(args.output.resolve()),
        "held_out_subject": held_out,
        "fit_subjects": fold["fit_subjects"],
        "validation_subjects": fold["validation_subjects"],
        "test_subjects": [held_out],
        "test_scope": fold["test_scope"],
        "fit_count": len(fold["fit_y"]),
        "validation_count": len(fold["validation_y"]),
        "test_count": len(fold["test_y"]),
        "input_contract": {
            "channels": list(CHANNELS),
            "sampling_rate_hz": SAMPLING_RATE,
            "time_points": TIME_POINTS,
            "bandpass_hz": [8.0, 30.0],
            "normalization": "per-channel z-score fitted on fit partition only",
        },
        "normalization_statistics": {
            "channel_mean": fold["normalization_mean"].tolist(),
            "channel_std": fold["normalization_std"].tolist(),
            "fitted_on": "fit partition only",
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "dropout_override": args.dropout,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
        },
        "fbcsp": {
            "components_per_side": args.csp_components_per_side,
            "selected_features": args.fbcsp_features,
            "svm_c": args.svm_c,
            "svm_kernel": args.svm_kernel,
        } if args.model == "fbcsp_svm" else None,
        "status": "started",
    }
    config_path = fold_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    (fold_dir / "splits.json").write_text(
        json.dumps(
            {
                "fit_ids": fold["fit_ids"].tolist(),
                "validation_ids": fold["validation_ids"].tolist(),
                "test_ids": fold["test_ids"].tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return fold_dir, config


def train_neural_fold(args, fold, fold_dir, held_out):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = build_model(
        args.model, fold["fit_x"].shape[1], fold["fit_x"].shape[2],
        2, SAMPLING_RATE, args.dropout
    ).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    pin_memory = device.type == "cuda"
    fit_loader = make_loader(
        fold["fit_x"], fold["fit_y"], args.batch_size, True,
        args.seed + held_out, pin_memory
    )
    validation_loader = make_loader(
        fold["validation_x"], fold["validation_y"], args.batch_size, False,
        args.seed + held_out, pin_memory
    )
    test_loader = make_loader(
        fold["test_x"], fold["test_y"], args.batch_size, False,
        args.seed + held_out, pin_memory
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.NLLLoss()
    best_loss, stale = float("inf"), 0
    history_path = fold_dir / "history.csv"
    with history_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "validation_loss", "validation_accuracy"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_total, train_items = 0.0, 0
            for features, labels in fit_loader:
                features = features.to(device, non_blocking=True).unsqueeze(1)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(features), labels)
                loss.backward()
                optimizer.step()
                train_total += float(loss.item()) * len(labels)
                train_items += len(labels)
            truth, predicted, validation_loss = evaluate_neural(
                model, validation_loader, device, loss_function
            )
            validation_accuracy = float(accuracy_score(truth, predicted))
            writer.writerow(
                [epoch, train_total / train_items, validation_loss, validation_accuracy]
            )
            stream.flush()
            if validation_loss < best_loss:
                best_loss, stale = validation_loss, 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "validation_loss": validation_loss,
                    },
                    fold_dir / "best.pth",
                )
            else:
                stale += 1
            print(
                "subject={} epoch={} val_loss={:.5f} val_acc={:.4f}".format(
                    held_out, epoch, validation_loss, validation_accuracy
                ),
                flush=True,
            )
            if args.patience and stale >= args.patience:
                break
    try:
        checkpoint = torch.load(fold_dir / "best.pth", map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility.
        checkpoint = torch.load(fold_dir / "best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    truth, predicted, _ = evaluate_neural(model, test_loader, device)
    extra = {
        "parameters": parameter_count,
        "best_epoch": int(checkpoint["epoch"]),
        "selection_metric": "validation_loss",
        "selection_value": float(checkpoint["validation_loss"]),
    }
    return truth, predicted, extra


def train_fbcsp_fold(args, fold, fold_dir):
    model = FBCSPSVM(
        sampling_rate=SAMPLING_RATE,
        components_per_side=args.csp_components_per_side,
        selected_features=args.fbcsp_features,
        svm_c=args.svm_c,
        svm_kernel=args.svm_kernel,
        seed=args.seed,
    )
    model.fit(fold["fit_x"], fold["fit_y"])
    validation_prediction = model.predict(fold["validation_x"])
    validation_accuracy = float(accuracy_score(fold["validation_y"], validation_prediction))
    with (fold_dir / "history.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "validation_accuracy"])
        writer.writerow(["fit", validation_accuracy])
    with (fold_dir / "best.pkl").open("xb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    predicted = model.predict(fold["test_x"])
    extra = {
        "parameters": None,
        "parameter_note": "N/A: CSP filters and SVM support vectors are data-dependent, not neural trainable parameters",
        "selected_feature_count": int(min(args.fbcsp_features, model.feature_count)),
        "validation_accuracy": validation_accuracy,
        "selection_metric": "not_applicable_single_fit",
    }
    return fold["test_y"], predicted, extra


def train_fold(args, records, held_out):
    fold = build_fold(
        records, args.dataset, held_out, args.seed + held_out,
        args.validation_fraction, args.protocol
    )
    fold_dir, config = prepare_fold_directory(args, fold, held_out)
    if fold_dir is None:
        return
    if args.model == "fbcsp_svm":
        truth, predicted, extra = train_fbcsp_fold(args, fold, fold_dir)
    else:
        truth, predicted, extra = train_neural_fold(args, fold, fold_dir, held_out)
    metrics = {
        "subject": held_out,
        "dataset": args.dataset,
        "protocol": args.protocol,
        "model": args.model,
        **metric_values(truth, predicted),
        **extra,
    }
    with (fold_dir / "test_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial_id", "y_true", "y_pred"])
        writer.writerows(zip(fold["test_ids"].tolist(), truth.tolist(), predicted.tolist()))
    (fold_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    config["status"] = "complete"
    (fold_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Completed subject {}: {}".format(held_out, metrics), flush=True)


def write_summary(output, dataset, protocol, model):
    rows = []
    for subject in DATASET_SUBJECTS[dataset]:
        path = output / "sub_{:02d}".format(subject) / "test_metrics.json"
        if not path.is_file():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        expected = (dataset, protocol, model)
        actual = (row.get("dataset"), row.get("protocol"), row.get("model"))
        if actual != expected:
            raise ValueError("Output mixes experiment configurations at {}".format(path))
        rows.append(row)
    if not rows:
        return None
    aggregate = {
        "dataset": dataset,
        "protocol": protocol,
        "model": model,
        "completed_subjects": [row["subject"] for row in rows],
        "expected_subjects": list(DATASET_SUBJECTS[dataset]),
        "complete_protocol": len(rows) == len(DATASET_SUBJECTS[dataset]),
        "subject_count": len(rows),
    }
    for name in ("accuracy", "kappa", "macro_f1"):
        values = np.asarray([row[name] for row in rows], dtype=float)
        aggregate[name + "_mean"] = float(values.mean())
        aggregate[name + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--dataset", choices=tuple(DATASET_SUBJECTS), default="bci_iv_2b")
    parser.add_argument(
        "--protocol", choices=("loso", "ho"), default="loso",
        help="LOSO for both datasets; HO is OpenBMI session 1 -> session 2"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="OpenBMI preprocessed cache")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", type=int, help="Default: every dataset subject")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10, help="0 trains every epoch")
    parser.add_argument("--dropout", type=float, help="Override each neural model's default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="Examples: cpu, cuda, cuda:0")
    parser.add_argument("--resume", action="store_true", help="Skip completed folds")
    parser.add_argument("--check-only", action="store_true", help="Validate data, split, and model")
    parser.add_argument("--csp-components-per-side", type=int, default=1)
    parser.add_argument("--fbcsp-features", type=int, default=8)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-kernel", choices=("linear", "rbf"), default="linear")
    return parser


def validate_arguments(parser, args):
    expected = DATASET_SUBJECTS[args.dataset]
    args.subjects = args.subjects or list(expected)
    if args.protocol == "ho" and args.dataset != "openbmi":
        parser.error("--protocol ho is only supported for OpenBMI")
    if len(set(args.subjects)) != len(args.subjects) or any(s not in expected for s in args.subjects):
        parser.error("Subjects must be unique and inside {}..{}".format(expected[0], expected[-1]))
    if not args.data_root.is_dir():
        parser.error("Data root does not exist: {}".format(args.data_root))
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        parser.error("Epochs/batch-size must be positive; patience cannot be negative")
    if args.dropout is not None and not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if args.csp_components_per_side < 1 or args.fbcsp_features < 1 or args.svm_c <= 0:
        parser.error("FBCSP component/feature counts and SVM C must be positive")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    seed_everything(args.seed)
    records = load_all_subjects(args)
    total_trials = sum(len(record["y"]) for record in records.values())
    print(
        "Validated {} subjects and {} trials for {}".format(
            len(records), total_trials, args.dataset
        ),
        flush=True,
    )
    sample_fold = build_fold(
        records, args.dataset, args.subjects[0], args.seed + args.subjects[0],
        args.validation_fraction, args.protocol
    )
    if args.model == "fbcsp_svm":
        if 2 * args.csp_components_per_side > sample_fold["fit_x"].shape[1]:
            parser.error("CSP selects too many components for the three-channel input")
        print("FBCSP split validation passed", flush=True)
    else:
        model = build_model(
            args.model, sample_fold["fit_x"].shape[1], sample_fold["fit_x"].shape[2],
            2, SAMPLING_RATE, args.dropout
        )
        model.eval()
        with torch.no_grad():
            output = model(torch.from_numpy(sample_fold["fit_x"][:2, None]))
        if tuple(output.shape) != (2, 2) or not torch.isfinite(output).all():
            raise RuntimeError("Model forward smoke test failed: {}".format(tuple(output.shape)))
        print(
            "Model smoke test passed; parameters={}".format(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            flush=True,
        )
    if args.check_only:
        return
    if args.output.exists() and not args.resume:
        parser.error("Output exists; choose a new path or pass --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    for held_out in args.subjects:
        seed_everything(args.seed + held_out)
        train_fold(args, records, held_out)
        summary = write_summary(args.output, args.dataset, args.protocol, args.model)
        print("Current summary: {}".format(summary), flush=True)


if __name__ == "__main__":
    main()
