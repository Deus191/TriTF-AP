"""Configurable LOSO runner for raw-signal PyTorch baselines.

Input MAT files must contain `data=(trials, channels, time)` and `label`.
Generated logs, checkpoints, predictions, and metrics belong under `outputs/`,
which the repository excludes from Git.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import scipy.io
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ctnet import CTNet
from dmsa import DMSANet
from eeg_inception import EEGInception


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_mat(path):
    content = scipy.io.loadmat(path)
    if "data" not in content or "label" not in content:
        raise ValueError("MAT file must contain data and label: {}".format(path))
    data = np.asarray(content["data"], dtype=np.float32)
    labels = np.asarray(content["label"]).reshape(-1).astype(np.int64)
    if data.ndim != 3 or len(data) != len(labels):
        raise ValueError("Expected data=(N,C,T) aligned with labels: {}".format(path))
    unique = set(np.unique(labels).tolist())
    if unique.issubset({0, 1}):
        pass
    elif unique.issubset({1, 2}):
        labels = labels - 1
    else:
        raise ValueError("Only binary labels encoded as 0/1 or 1/2 are supported: {}".format(path))
    return data, labels


def subject_path(root, subject, split, processed):
    suffix = "_processed" if processed else ""
    return root / "B{:02d}{}{}.mat".format(subject, split, suffix)


def load_subjects(root, subjects, splits, processed):
    all_data, all_labels, all_groups, all_ids = [], [], [], []
    for subject in subjects:
        for split in splits:
            path = subject_path(root, subject, split, processed)
            data, labels = load_mat(path)
            all_data.append(data)
            all_labels.append(labels)
            all_groups.extend([subject] * len(data))
            all_ids.extend(["B{:02d}{}/{}".format(subject, split, index) for index in range(len(data))])
    return (np.concatenate(all_data), np.concatenate(all_labels),
            np.asarray(all_groups), np.asarray(all_ids))


def split_source(labels, groups, seed, validation_fraction):
    indices = np.arange(len(labels))
    splitter = GroupShuffleSplit(n_splits=64, test_size=validation_fraction, random_state=seed)
    for fit, validation in splitter.split(indices, labels, groups):
        if set(labels[fit]) == set(labels) and set(labels[validation]) == set(labels):
            return fit, validation
    raise ValueError("Could not create a group-disjoint validation split covering every class")


def build_model(name, channels, time_points, classes, sampling_rate):
    if name == "dmsa":
        return DMSANet(nChan=channels, nTime=time_points, nClass=classes)
    if name == "eeg_inception":
        duration_ms = round(time_points / sampling_rate * 1000)
        return EEGInception(input_time=duration_ms, fs=sampling_rate, ncha=channels, n_classes=classes)
    if name == "ctnet":
        return CTNet(n_channels=channels, n_time=time_points, n_classes=classes)
    raise ValueError("Unknown model: " + name)


def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for features, labels in loader:
            output = model(features.to(device).unsqueeze(1))
            predictions.extend(output.argmax(1).cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())
    return np.asarray(targets), np.asarray(predictions)


def train_fold(args, held_out):
    subjects = list(range(1, 10))
    source_subjects = [subject for subject in subjects if subject != held_out]
    source_splits = tuple(args.source_splits)
    test_splits = tuple(args.test_splits)
    source_x, source_y, groups, source_ids = load_subjects(
        args.data_root, source_subjects, source_splits, args.processed
    )
    test_x, test_y, _, test_ids = load_subjects(
        args.data_root, [held_out], test_splits, args.processed
    )
    fit, validation = split_source(source_y, groups, args.seed, args.validation_fraction)

    fold_dir = args.output / "sub_{:02d}".format(held_out)
    fold_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update({
        "data_root": str(args.data_root),
        "output": str(args.output),
        "held_out_subject": held_out,
        "source_subjects": source_subjects,
        "source_splits": list(source_splits),
        "test_splits": list(test_splits),
        "status": "started",
    })
    (fold_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (fold_dir / "splits.json").write_text(json.dumps({
        "fit_ids": source_ids[fit].tolist(),
        "validation_ids": source_ids[validation].tolist(),
        "test_ids": test_ids.tolist(),
    }, indent=2), encoding="utf-8")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(args.model, source_x.shape[1], source_x.shape[2], len(np.unique(source_y)), args.sampling_rate)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.NLLLoss()
    fit_loader = DataLoader(TensorDataset(torch.from_numpy(source_x[fit]), torch.from_numpy(source_y[fit])),
                            batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(source_x[validation]), torch.from_numpy(source_y[validation])),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)),
                             batch_size=args.batch_size)

    best_loss, stale = float("inf"), 0
    history_path = fold_dir / "history.csv"
    with history_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "validation_loss", "validation_accuracy"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, total_items = 0.0, 0
            for features, labels in fit_loader:
                features, labels = features.to(device).unsqueeze(1), labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(features), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
                total_items += len(labels)
            model.eval()
            validation_loss, validation_items = 0.0, 0
            with torch.no_grad():
                for features, labels in validation_loader:
                    features, labels = features.to(device).unsqueeze(1), labels.to(device)
                    loss = loss_fn(model(features), labels)
                    validation_loss += loss.item() * len(labels)
                    validation_items += len(labels)
            truth, predicted = evaluate(model, validation_loader, device)
            validation_loss /= validation_items
            writer.writerow([epoch, total_loss / total_items, validation_loss, accuracy_score(truth, predicted)])
            stream.flush()
            if validation_loss < best_loss:
                best_loss, stale = validation_loss, 0
                torch.save(model.state_dict(), fold_dir / "best.pth")
            else:
                stale += 1
                if stale >= args.patience:
                    break

    model.load_state_dict(torch.load(fold_dir / "best.pth", map_location=device, weights_only=True))
    truth, predicted = evaluate(model, test_loader, device)
    metrics = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "kappa": float(cohen_kappa_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "n_test": int(len(truth)),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    with (fold_dir / "test_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial_id", "y_true", "y_pred"])
        writer.writerows(zip(test_ids.tolist(), truth.tolist(), predicted.tolist()))
    (fold_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config["status"] = "complete"
    (fold_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("subject", held_out, metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("dmsa", "eeg_inception", "ctnet"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--source-splits", choices=("T", "E"), nargs="+", default=("T", "E"))
    parser.add_argument("--test-splits", choices=("T", "E"), nargs="+", default=("E",))
    parser.add_argument("--processed", action="store_true", help="Read *_processed.mat names")
    parser.add_argument("--sampling-rate", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="Examples: cpu, cuda, cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output path already exists; choose a new directory")
    if any(subject not in range(1, 10) for subject in args.subjects):
        parser.error("Subjects must be in 1..9")
    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    for subject in args.subjects:
        train_fold(args, subject)


if __name__ == "__main__":
    main()
