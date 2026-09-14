"""Leakage-safe MASSANet baseline for the paper's IV-2b/OpenBMI protocols.

The upstream MASSANet source is an external runtime dependency. Clone it from
https://github.com/Taowelll/MASSANet and pass its directory with
``--massanet-root``. Dataset files and all outputs remain outside Git.
"""
import argparse
import csv
import hashlib
import importlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt
import torch
import torch.nn.functional as functional
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset


UPSTREAM_URL = "https://github.com/Taowelll/MASSANet"
CHANNELS = ("C3", "Cz", "C4")
SAMPLING_RATE = 250
WINDOW_SECONDS = 4.0
TIME_POINTS = int(SAMPLING_RATE * WINDOW_SECONDS)
LOWCUT = 8.0
HIGHCUT = 30.0
DATASET_SUBJECTS = {
    "bci_iv_2b": tuple(range(1, 10)),
    "openbmi": tuple(range(1, 55)),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_seed(worker_id):
    del worker_id
    value = torch.initial_seed() % (2 ** 32)
    random.seed(value)
    np.random.seed(value)


def labels_to_int(labels, source="dataset"):
    labels = np.asarray(labels).reshape(-1)
    if np.issubdtype(labels.dtype, np.number):
        result = labels.astype(np.int64)
        unique = set(np.unique(result).tolist())
        if unique.issubset({0, 1}):
            return result
        if unique.issubset({1, 2}):
            return result - 1
    aliases = {
        "left": 0,
        "left_hand": 0,
        "lefthand": 0,
        "right": 1,
        "right_hand": 1,
        "righthand": 1,
    }
    normalized = [str(value).strip().lower().replace(" ", "_") for value in labels]
    if all(value in aliases for value in normalized):
        return np.asarray([aliases[value] for value in normalized], dtype=np.int64)
    raise ValueError("Unsupported binary labels in {}: {}".format(source, sorted(set(normalized))))


def validate_trials(data, labels, source="dataset"):
    data = np.asarray(data, dtype=np.float32)
    labels = labels_to_int(labels, source)
    if data.ndim != 3 or data.shape[1:] != (3, TIME_POINTS):
        raise ValueError(
            "Expected data=(N,3,{}) in {}; got {}".format(TIME_POINTS, source, data.shape)
        )
    if len(data) != len(labels) or not len(data):
        raise ValueError("Data/label length mismatch or empty input in {}".format(source))
    if not np.isfinite(data).all() or not np.isfinite(labels).all():
        raise ValueError("Non-finite data in {}".format(source))
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Both classes must be present in {}".format(source))
    return data, labels


def fit_channel_standardizer(data):
    """Fit per-channel mean/std on the fit partition only."""
    data = np.asarray(data, dtype=np.float32)
    mean = data.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = data.std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Cannot fit normalization: a channel has invalid or zero variance")
    return mean, std


def apply_channel_standardizer(data, mean, std):
    data = np.asarray(data, dtype=np.float32)
    standardized = (data - mean) / std
    if not np.isfinite(standardized).all():
        raise ValueError("Non-finite values after train-statistics normalization")
    return standardized.astype(np.float32, copy=False)


def preprocess_iv2b_trials(data):
    """Apply the paper's fixed 8--30 Hz filter to pre-epoched 250 Hz MAT data."""
    data = np.asarray(data, dtype=np.float64)
    coefficients_b, coefficients_a = butter(
        4, [LOWCUT, HIGHCUT], btype="bandpass", fs=SAMPLING_RATE
    )
    filtered = filtfilt(coefficients_b, coefficients_a, data, axis=-1)
    return filtered.astype(np.float32)


def load_iv2b_subject(data_root, subject):
    arrays, labels, ids = [], [], []
    for split in ("T", "E"):
        path = data_root / "B{:02d}{}.mat".format(subject, split)
        if not path.is_file():
            raise FileNotFoundError(path)
        content = scipy.io.loadmat(path)
        if "data" not in content or "label" not in content:
            raise ValueError("MAT file must contain data and label: {}".format(path))
        trial_data, trial_labels = validate_trials(content["data"], content["label"], str(path))
        trial_data = preprocess_iv2b_trials(trial_data)
        arrays.append(trial_data)
        labels.append(trial_labels)
        ids.extend(
            "BCI-IV-2b/B{:02d}{}/{}".format(subject, split, index)
            for index in range(len(trial_data))
        )
    return {
        "x": np.concatenate(arrays),
        "y": np.concatenate(labels),
        "ids": np.asarray(ids),
        "split_lengths": {"T": len(arrays[0]), "E": len(arrays[1])},
    }


def _openbmi_cache_name(subject):
    return "openbmi_paper3ch_8-30hz_250hz_4s_filtered_v2_sub{:02d}.npz".format(subject)


def configure_moabb_upstream(moabb, data_root):
    """Select the original GigaDB files instead of MOABB's automatic NEMAR mirror."""
    set_provider = getattr(moabb, "set_download_provider", None)
    if set_provider is None:
        raise RuntimeError("MOABB >= 1.7 is required to select the OpenBMI upstream provider")
    set_provider("upstream")
    moabb.set_download_dir(str(data_root))


def load_openbmi_subject(data_root, cache_dir, subject):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _openbmi_cache_name(subject)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            data, labels = validate_trials(cached["data"], cached["label"], str(cache_path))
            ids = np.asarray(cached["trial_id"]).astype(str)
            sessions = np.asarray(cached["session"], dtype=np.int8)
        if len(ids) != len(labels) or len(set(ids.tolist())) != len(ids):
            raise ValueError("Invalid cached OpenBMI trial IDs: {}".format(cache_path))
        if len(sessions) != len(labels) or set(np.unique(sessions).tolist()) != {1, 2}:
            raise ValueError("Invalid cached OpenBMI session metadata: {}".format(cache_path))
        return {
            "x": data,
            "y": labels,
            "ids": ids,
            "session": sessions,
            "split_lengths": {
                "session_1": int(np.sum(sessions == 1)),
                "session_2": int(np.sum(sessions == 2)),
                "all": len(data),
            },
        }

    try:
        import moabb
        from moabb.datasets import Lee2019_MI
        from moabb.paradigms import MotorImagery
    except ImportError as error:
        raise RuntimeError(
            "OpenBMI preparation requires MOABB; install requirements-massanet.txt"
        ) from error

    configure_moabb_upstream(moabb, data_root)
    dataset = Lee2019_MI(train_run=True, test_run=False, sessions=[1, 2])
    paradigm = MotorImagery(
        n_classes=2,
        events=["left_hand", "right_hand"],
        fmin=LOWCUT,
        fmax=HIGHCUT,
        tmin=0.0,
        tmax=WINDOW_SECONDS,
        baseline=None,
        channels=list(CHANNELS),
        resample=SAMPLING_RATE,
    )
    data, labels, metadata = paradigm.get_data(
        dataset=dataset, subjects=[subject], return_epochs=False
    )
    data = np.asarray(data)
    if data.ndim != 3 or data.shape[1] != 3 or data.shape[-1] < TIME_POINTS:
        raise ValueError("Unexpected MOABB OpenBMI shape for subject {}: {}".format(subject, data.shape))
    data = np.asarray(data[..., :TIME_POINTS], dtype=np.float32)
    labels = labels_to_int(labels, "OpenBMI subject {}".format(subject))
    session_values = metadata["session"].astype(str).tolist()
    run_values = metadata["run"].astype(str).tolist()
    unique_sessions = sorted(set(session_values))
    if len(unique_sessions) != 2:
        raise ValueError(
            "Expected two labelled OpenBMI sessions for subject {}; got {}".format(
                subject, unique_sessions
            )
        )
    session_mapping = {value: index + 1 for index, value in enumerate(unique_sessions)}
    sessions = np.asarray([session_mapping[value] for value in session_values], dtype=np.int8)
    ids = np.asarray(
        [
            "OpenBMI/S{:02d}/{}/{}/{}".format(subject, session, run, index)
            for index, (session, run) in enumerate(zip(session_values, run_values))
        ]
    )
    data, labels = validate_trials(data, labels, "OpenBMI subject {}".format(subject))
    for session in (1, 2):
        if set(np.unique(labels[sessions == session]).tolist()) != {0, 1}:
            raise ValueError(
                "Both classes must be present in OpenBMI subject {} session {}".format(
                    subject, session
                )
            )
    temporary = cache_path.with_suffix(".part.npz")
    np.savez_compressed(
        temporary, data=data, label=labels, trial_id=ids, session=sessions
    )
    temporary.replace(cache_path)
    return {
        "x": data,
        "y": labels,
        "ids": ids,
        "session": sessions,
        "split_lengths": {
            "session_1": int(np.sum(sessions == 1)),
            "session_2": int(np.sum(sessions == 2)),
            "all": len(data),
        },
    }


def load_all_subjects(args):
    records = {}
    expected = DATASET_SUBJECTS[args.dataset]
    required = expected if args.protocol == "loso" else tuple(args.subjects)
    if args.dataset == "bci_iv_2b":
        for subject in required:
            records[subject] = load_iv2b_subject(args.data_root, subject)
    else:
        cache_dir = args.cache_dir or args.data_root / "massanet_3ch_cache"
        for position, subject in enumerate(required, start=1):
            print(
                "Preparing OpenBMI subject {} ({}/{})".format(subject, position, len(required)),
                flush=True,
            )
            records[subject] = load_openbmi_subject(args.data_root, cache_dir, subject)
    return records


def concatenate_subjects(records, subjects, split=None, session=None):
    arrays, labels, groups, ids = [], [], [], []
    for subject in subjects:
        record = records[subject]
        if session is not None:
            if "session" not in record:
                raise ValueError("Session selection is only available for OpenBMI")
            indices = np.flatnonzero(record["session"] == session)
        elif split is None:
            indices = np.arange(len(record["y"]))
        else:
            start = 0 if split == "T" else record["split_lengths"]["T"]
            stop = record["split_lengths"]["T"] if split == "T" else len(record["y"])
            indices = np.arange(start, stop)
        arrays.append(record["x"][indices])
        labels.append(record["y"][indices])
        groups.extend([subject] * len(indices))
        ids.extend(record["ids"][indices].tolist())
    return (
        np.concatenate(arrays),
        np.concatenate(labels),
        np.asarray(groups),
        np.asarray(ids),
    )


def build_fold(records, dataset, held_out, seed, validation_fraction, protocol="loso"):
    all_subjects = DATASET_SUBJECTS[dataset]
    if protocol == "ho":
        if dataset != "openbmi":
            raise ValueError("The HO protocol is only defined for OpenBMI")
        source_x, source_y, groups, source_ids = concatenate_subjects(
            records, [held_out], session=1
        )
        test_x, test_y, _, test_ids = concatenate_subjects(
            records, [held_out], session=2
        )
        indices = np.arange(len(source_y))
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=validation_fraction, random_state=seed
        )
        fit, validation = next(splitter.split(indices, source_y))
        fit_groups = [held_out]
        validation_groups = [held_out]
        test_scope = "same subject: session 1 fit/validation, session 2 test"
    else:
        source_subjects = [subject for subject in all_subjects if subject != held_out]
        source_x, source_y, groups, source_ids = concatenate_subjects(records, source_subjects)
        if dataset == "bci_iv_2b":
            test_x, test_y, _, test_ids = concatenate_subjects(records, [held_out], split="E")
            test_scope = "held-out subject E split"
        else:
            test_x, test_y, _, test_ids = concatenate_subjects(records, [held_out])
            test_scope = "held-out subject, both labelled sessions"
        indices = np.arange(len(source_y))
        splitter = GroupShuffleSplit(
            n_splits=128, test_size=validation_fraction, random_state=seed
        )
        for fit, validation in splitter.split(indices, source_y, groups):
            if set(source_y[fit]) == {0, 1} and set(source_y[validation]) == {0, 1}:
                break
        else:
            raise ValueError("Could not create a group-disjoint validation split")
        fit_groups = sorted(set(groups[fit].tolist()))
        validation_groups = sorted(set(groups[validation].tolist()))
        if set(fit_groups) & set(validation_groups) or held_out in fit_groups + validation_groups:
            raise AssertionError("Subject leakage detected")
    if set(source_ids.tolist()) & set(test_ids.tolist()):
        raise AssertionError("Trial leakage detected")
    mean, std = fit_channel_standardizer(source_x[fit])
    return {
        "fit_x": apply_channel_standardizer(source_x[fit], mean, std),
        "fit_y": source_y[fit],
        "validation_x": apply_channel_standardizer(source_x[validation], mean, std),
        "validation_y": source_y[validation],
        "test_x": apply_channel_standardizer(test_x, mean, std),
        "test_y": test_y,
        "fit_ids": source_ids[fit],
        "validation_ids": source_ids[validation],
        "test_ids": test_ids,
        "fit_subjects": fit_groups,
        "validation_subjects": validation_groups,
        "test_scope": test_scope,
        "normalization_mean": mean.reshape(-1),
        "normalization_std": std.reshape(-1),
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def external_source_info(root):
    model_path = root / "model" / "MASSANet.py"
    layers_path = root / "model" / "layers.py"
    missing = [str(path) for path in (model_path, layers_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Invalid MASSANet root; missing {}".format(missing))
    commit = None
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "url": UPSTREAM_URL,
        "root": str(root.resolve()),
        "git_commit": commit,
        "massanet_py_sha256": file_sha256(model_path),
        "layers_py_sha256": file_sha256(layers_path),
    }


def load_external_model(root):
    source = external_source_info(root)
    root_string = str(root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    existing = sys.modules.get("model")
    if existing is not None:
        location = Path(getattr(existing, "__file__", "") or "").resolve()
        if root.resolve() not in location.parents:
            raise RuntimeError("A different top-level 'model' package is already imported: {}".format(location))
    module = importlib.import_module("model.MASSANet")
    location = Path(module.__file__).resolve()
    if root.resolve() not in location.parents:
        raise RuntimeError("MASSANet import resolved outside --massanet-root: {}".format(location))
    return module.Net, source


class SignalDataset(Dataset):
    def __init__(self, data, labels, augmentation="none", cutcat_ratio=10):
        self.data = np.asarray(data, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.augmentation = augmentation
        self.cutcat_ratio = cutcat_ratio
        self.opposite = {
            label: np.flatnonzero(self.labels != label) for label in (0, 1)
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = self.data[index]
        label = int(self.labels[index])
        if self.augmentation == "none":
            return torch.from_numpy(data[None, ...]), torch.tensor(label, dtype=torch.long)
        negative = int(np.random.choice(self.opposite[label]))
        other = self.data[negative]
        maximum = max(1, data.shape[-1] // self.cutcat_ratio)
        length = int(np.random.randint(1, maximum + 1))
        center = int(np.random.randint(data.shape[-1]))
        start = max(0, center - length // 2)
        stop = min(data.shape[-1], max(start + 1, center + (length + 1) // 2))
        mixed = data.copy()
        mixed[:, start:stop] = other[:, start:stop]
        fraction = (stop - start) / data.shape[-1]
        soft = np.zeros(2, dtype=np.float32)
        soft[label] = 1.0 - fraction
        soft[1 - label] = fraction
        return torch.from_numpy(mixed[None, ...]), torch.from_numpy(soft)


class SmoothedNLLLoss(nn.Module):
    def __init__(self, classes=2, smoothing=0.05):
        super().__init__()
        self.classes = classes
        self.smoothing = smoothing

    def forward(self, log_probabilities, target):
        if target.ndim == 1:
            target = functional.one_hot(target.long(), num_classes=self.classes).float()
        else:
            target = target.float()
        if self.smoothing:
            target = target * (1.0 - self.smoothing) + self.smoothing / self.classes
        return -(target * log_probabilities).sum(dim=1).mean()


def make_loader(data, labels, batch_size, shuffle, augmentation, workers, seed, pin_memory):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        SignalDataset(data, labels, augmentation=augmentation),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_seed if workers else None,
        generator=generator,
    )


def hard_labels(labels):
    return labels.long() if labels.ndim == 1 else labels.argmax(dim=1).long()


def evaluate(model, loader, device, loss_function=None):
    model.eval()
    predictions, targets = [], []
    loss_total, item_total = 0.0, 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(features)
            if loss_function is not None:
                loss_total += float(loss_function(output, labels).item()) * len(labels)
                item_total += len(labels)
            predictions.extend(output.argmax(1).cpu().numpy().tolist())
            targets.extend(hard_labels(labels).cpu().numpy().tolist())
    return (
        np.asarray(targets),
        np.asarray(predictions),
        loss_total / item_total if item_total else None,
    )


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError("Not JSON serializable: {}".format(type(value).__name__))


def train_fold(args, records, model_class, source_info, held_out):
    fold = build_fold(
        records, args.dataset, held_out, args.seed, args.validation_fraction, args.protocol
    )
    fold_dir = args.output / "sub_{:02d}".format(held_out)
    if fold_dir.exists():
        config_path = fold_dir / "config.json"
        if args.resume and config_path.is_file():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous.get("status") == "complete":
                if (
                    previous.get("dataset") != args.dataset
                    or previous.get("protocol") != args.protocol
                ):
                    raise ValueError(
                        "Existing completed fold uses a different dataset/protocol: {}".format(
                            fold_dir
                        )
                    )
                print("Skipping completed subject {}".format(held_out), flush=True)
                return
        raise FileExistsError("Incomplete/existing fold directory: {}".format(fold_dir))
    fold_dir.mkdir(parents=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = model_class(
        num_classes=2,
        num_channels=3,
        sampling_rate=SAMPLING_RATE,
        dropout=args.dropout,
    ).to(device)
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    config = vars(args).copy()
    config.update(
        {
            "data_root": str(args.data_root.resolve()),
            "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else None,
            "massanet_root": str(args.massanet_root.resolve()),
            "output": str(args.output.resolve()),
            "held_out_subject": held_out,
            "fit_subjects": fold["fit_subjects"],
            "validation_subjects": fold["validation_subjects"],
            "test_subjects": [held_out],
            "test_scope": fold["test_scope"],
            "fit_count": len(fold["fit_y"]),
            "validation_count": len(fold["validation_y"]),
            "test_count": len(fold["test_y"]),
            "parameters": parameters,
            "input_contract": {
                "channels": list(CHANNELS),
                "sampling_rate_hz": SAMPLING_RATE,
                "window_seconds": WINDOW_SECONDS,
                "time_points": TIME_POINTS,
                "bandpass_hz": [LOWCUT, HIGHCUT],
                "normalization": "per-channel z-score fitted on fit partition only",
            },
            "normalization_statistics": {
                "channel_mean": fold["normalization_mean"].tolist(),
                "channel_std": fold["normalization_std"].tolist(),
                "fitted_on": "fit partition only",
            },
            "external_source": source_info,
            "status": "started",
        }
    )
    config_path = fold_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, default=json_ready), encoding="utf-8")
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

    pin_memory = device.type == "cuda"
    fit_loader = make_loader(
        fold["fit_x"], fold["fit_y"], args.batch_size, True,
        args.augmentation, args.workers, args.seed + held_out, pin_memory,
    )
    validation_loader = make_loader(
        fold["validation_x"], fold["validation_y"], args.batch_size, False,
        "none", args.workers, args.seed + held_out, pin_memory,
    )
    test_loader = make_loader(
        fold["test_x"], fold["test_y"], args.batch_size, False,
        "none", args.workers, args.seed + held_out, pin_memory,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_function = SmoothedNLLLoss(classes=2, smoothing=args.label_smoothing)

    best_value = float("inf") if args.selection_metric == "validation_loss" else -float("inf")
    stale = 0
    history_path = fold_dir / "history.csv"
    with history_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["epoch", "learning_rate", "train_loss", "validation_loss", "validation_accuracy"]
        )
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss, train_items = 0.0, 0
            for features, labels in fit_loader:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(features), labels)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item()) * len(labels)
                train_items += len(labels)
            truth, predicted, validation_loss = evaluate(
                model, validation_loader, device, loss_function
            )
            validation_accuracy = float(accuracy_score(truth, predicted))
            writer.writerow(
                [
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    train_loss / train_items,
                    validation_loss,
                    validation_accuracy,
                ]
            )
            stream.flush()
            candidate = validation_loss if args.selection_metric == "validation_loss" else validation_accuracy
            improved = candidate < best_value if args.selection_metric == "validation_loss" else candidate > best_value
            if improved:
                best_value = candidate
                stale = 0
                torch.save(
                    {"model_state_dict": model.state_dict(), "epoch": epoch, "selection_value": candidate},
                    fold_dir / "best.pth",
                )
            else:
                stale += 1
            scheduler.step()
            print(
                "subject={} epoch={} val_loss={:.5f} val_acc={:.4f}".format(
                    held_out, epoch, validation_loss, validation_accuracy
                ),
                flush=True,
            )
            if args.patience and stale >= args.patience:
                break

    checkpoint = torch.load(fold_dir / "best.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    truth, predicted, _ = evaluate(model, test_loader, device)
    metrics = {
        "subject": held_out,
        "protocol": args.protocol,
        "accuracy": float(accuracy_score(truth, predicted)),
        "kappa": float(cohen_kappa_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "n_test": int(len(truth)),
        "parameters": parameters,
        "best_epoch": int(checkpoint["epoch"]),
        "selection_metric": args.selection_metric,
        "selection_value": float(checkpoint["selection_value"]),
    }
    with (fold_dir / "test_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial_id", "y_true", "y_pred"])
        writer.writerows(zip(fold["test_ids"].tolist(), truth.tolist(), predicted.tolist()))
    (fold_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    config["status"] = "complete"
    config_path.write_text(json.dumps(config, indent=2, default=json_ready), encoding="utf-8")
    print("Completed subject {}: {}".format(held_out, metrics), flush=True)


def write_summary(output, dataset, protocol):
    rows = []
    for subject in DATASET_SUBJECTS[dataset]:
        path = output / "sub_{:02d}".format(subject) / "test_metrics.json"
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("protocol") != protocol:
                raise ValueError(
                    "Output directory mixes protocols at {}: expected {}, got {}".format(
                        path, protocol, row.get("protocol")
                    )
                )
            rows.append(row)
    if not rows:
        return None
    metric_names = ("accuracy", "kappa", "macro_f1")
    aggregate = {
        "dataset": dataset,
        "protocol": protocol,
        "completed_subjects": [row["subject"] for row in rows],
        "expected_subjects": list(DATASET_SUBJECTS[dataset]),
        "complete_protocol": len(rows) == len(DATASET_SUBJECTS[dataset]),
        "complete_loso": protocol == "loso" and len(rows) == len(DATASET_SUBJECTS[dataset]),
        "subject_count": len(rows),
    }
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=float)
        aggregate[name + "_mean"] = float(values.mean())
        aggregate[name + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_SUBJECTS))
    parser.add_argument(
        "--protocol", choices=("loso", "ho"), default="loso",
        help="LOSO for both datasets; HO is OpenBMI session 1 -> session 2",
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--massanet-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, help="OpenBMI 3-channel preprocessed NPZ cache")
    parser.add_argument("--subjects", type=int, nargs="+", help="Evaluation subjects; default is the full dataset")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.075)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--augmentation", choices=("cutcat", "none"), default="cutcat")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--selection-metric", choices=("validation_loss", "validation_accuracy"), default="validation_loss")
    parser.add_argument("--patience", type=int, default=30, help="0 trains all epochs")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="Examples: cpu, cuda, cuda:0")
    parser.add_argument("--resume", action="store_true", help="Skip folds already marked complete")
    parser.add_argument("--check-only", action="store_true", help="Validate data/model and do one forward pass")
    return parser


def validate_arguments(parser, args):
    expected = DATASET_SUBJECTS[args.dataset]
    args.subjects = args.subjects or list(expected)
    if args.protocol == "ho" and args.dataset != "openbmi":
        parser.error("--protocol ho is only supported for OpenBMI")
    if len(set(args.subjects)) != len(args.subjects) or any(subject not in expected for subject in args.subjects):
        parser.error("Held-out subjects must be unique and inside {}..{}".format(expected[0], expected[-1]))
    if not args.data_root.is_dir():
        parser.error("Data root does not exist: {}".format(args.data_root))
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0 or args.workers < 0:
        parser.error("Epochs/batch-size must be positive; patience/workers cannot be negative")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in [0, 1)")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    seed_everything(args.seed)
    model_class, source_info = load_external_model(args.massanet_root)
    records = load_all_subjects(args)
    total_trials = sum(len(record["y"]) for record in records.values())
    print(
        "Validated {} subjects and {} trials for {}".format(
            len(records), total_trials, args.dataset
        ),
        flush=True,
    )
    sample_subject = args.subjects[0]
    sample_fold = build_fold(
        records,
        args.dataset,
        sample_subject,
        args.seed,
        args.validation_fraction,
        args.protocol,
    )
    sample = torch.from_numpy(sample_fold["fit_x"][:2, None])
    model = model_class(2, 3, SAMPLING_RATE, args.dropout)
    with torch.no_grad():
        output = model(sample)
    if tuple(output.shape) != (2, 2) or not torch.isfinite(output).all():
        raise RuntimeError("MASSANet forward smoke test failed: {}".format(tuple(output.shape)))
    print("MASSANet forward smoke test passed; parameters={}".format(sum(p.numel() for p in model.parameters())))
    print(
        "{} split smoke test passed for subject {}: fit={}, validation={}, test={}".format(
            args.protocol.upper(),
            sample_subject,
            len(sample_fold["fit_y"]),
            len(sample_fold["validation_y"]),
            len(sample_fold["test_y"]),
        ),
        flush=True,
    )
    del model, sample, sample_fold
    if args.check_only:
        return
    if args.output.exists() and not args.resume:
        parser.error("Output exists; choose a new path or pass --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    for held_out in args.subjects:
        seed_everything(args.seed + held_out)
        train_fold(args, records, model_class, source_info, held_out)
        summary = write_summary(args.output, args.dataset, args.protocol)
        print("Current summary: {}".format(summary), flush=True)


if __name__ == "__main__":
    main()
