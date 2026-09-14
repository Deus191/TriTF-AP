"""BCI Competition IV-2b loading and time-frequency image generation.

The dataset itself is not included. Configure its local MAT directory in
`config/datasets.json` or pass `data_root` explicitly.
"""
from pathlib import Path

import numpy as np

import gumpy
import preWT
import preprocessing_3_IIb
from image_data import dataset_config
from tf_preprocessing import apply_ea, fit_ea, normalize_trials

FS = 250
LOWCUT = 8.0
HIGHCUT = 30.0
ANTI_DRIFT = 0.5
NOTCH = 50.0
NOTCH_Q = 30.0


def _resolve_data_root(data_root=None, config_path=None):
    root = data_root or dataset_config("bci_iv_2b", config_path).get("raw_local_path")
    if not root:
        raise ValueError("Configure bci_iv_2b.raw_local_path or pass data_root.")
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def _load_split(data_root, subject, training):
    dataset = gumpy.data.GrazB(str(data_root), "B{:02d}".format(subject))
    loaded = dataset.load(training)
    signal = gumpy.signal.notch(loaded.raw_data, NOTCH, axis=0, fs=FS, Q=NOTCH_Q)
    signal = gumpy.signal.butter_highpass(signal, ANTI_DRIFT, axis=0, fs=FS)
    signal = gumpy.signal.butter_bandpass(signal, LOWCUT, HIGHCUT, axis=0, fs=FS)
    left, right = gumpy.utils.extract_trials2(
        signal,
        loaded.trials,
        loaded.labels,
        loaded.trial_total,
        FS,
        nbClasses=2,
    )
    return normalize_trials(left), normalize_trials(right)


def GetdataET(index, *, data_root=None, config_path=None):
    """Return train-left, train-right, evaluation-left, evaluation-right trials."""
    if index not in range(1, 10):
        raise ValueError("IV-2b subject must be in 1..9")
    root = _resolve_data_root(data_root, config_path)
    train_left, train_right = _load_split(root, index, True)
    eval_left, eval_right = _load_split(root, index, False)
    return train_left, train_right, eval_left, eval_right


def data_norm(data):
    return normalize_trials(data)


def preprocess_ea(data, transform=None):
    if transform is None:
        raise ValueError("Supply an EA transform fitted on the training split using fit_ea().")
    return apply_ea(data, transform)


def GetPrecossedData(sub, output_root=None, *, data_root=None, config_path=None):
    """Generate new class-independent CWT images for one IV-2b subject."""
    train_left, train_right, eval_left, eval_right = GetdataET(
        sub, data_root=data_root, config_path=config_path
    )
    output_root = output_root or Path(__file__).resolve().parents[1] / "outputs/preprocessed_2b"
    train = np.concatenate((train_left, train_right)).transpose(0, 2, 1)
    test = np.concatenate((eval_left, eval_right)).transpose(0, 2, 1)
    train_labels = np.concatenate((np.zeros(len(train_left), dtype=int), np.ones(len(train_right), dtype=int)))
    test_labels = np.concatenate((np.zeros(len(eval_left), dtype=int), np.ones(len(eval_right), dtype=int)))
    # IV-2b already contains three recorded bipolar channels; do not re-reference them.
    return (
        preWT.CWT(train, train_labels, sub, "T", output_root=output_root, bipolar=False),
        preWT.CWT(test, test_labels, sub, "E", output_root=output_root, bipolar=False),
    )


def GetPrecossedData2a(sub, *, data_root, labels_path, output_root):
    """Auxiliary IV-2a writer retained for four-class experiments."""
    train_data, train_labels = preprocessing_3_IIb.GetData(sub, "T", data_root=data_root)
    eval_data, eval_labels = preprocessing_3_IIb.GetData(
        sub, "E", data_root=data_root, labels_path=labels_path
    )
    return (
        preWT.CWT(train_data, train_labels, sub, "T", output_root=output_root),
        preWT.CWT(eval_data, eval_labels, sub, "E", output_root=output_root),
    )

