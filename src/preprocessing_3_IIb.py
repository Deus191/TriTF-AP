"""Corrected auxiliary BCI IV-2a GDF loader (historical filename was misleading).
Evaluation labels must be supplied in a MAT file containing classlabel.
"""
from pathlib import Path
import numpy as np
from tf_preprocessing import normalize_trials, fit_ea, apply_ea


def GetPredata(sub, TorE="T", *, data_root=None, labels_path=None, duration=4.0):
    import mne
    import scipy.io
    if data_root is None or TorE not in ("T", "E"):
        raise ValueError("Explicit 2a data_root and T/E split required.")
    filename = Path(data_root) / "A{:02d}{}.gdf".format(sub, TorE)
    raw = mne.io.read_raw_gdf(str(filename), preload=True, verbose="ERROR")
    raw.rename_channels({name: name[4:] for name in raw.ch_names if name in ("EEG-C3", "EEG-Cz", "EEG-C4")})
    raw.pick(["C3", "Cz", "C4"])
    if not np.isfinite(raw.get_data()).all():
        raise ValueError("Nonfinite EEG; define artifact handling before continuing.")
    fs = float(raw.info["sfreq"])
    if fs != 250:
        raise ValueError("Expected 250 Hz for this 2a loader.")
    raw.filter(8, 30, verbose="ERROR")
    events, mapping = mne.events_from_annotations(raw, verbose="ERROR")
    codes = ("769", "770", "771", "772") if TorE == "T" else ("768",)
    if any(code not in mapping for code in codes):
        raise ValueError("Expected MI cue annotations missing.")
    epochs = mne.Epochs(raw, events, {code: mapping[code] for code in codes},
                        tmin=0, tmax=duration - 1 / fs, baseline=None, preload=True, verbose="ERROR")
    if TorE == "T":
        inverse = {mapping[code]: i for i, code in enumerate(codes)}
        labels = np.asarray([inverse[event] for event in epochs.events[:, -1]])
    else:
        if labels_path is None:
            raise ValueError("Evaluation labels_path is required; cue 768 is NOT a class label.")
        all_labels = scipy.io.loadmat(str(labels_path))["classlabel"].ravel().astype(int) - 1
        # Epoch selection indexes the input events array, so filter it to cues first.
        cue_event_indices = np.flatnonzero(np.isin(events[:, -1], [mapping[code] for code in codes]))
        lookup = {event_index: i for i, event_index in enumerate(cue_event_indices)}
        if len(all_labels) != len(cue_event_indices):
            raise ValueError("Evaluation labels do not match cue count.")
        labels = all_labels[[lookup[index] for index in epochs.selection]]
    return epochs.get_data(), labels


def data_norm(data):
    return normalize_trials(data)


def preprocess_ea(data, transform=None):
    # Explicitly refuse implicit fitting on an evaluation batch.
    if transform is None:
        raise ValueError("Supply an EA transform fitted on the training split.")
    return apply_ea(np.asarray(data).transpose(0, 2, 1), transform).transpose(0, 2, 1)


def GetData(sub, TorE="T", **kwargs):
    data, labels = GetPredata(sub, TorE, **kwargs)
    return normalize_trials(data), labels
