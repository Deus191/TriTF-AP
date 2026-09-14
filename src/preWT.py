"""Corrected common TF writer. Output is NEW data, not the historical TF images."""
import json
from pathlib import Path
import numpy as np
from protocol import labels_to_int
from tf_preprocessing import trial_tf_image


def CWT(data, label, subject, EorT, *, fs=250, output_root=None, wavelet="morl", bipolar=True):
    data = np.asarray(data)
    labels, _ = labels_to_int(label, num_classes=4)
    if data.ndim != 3 or data.shape[1] != 3 or len(data) != len(labels):
        raise ValueError("Expected data=(N,3,T), labels=(N,) with zero-based classes.")
    if EorT not in ("T", "E"):
        raise ValueError("Split must be T or E.")
    output_root = Path(output_root) if output_root else Path(__file__).resolve().parents[1] / "outputs/preprocessed"
    destination = output_root / EorT / "sub_{}".format(subject)
    destination.mkdir(parents=True, exist_ok=False)  # Never overwrite prior images.
    names = ["left", "right", "foot", "tone"]
    if labels.max() >= len(names):
        raise ValueError("Only two/four-class MI supported.")
    counts = {name: 0 for name in names}
    records = []
    for trial, cl in zip(data, labels):
        name = names[cl]
        index = counts[name]
        counts[name] += 1
        folder = destination / ("CL" + name)
        folder.mkdir(exist_ok=True)
        path = folder / "cl{} ch3_{}_tr{}.bmp".format(name, subject, index)
        trial_tf_image(trial, fs, wavelet, bipolar).save(path)
        records.append(str(path))
    metadata = dict(fs=fs, channels=["C3", "Cz", "C4"], bipolar=bipolar,
                    frequency_hz=[8, 30], frequency_bins=64, wavelet=wavelet,
                    representation_version="repaired_v1", class_counts=counts,
                    note="No class-dependent transform; labels used only for file grouping.")
    (destination / "preprocessing.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return records
