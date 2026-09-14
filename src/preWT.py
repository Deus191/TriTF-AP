"""Writer for the paper-defined TriTF-AP two-map TF representation."""
import json
from pathlib import Path
import numpy as np
from protocol import labels_to_int
from tf_preprocessing import trial_tf_image


def CWT(data, label, subject, EorT, *, fs=250, output_root=None, wavelet="morl",
        source_window_start_seconds=None):
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
        trial_tf_image(trial, fs, wavelet).save(path)
        records.append(str(path))
    metadata = dict(
        representation="TriTF-AP paper TF image",
        representation_version="tritf_ap_paper_v1",
        fs=fs,
        input_channels=["C3", "Cz", "C4"],
        derivations=["C3-Cz", "C4-Cz"],
        map_count=2,
        canvas_layout="vertical: C3-Cz then C4-Cz",
        frequency_hz=[8, 30],
        frequency_bins=64,
        wavelet=wavelet,
        coefficient_transform="log1p(abs(CWT)**2)",
        normalization="per-trial joint maximum across both maps",
        colormap="jet",
        output_shape=[64, 64, 3],
        trial_samples=int(data.shape[2]),
        trial_duration_seconds=float(data.shape[2] / fs),
        source_window_start_seconds=source_window_start_seconds,
        class_counts=counts,
        note="No class-dependent transform; labels are used only for file grouping.",
    )
    (destination / "preprocessing.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return records
