"""Configured read-only image loading, explicit labels and stable trial IDs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def dataset_config(name, path=None):
    if path:
        config_path = Path(path)
    else:
        configured = ROOT / "config/datasets.json"
        config_path = configured if configured.is_file() else ROOT / "config/datasets.example.json"
    return json.loads(config_path.read_text(encoding="utf-8-sig"))["datasets"][name]


def load_records(records):
    paths = [Path(record[0]).resolve() for record in records]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("{} missing images; first: {}".format(len(missing), missing[:3]))
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate image paths in input records.")
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0)
    if not images:
        raise ValueError("No image records found.")
    return (np.stack(images), np.asarray([r[1] for r in records]),
            np.asarray([r[2] for r in records]), [str(p) for p in paths])


def openbmi_records(subjects, config_path=None, images_root=None):
    config = dataset_config("openbmi", config_path)
    root = images_root or config.get("images_local_path")
    if not root or not Path(root).is_dir():
        raise FileNotFoundError("Configure OpenBMI images_local_path or --images-root.")
    return [(Path(root) / config["image_relative_template"].format(subject=s, label=label, trial=t), c, s)
            for s in subjects for c, label in enumerate(("left", "right"))
            for t in range(config["legacy_trials_per_class"])]


def class_folder_records(root, subjects, split, classes=("left", "right")):
    records = []
    for subject in subjects:
        for label, name in enumerate(classes):
            folder = Path(root) / split / "sub_{}".format(subject) / ("CL" + name)
            paths = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg"))
            if not paths:
                raise FileNotFoundError("No images for subject/class: " + str(folder))
            records.extend((p, label, subject) for p in paths)
    return records
