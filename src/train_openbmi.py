"""Corrected OpenBMI LOSO entry: final target never used for model selection."""
import argparse
from pathlib import Path
from image_data import ROOT, dataset_config, load_records, openbmi_records


def TrainOpenbmi(k, full_path, *, config_path=None, images_root=None, seed=42, augment=False, **kwargs):
    config = dataset_config("openbmi", config_path)
    subjects = list(range(config["subjects"][0], config["subjects"][1] + 1))
    if k not in subjects:
        raise ValueError("Held-out subject outside configured range.")
    source = openbmi_records([s for s in subjects if s != k], config_path, images_root)
    test = openbmi_records([k], config_path, images_root)
    missing = [str(r[0]) for r in source + test if not Path(r[0]).is_file()]
    if missing:
        raise FileNotFoundError("Missing {} images; first {}".format(len(missing), missing[:3]))
    x, y, groups, ids = load_records(source)
    xt, yt, _, test_ids = load_records(test)
    from Hope import Train
    return Train(x, y, xt, yt, groups=groups, source_ids=ids, test_ids=test_ids,
                 output_dir=Path(full_path) / "sub_{}".format(k), seed=seed, augment=augment,
                 dataset_name="OpenBMI", protocol_name="LOSO", **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/openbmi")
    parser.add_argument("--subjects", nargs="+", type=int, help="Default: all configured subjects, 1-54")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--vae-epochs", type=int, default=100)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = dataset_config("openbmi", args.config)
    subjects = args.subjects or list(range(config["subjects"][0], config["subjects"][1] + 1))
    if len(set(subjects)) != len(subjects):
        parser.error("Duplicate held-out subjects.")
    if args.check_only:
        records = openbmi_records(range(config["subjects"][0], config["subjects"][1] + 1), args.config, args.images_root)
        missing = [str(r[0]) for r in records if not Path(r[0]).is_file()]
        print("Expected:", len(records), "Missing:", len(missing), missing[:3])
        raise SystemExit(2 if missing else 0)
    for subject in subjects:
        TrainOpenbmi(subject, args.output, config_path=args.config, images_root=args.images_root,
                     seed=args.seed, augment=args.augment, epochs=args.epochs, vae_epochs=args.vae_epochs)


if __name__ == "__main__":
    main()
