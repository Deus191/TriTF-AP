"""Repaired image entries. VAE fitting is delegated AFTER source split."""
import argparse
from pathlib import Path
from image_data import class_folder_records, dataset_config, load_records
from train_openbmi import TrainOpenbmi


def run_image_subject(k, full_path, images_root, *, dataset, protocol="subject_dependent",
                      augment=True, seed=42, classes=("left", "right"), **kwargs):
    if protocol == "subject_dependent":
        train_subjects = [k]
    elif protocol == "LOSO":
        train_subjects = [s for s in range(1, 10) if s != k]
    else:
        raise ValueError("Protocol must be subject_dependent or LOSO.")
    if k not in range(1, 10):
        raise ValueError("Subject must lie in 1..9.")
    source = class_folder_records(images_root, train_subjects, "T", classes)
    if protocol == "LOSO":
        source += class_folder_records(images_root, train_subjects, "E", classes)
    test = class_folder_records(images_root, [k], "E", classes)
    x, y, groups, ids = load_records(source)
    xt, yt, _, test_ids = load_records(test)
    from Hope import Train
    return Train(x, y, xt, yt, source_ids=ids, test_ids=test_ids,
                 groups=groups if protocol == "LOSO" else None,
                 output_dir=Path(full_path) / "sub_{}".format(k), seed=seed,
                 augment=augment, protocol_name=protocol, dataset_name=dataset, **kwargs)


def Train2b(k, full_path, *, config_path=None, images_root=None, **kwargs):
    root = images_root or dataset_config("bci_iv_2b", config_path).get("images_local_path")
    if not root:
        raise ValueError("Configure bci_iv_2b.images_local_path.")
    return run_image_subject(k, full_path, root, dataset="BCI_IV_2b", **kwargs)


def Train2a(sub, full_path, *, images_root=None, protocol="LOSO", **kwargs):
    if images_root is None:
        raise ValueError("2a is auxiliary four-class data; provide its explicit image root.")
    return run_image_subject(sub, full_path, images_root, dataset="BCI_IV_2a_auxiliary",
                             protocol=protocol, classes=("left", "right", "foot", "tone"), **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("2b", "2a"))
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--protocol", choices=("subject_dependent", "LOSO"), default="subject_dependent")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--vae-epochs", type=int, default=100)
    args = parser.parse_args()
    for subject in args.subjects:
        options = dict(images_root=args.images_root, protocol=args.protocol, augment=not args.no_augment,
                       seed=args.seed, epochs=args.epochs, vae_epochs=args.vae_epochs)
        if args.dataset == "2b":
            Train2b(subject, args.output, config_path=args.config, **options)
        else:
            Train2a(subject, args.output, **options)


if __name__ == "__main__":
    main()
