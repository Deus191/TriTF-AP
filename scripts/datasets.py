"""Dataset references and explicit single-file fetching; never imports training code."""
import argparse
import json
from pathlib import Path
import shutil
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    if path:
        config_path = Path(path)
    else:
        configured = ROOT / "config/datasets.json"
        config_path = configured if configured.is_file() else ROOT / "config/datasets.example.json"
    return json.loads(config_path.read_text(encoding="utf-8-sig"))


def fetch_file(url, destination):
    """Explicit download, exclusive creation, never overwrite an existing file."""
    if urlparse(url).scheme not in ("https", "http", "ftp"):
        raise ValueError("Only HTTP(S)/FTP direct-file URLs are supported.")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Refusing to overwrite: {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    # Keep partial files after a network failure for inspection; never delete user files.
    with partial.open("xb") as output:
        with urlopen(url, timeout=60) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise ValueError("This URL returned a web page, not a dataset file.")
            shutil.copyfileobj(response, output)
    # Exclusive destination creation prevents overwriting, including on POSIX.
    with destination.open("xb") as output, partial.open("rb") as source:
        shutil.copyfileobj(source, output)
    partial.unlink()  # Only this function's completed temporary file.
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", choices=("openbmi", "bci_iv_2b", "bci_ii_iii"))
    parser.add_argument("--check-local", action="store_true", help="Check paths; no downloads or imports")
    parser.add_argument("--check-openbmi-images", action="store_true")
    parser.add_argument("--images-root", type=Path, help="Override OpenBMI image path for the check")
    parser.add_argument("--url", help="Explicit direct-file URL to download, not a landing page")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.url) != bool(args.output):
        parser.error("--url and --output must be used together")
    if args.url:
        print(fetch_file(args.url, args.output))
        return
    config = load_config(args.config)
    selected = config["datasets"]
    if args.dataset:
        selected = {args.dataset: selected[args.dataset]}
    print(json.dumps(selected, indent=2, ensure_ascii=False))
    if args.check_local:
        for name, dataset in selected.items():
            for key in ("raw_local_path", "images_local_path"):
                value = dataset.get(key)
                print("{} {}: {}".format(name, key, "UNCONFIGURED" if not value else
                      ("EXISTS " if Path(value).is_dir() else "MISSING ") + value))
    if args.check_openbmi_images:
        dataset = config["datasets"]["openbmi"]
        image_root = args.images_root or Path(dataset["images_local_path"])
        missing, total = [], 0
        for subject in range(dataset["subjects"][0], dataset["subjects"][1] + 1):
            for label in ("left", "right"):
                for trial in range(dataset["legacy_trials_per_class"]):
                    relative = dataset["image_relative_template"].format(subject=subject, label=label, trial=trial)
                    total += 1
                    if not (image_root / relative).is_file():
                        missing.append(relative)
        print("OpenBMI images: expected={}, missing={}".format(total, len(missing)))
        for relative in missing[:10]:
            print("MISSING " + str(image_root / relative))
        if missing:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
