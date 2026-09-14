"""Offline tests of NEW tooling only; never import the archived research code."""
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dataset_tools", ROOT / "scripts/datasets.py")
DATASETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATASETS)


class Response(io.BytesIO):
    def __init__(self, data, content_type="application/octet-stream"):
        super().__init__(data)
        self.headers = {"Content-Type": content_type}


class DatasetToolTests(unittest.TestCase):
    def test_original_openbmi_filename_mapping(self):
        dataset = DATASETS.load_config()["datasets"]["openbmi"]
        self.assertEqual(dataset["image_relative_template"].format(subject=6, label="left", trial=0),
                         "sub6/CLleft_3ch/clleft STFT_6_ch3_6_tr0.bmp")
        self.assertEqual(dataset["historical_file_url_template"].format(session=1, subject=6),
                         "ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542/session1/s6/sess01_subj06_EEG_MI.mat")

    def test_download_mocked_bytes(self):
        with tempfile.TemporaryDirectory(prefix="icassp_dataset_test_") as directory:
            output = Path(directory) / "record.mat"
            with patch.object(DATASETS, "urlopen", return_value=Response(b"test-mat-bytes")):
                DATASETS.fetch_file("https://example.invalid/record.mat", output)
            self.assertEqual(output.read_bytes(), b"test-mat-bytes")
            self.assertFalse(output.with_name("record.mat.part").exists())

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="icassp_dataset_test_") as directory:
            output = Path(directory) / "existing.mat"
            output.write_bytes(b"keep-original")
            with patch.object(DATASETS, "urlopen") as request:
                with self.assertRaises(FileExistsError):
                    DATASETS.fetch_file("https://example.invalid/record.mat", output)
                request.assert_not_called()
            self.assertEqual(output.read_bytes(), b"keep-original")

    def test_landing_page_is_not_saved_as_data(self):
        with tempfile.TemporaryDirectory(prefix="icassp_dataset_test_") as directory:
            output = Path(directory) / "record.mat"
            with patch.object(DATASETS, "urlopen", return_value=Response(b"<html>page</html>", "text/html")):
                with self.assertRaises(ValueError):
                    DATASETS.fetch_file("https://example.invalid/page", output)
            self.assertFalse(output.exists())

    def test_disallowed_scheme(self):
        with self.assertRaises(ValueError):
            DATASETS.fetch_file("file:///outside/file", "unused.mat")

    def test_existing_partial_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="icassp_dataset_test_") as directory:
            output = Path(directory) / "record.mat"
            partial = output.with_name("record.mat.part")
            partial.write_bytes(b"keep-partial")
            with patch.object(DATASETS, "urlopen") as request:
                with self.assertRaises(FileExistsError):
                    DATASETS.fetch_file("https://example.invalid/record.mat", output)
                request.assert_not_called()
            self.assertEqual(partial.read_bytes(), b"keep-partial")


if __name__ == "__main__":
    unittest.main()
