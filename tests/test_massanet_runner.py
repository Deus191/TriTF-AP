import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.io

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
RUNNER = None
if torch is not None:
    SPEC = importlib.util.spec_from_file_location(
        "massanet_runner", ROOT / "baselines" / "run_massanet.py"
    )
    RUNNER = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(RUNNER)


@unittest.skipUnless(torch is not None, "PyTorch baseline dependencies are optional")
class MASSANetRunnerTests(unittest.TestCase):
    def test_label_mapping(self):
        np.testing.assert_array_equal(RUNNER.labels_to_int([1, 2, 1]), [0, 1, 0])
        np.testing.assert_array_equal(
            RUNNER.labels_to_int(["left_hand", "right_hand"]), [0, 1]
        )

    def test_paper_preprocessing_contract(self):
        rng = np.random.default_rng(4)
        data = rng.normal(size=(4, 3, 1000))
        processed = RUNNER.preprocess_iv2b_trials(data)
        self.assertEqual(processed.shape, (4, 3, 1000))
        self.assertEqual(processed.dtype, np.float32)
        np.testing.assert_allclose(np.max(np.abs(processed), axis=(1, 2)), 1.0, atol=1e-6)

    def test_iv2b_loader_and_leakage_safe_fold(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory(prefix="massanet_iv2b_") as directory:
            root = Path(directory)
            for subject in range(1, 10):
                for split in ("T", "E"):
                    data = rng.normal(size=(4, 3, 1000))
                    labels = np.asarray([[1], [2], [1], [2]])
                    scipy.io.savemat(root / "B{:02d}{}.mat".format(subject, split), {"data": data, "label": labels})
            records = {
                subject: RUNNER.load_iv2b_subject(root, subject) for subject in range(1, 10)
            }
            fold = RUNNER.build_fold(records, "bci_iv_2b", 1, 42, 0.2)
            self.assertEqual(len(fold["test_y"]), 4)
            self.assertFalse(set(fold["fit_subjects"]) & set(fold["validation_subjects"]))
            self.assertNotIn(1, fold["fit_subjects"])
            self.assertNotIn(1, fold["validation_subjects"])
            self.assertTrue(all("B01E" in trial_id for trial_id in fold["test_ids"]))

    def test_cutcat_only_uses_fit_dataset(self):
        data = np.zeros((4, 3, 1000), dtype=np.float32)
        data[1] = 1.0
        data[3] = 1.0
        labels = np.asarray([0, 1, 0, 1])
        dataset = RUNNER.SignalDataset(data, labels, augmentation="cutcat")
        mixed, soft = dataset[0]
        self.assertEqual(tuple(mixed.shape), (1, 3, 1000))
        self.assertEqual(tuple(soft.shape), (2,))
        self.assertAlmostEqual(float(soft.sum()), 1.0)
        self.assertGreater(float(mixed.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
