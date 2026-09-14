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

    def test_normalization_is_fitted_on_fit_partition_only(self):
        fit = np.stack(
            [
                np.full((3, 1000), -1.0, dtype=np.float32),
                np.full((3, 1000), 1.0, dtype=np.float32),
            ]
        )
        test = np.full((1, 3, 1000), 10.0, dtype=np.float32)
        mean, std = RUNNER.fit_channel_standardizer(fit)
        normalized_fit = RUNNER.apply_channel_standardizer(fit, mean, std)
        normalized_test = RUNNER.apply_channel_standardizer(test, mean, std)
        np.testing.assert_allclose(mean, 0.0)
        np.testing.assert_allclose(std, 1.0)
        np.testing.assert_allclose(normalized_fit.mean(axis=(0, 2)), 0.0)
        np.testing.assert_allclose(normalized_test, 10.0)

    def test_moabb_provider_is_forced_to_upstream(self):
        calls = []

        class FakeMOABB:
            @staticmethod
            def set_download_provider(provider):
                calls.append(("provider", provider))

            @staticmethod
            def set_download_dir(path):
                calls.append(("directory", path))

        RUNNER.configure_moabb_upstream(FakeMOABB, Path("dataset-root"))
        self.assertEqual(calls[0], ("provider", "upstream"))
        self.assertEqual(calls[1][0], "directory")

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
            fold = RUNNER.build_fold(records, "bci_iv_2b", 1, 42, 0.2, "loso")
            self.assertEqual(len(fold["test_y"]), 4)
            self.assertFalse(set(fold["fit_subjects"]) & set(fold["validation_subjects"]))
            self.assertNotIn(1, fold["fit_subjects"])
            self.assertNotIn(1, fold["validation_subjects"])
            self.assertTrue(all("B01E" in trial_id for trial_id in fold["test_ids"]))

    def test_openbmi_ho_uses_session_one_for_fit_and_session_two_for_test(self):
        rng = np.random.default_rng(6)
        labels = np.tile([0, 1], 8)
        sessions = np.repeat([1, 2], 8)
        record = {
            "x": rng.normal(size=(16, 3, 1000)).astype(np.float32),
            "y": labels,
            "ids": np.asarray(
                ["OpenBMI/S01/{}/{}".format(session, index) for index, session in enumerate(sessions)]
            ),
            "session": sessions,
            "split_lengths": {"session_1": 8, "session_2": 8, "all": 16},
        }
        fold = RUNNER.build_fold({1: record}, "openbmi", 1, 42, 0.25, "ho")
        self.assertEqual(len(fold["fit_y"]), 6)
        self.assertEqual(len(fold["validation_y"]), 2)
        self.assertEqual(len(fold["test_y"]), 8)
        self.assertTrue(all("/1/" in trial_id for trial_id in fold["fit_ids"]))
        self.assertTrue(all("/1/" in trial_id for trial_id in fold["validation_ids"]))
        self.assertTrue(all("/2/" in trial_id for trial_id in fold["test_ids"]))
        self.assertFalse(set(fold["fit_ids"]) & set(fold["validation_ids"]))
        self.assertFalse(set(fold["fit_ids"]) & set(fold["test_ids"]))

    def test_openbmi_loso_keeps_held_out_subject_out_of_training(self):
        rng = np.random.default_rng(7)
        records = {}
        for subject in range(1, 55):
            sessions = np.repeat([1, 2], 4)
            labels = np.tile([0, 1], 4)
            records[subject] = {
                "x": rng.normal(size=(8, 3, 1000)).astype(np.float32),
                "y": labels,
                "ids": np.asarray(
                    [
                        "OpenBMI/S{:02d}/{}/{}".format(subject, session, index)
                        for index, session in enumerate(sessions)
                    ]
                ),
                "session": sessions,
                "split_lengths": {"session_1": 4, "session_2": 4, "all": 8},
            }
        fold = RUNNER.build_fold(records, "openbmi", 1, 42, 0.2, "loso")
        self.assertEqual(len(fold["test_y"]), 8)
        self.assertNotIn(1, fold["fit_subjects"])
        self.assertNotIn(1, fold["validation_subjects"])
        self.assertFalse(set(fold["fit_subjects"]) & set(fold["validation_subjects"]))
        self.assertTrue(all("OpenBMI/S01/" in trial_id for trial_id in fold["test_ids"]))

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
