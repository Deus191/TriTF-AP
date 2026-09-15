import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARALLEL = load_module("massanet_parallel", "scripts/run_massanet_openbmi_parallel.py")
STATS = load_module("paired_statistics", "scripts/paired_statistics.py")


class ParallelLauncherTests(unittest.TestCase):
    def test_round_robin_partition_is_complete_and_unique(self):
        groups = PARALLEL.split_subjects(list(range(1, 11)), 4)
        flattened = [subject for group in groups for subject in group]
        self.assertEqual(sorted(flattened), list(range(1, 11)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(groups[0], [1, 5, 9])

    def test_completed_subjects_requires_loso_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            good = output / "sub_01"
            bad = output / "sub_02"
            good.mkdir()
            bad.mkdir()
            (good / "test_metrics.json").write_text(
                json.dumps({"subject": 1, "protocol": "loso"}), encoding="utf-8"
            )
            (bad / "test_metrics.json").write_text(
                json.dumps({"subject": 2, "protocol": "ho"}), encoding="utf-8"
            )
            self.assertEqual(PARALLEL.completed_subjects(output), {1})


class PairedStatisticsTests(unittest.TestCase):
    def test_holm_adjustment_is_monotone_in_rank_order(self):
        adjusted = STATS.holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_subject_parser_accepts_common_labels(self):
        self.assertEqual(STATS.subject_number("sub_12"), 12)
        self.assertEqual(STATS.subject_number("B09"), 9)


if __name__ == "__main__":
    unittest.main()
