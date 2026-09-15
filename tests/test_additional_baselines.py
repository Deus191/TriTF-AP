import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "baselines"
sys.path.insert(0, str(BASELINES))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FBCSP = load_module("fbcsp_baseline", BASELINES / "fbcsp.py")

try:
    import torch
except ImportError:
    torch = None


class FBCSPTests(unittest.TestCase):
    def test_synthetic_spatial_frequency_problem(self):
        rng = np.random.default_rng(12)
        time = np.arange(500) / 250.0
        labels = np.repeat([0, 1], 16)
        data = rng.normal(0.0, 0.2, size=(32, 3, 500))
        data[labels == 0, 0] += np.sin(2 * np.pi * 10 * time)
        data[labels == 1, 2] += np.sin(2 * np.pi * 22 * time)
        model = FBCSP.FBCSPSVM(seed=9)
        model.fit(data, labels)
        predicted = model.predict(data)
        self.assertGreaterEqual(float(np.mean(predicted == labels)), 0.95)
        self.assertEqual(model.feature_count, 12)

    def test_invalid_component_count_is_rejected(self):
        data = np.ones((4, 3, 500))
        labels = np.asarray([0, 0, 1, 1])
        with self.assertRaises(ValueError):
            FBCSP.CSPBank(components_per_side=2).fit(data, labels)


@unittest.skipUnless(torch is not None, "PyTorch baseline dependencies are optional")
class NeuralBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_module("unified_baseline_runner", BASELINES / "train_loso.py")

    def test_all_neural_models_accept_three_channel_windows(self):
        sample = torch.randn(2, 1, 3, 1000)
        for name in self.runner.NEURAL_MODELS:
            with self.subTest(model=name):
                model = self.runner.build_model(name, 3, 1000, 2, 250)
                model.eval()
                with torch.no_grad():
                    output = model(sample)
                self.assertEqual(tuple(output.shape), (2, 2))
                self.assertTrue(bool(torch.isfinite(output).all()))

    def test_new_models_have_fixed_parameter_counts(self):
        expected = {"eegnet": 2874, "deepconvnet": 268177}
        for name, count in expected.items():
            model = self.runner.build_model(name, 3, 1000, 2, 250)
            self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), count)


if __name__ == "__main__":
    unittest.main()
