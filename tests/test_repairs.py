"""Regression tests for repaired scientific dataflow and deployment equivalence."""
import ast
import csv
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")
from protocol import labels_to_int, split_source, check_ids, validate_images
from tf_preprocessing import cwt_band, normalize_trials, trial_tf_image, fit_ea, apply_ea


class ProtocolTests(unittest.TestCase):
    def test_subject_disjoint_validation(self):
        groups = np.repeat(np.arange(5), 8)
        y = np.tile([0, 1], 20)
        train, val = split_source(y, groups=groups)
        self.assertFalse(set(groups[train]) & set(groups[val]))
        self.assertEqual(set(train) | set(val), set(range(40)))

    def test_repeatable_stratified_split(self):
        y = np.tile([0, 1], 20)
        first = split_source(y, seed=7)
        second = split_source(y, seed=7)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(set(y[a]), {0, 1})

    def test_invalid_onehot_rejected(self):
        for y in ([[0, 0], [0, 1]], [[1, 1], [1, 0]]):
            with self.assertRaises(ValueError):
                labels_to_int(y)

    def test_four_classes_not_silently_binary(self):
        y, count = labels_to_int(np.eye(4))
        self.assertEqual(count, 4)
        np.testing.assert_array_equal(y, [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            labels_to_int(np.eye(4), num_classes=2)

    def test_overlapping_and_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            check_ids(['trial1'], ['trial1'], 1, 1)
        with self.assertRaises(ValueError):
            check_ids(['trial1', 'trial1'], ['test'], 2, 1)

    def test_bad_images_rejected(self):
        with self.assertRaises(ValueError):
            validate_images(np.zeros((2, 3, 1000)), [0, 1])
        with self.assertRaises(ValueError):
            validate_images(np.full((2, 64, 64, 3), np.nan), [0, 1])


class PreprocessingTests(unittest.TestCase):
    def test_frequency_grid_and_sinusoid(self):
        fs = 250
        signal = np.sin(2 * np.pi * 12 * np.arange(1000) / fs)
        magnitude, frequencies = cwt_band(signal, fs)
        self.assertAlmostEqual(frequencies.min(), 8)
        self.assertAlmostEqual(frequencies.max(), 30)
        peak = frequencies[magnitude[:, 100:-100].mean(axis=1).argmax()]
        self.assertLess(abs(peak - 12), 1)

    def test_zero_normalization_and_tf(self):
        np.testing.assert_array_equal(normalize_trials(np.zeros((2, 3, 100))), 0)
        image = trial_tf_image(np.zeros((3, 1000)), 250)
        self.assertEqual(image.size, (64, 64))
        self.assertEqual(image.mode, "RGB")

    def test_label_cannot_control_transform(self):
        import inspect
        self.assertNotIn('label', inspect.signature(trial_tf_image).parameters)
        import preWT
        trial = np.random.default_rng(1).normal(size=(3, 300))
        with tempfile.TemporaryDirectory(prefix="icassp_tf_test_") as tmp:
            paths = preWT.CWT(np.stack((trial, trial)), [0, 1], 1, 'T', output_root=tmp)
            self.assertEqual(Path(paths[0]).read_bytes(), Path(paths[1]).read_bytes())

    def test_2a_distinct_train_test_dispatch(self):
        import preprocessing_2b as prep
        train, test = np.ones((2, 3, 100)), np.zeros((2, 3, 100))
        labels = np.array([0, 1])
        with patch.object(prep.preprocessing_3_IIb, 'GetData', side_effect=[(train, labels), (test, labels)]), \
             patch.object(prep.preWT, 'CWT', side_effect=['train_paths', 'test_paths']) as writer:
            result = prep.GetPrecossedData2a(1, data_root='unused', labels_path='unused', output_root='unused')
        self.assertIs(writer.call_args_list[0].args[0], train)
        self.assertIs(writer.call_args_list[1].args[0], test)
        self.assertEqual(result, ('train_paths', 'test_paths'))

    def test_ea_transform_reused_without_test_fit(self):
        x = np.random.default_rng(2).normal(size=(5, 100, 3))
        transform = fit_ea(x)
        self.assertEqual(apply_ea(x, transform).shape, x.shape)
        with self.assertRaises(ValueError):
            import preprocessing_2b
            preprocessing_2b.preprocess_ea(x)

    def test_filters_receive_sample_rate(self):
        import utilss
        source = type('Data', (), {'load': lambda self, _: self, 'raw_data': np.zeros((5000, 3)),
                                   'trials': np.array([0, 2000]), 'labels': np.array([0, 1]), 'trial_total': 8})()
        with patch.object(utilss.gumpy.signal, 'notch', return_value=source.raw_data) as notch, \
             patch.object(utilss.gumpy.signal, 'butter_highpass', return_value=source.raw_data) as high, \
             patch.object(utilss.gumpy.signal, 'butter_bandpass', return_value=source.raw_data) as band:
            utilss.load_preprocess_data(source, True, 8, 30, 0.4, 30, 0.5, 2, 50, 0, 250, True)
        for method in (notch, high, band):
            self.assertEqual(method.call_args.kwargs['fs'], 250)


class EntryAndSummaryTests(unittest.TestCase):
    def test_openbmi_target_excluded_from_source(self):
        from train_openbmi import TrainOpenbmi
        from PIL import Image
        config = {'datasets': {'openbmi': {'subjects': [1, 3], 'legacy_trials_per_class': 2,
                  'image_relative_template': 'sub{subject}/{label}_{trial}.bmp'}}}
        with tempfile.TemporaryDirectory(prefix='icassp_entry_test_') as tmp:
            root = Path(tmp)
            config['datasets']['openbmi']['images_local_path'] = str(root)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            for subject in range(1, 4):
                (root / ('sub'+str(subject))).mkdir()
                for label in ('left', 'right'):
                    for trial in range(2):
                        Image.new('RGB', (64,64)).save(root / ('sub'+str(subject)) / (label+'_'+str(trial)+'.bmp'))
            with patch('Hope.Train', return_value='model') as training:
                self.assertEqual(TrainOpenbmi(2, root / 'output', config_path=config_path), 'model')
            self.assertEqual(set(training.call_args.kwargs['groups']), {1,3})
            self.assertEqual(training.call_args.args[0].shape[0], 8)
            self.assertEqual(training.call_args.args[2].shape[0], 4)
            self.assertFalse(set(training.call_args.kwargs['source_ids']) & set(training.call_args.kwargs['test_ids']))

    def test_summary_reads_final_metrics_not_history_max(self):
        from count import summarize
        with tempfile.TemporaryDirectory(prefix='icassp_summary_test_') as tmp:
            root = Path(tmp)
            run = root / 'sub_1'
            run.mkdir()
            (run / 'config.json').write_text(json.dumps(dict(status='complete', dataset='d', protocol='p', model='m', seed=42, augment=False)))
            (run / 'test_metrics.json').write_text(json.dumps(dict(accuracy=0.6, kappa=0.2, n_test=20)))
            (run / 'history.csv').write_text('epoch,val_accuracy\n0,0.99\n1,0.80\n')
            result = summarize(root, root / 'summary.csv')
            self.assertEqual(result['macro_accuracy'], 0.6)
            with self.assertRaises(FileExistsError):
                summarize(root, root / 'summary.csv')

    def test_legacy_bulk_download_cannot_delete_files(self):
        import utilss
        with patch.object(utilss.os, 'remove') as removal:
            with self.assertRaises(RuntimeError):
                utilss.load_raw('OpenBMI')
            removal.assert_not_called()


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        cls.tf = tf
        tf.keras.utils.set_random_seed(42)

    def test_fusion_with_nontrivial_bn_and_gates(self):
        from paper_model import build_classifier, export_deploy
        model = build_classifier()
        rng = np.random.default_rng(4)
        for layer in [model.get_layer('stem_bn'), model.get_layer('rep_bn')] + model.get_layer('rep_dw').norms:
            shape = layer.get_weights()[0].shape
            layer.set_weights([rng.normal(size=shape).astype('float32'), rng.normal(size=shape).astype('float32'),
                               rng.normal(size=shape).astype('float32'), rng.uniform(0.2, 2, size=shape).astype('float32')])
        model.get_layer('rep_dw').gates.assign([0.5, -0.4, 1.2, 0.8])
        x = rng.random((3, 64, 64, 3), dtype=np.float32)
        deploy = export_deploy(model)
        np.testing.assert_allclose(model(x, training=False), deploy(x, training=False), atol=1e-5, rtol=1e-4)
        self.assertLess(deploy.count_params(), model.count_params())
        self.assertFalse(any(isinstance(layer, self.tf.keras.layers.BatchNormalization) for layer in deploy.layers))

    def test_roundtrip_and_four_class_forward(self):
        from paper_model import build_classifier, export_deploy
        model = build_classifier(num_classes=4)
        x = np.random.default_rng(5).random((2, 64, 64, 3), dtype=np.float32)
        self.assertEqual(model(x).shape, (2, 4))
        with tempfile.TemporaryDirectory(prefix="icassp_model_test_") as tmp:
            for name, graph in [('train', model), ('deploy', export_deploy(model))]:
                path = Path(tmp) / (name + '.keras')
                graph.save(path)
                restored = self.tf.keras.models.load_model(path)
                np.testing.assert_allclose(graph(x, training=False), restored(x, training=False), atol=1e-6)


class TrainingSmokeTests(unittest.TestCase):
    def test_actual_training_and_final_test_artifacts(self):
        from Hope import Train
        from paper_model import build_classifier
        rng = np.random.default_rng(6)
        x = rng.random((20, 64, 64, 3), dtype=np.float32)
        y = np.tile([0, 1], 10)
        xt = rng.random((4, 64, 64, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory(prefix="icassp_train_test_") as tmp:
            output = Path(tmp) / 'run'
            model = Train(x, y, xt, [0, 1, 0, 1], epochs=1, batch_size=4, augment=True, vae_epochs=1,
                          output_dir=output, source_ids=['s'+str(i) for i in range(20)],
                          test_ids=['t'+str(i) for i in range(4)])
            self.assertEqual(model.output_shape[-1], 2)
            config = json.loads((output / 'config.json').read_text())
            self.assertEqual(config['status'], 'complete')
            self.assertEqual(config['fit_original_count'], 16)
            self.assertEqual(config['validation_count'], 4)
            self.assertEqual(config['test_count'], 4)
            self.assertEqual(config['fit_total_count'], 32)
            self.assertTrue((output / 'vae_encoder.keras').is_file())
            self.assertTrue((output / 'vae_decoder.keras').is_file())
            self.assertTrue((output / 'classifier_deploy.keras').is_file())
            with (output / 'test_predictions.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)

    def test_vae_only_receives_fit_partition(self):
        from Hope import Train
        import augmentation
        rng = np.random.default_rng(7)
        x = rng.random((20, 64, 64, 3), dtype=np.float32)
        y = np.tile([0, 1], 10)
        train, val = split_source(y)
        observed = []
        def fake_vae(fit_x, fit_y, **kwargs):
            observed.append(fit_x.copy())
            return np.concatenate((fit_x, fit_x)), np.concatenate((fit_y, fit_y)), np.arange(len(fit_x))
        with tempfile.TemporaryDirectory(prefix="icassp_split_test_") as tmp, \
             patch.object(augmentation, 'augment_training', side_effect=fake_vae):
            output = Path(tmp) / 'run'
            Train(x, y, x[:4].copy() * 0.5, [0, 1, 0, 1], output_dir=output,
                  epochs=1, batch_size=8, augment=True)
            config = json.loads((output / 'config.json').read_text())
            self.assertEqual(config['fit_total_count'], 32)
            self.assertEqual(config['validation_count'], 4)
        np.testing.assert_array_equal(observed[0], x[train])

    def test_actual_vae_single_epoch(self):
        from augmentation import augment_training
        x = np.random.default_rng(8).random((4, 64, 64, 3), dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        augmented, labels, origins = augment_training(x, y, epochs=1, batch_size=2)
        self.assertEqual(augmented.shape, (8, 64, 64, 3))
        np.testing.assert_array_equal(labels, np.tile(y, 2))
        np.testing.assert_array_equal(origins, np.arange(4))
        self.assertTrue(np.isfinite(augmented).all())


if __name__ == '__main__':
    unittest.main()
