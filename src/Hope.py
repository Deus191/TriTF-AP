"""TriTF-AP training entry; x_val/y_val are FINAL TEST, not early-stop data."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid
import numpy as np
from protocol import check_ids, labels_to_int, split_source, validate_images


def Train(x_train, y_train, x_val=None, y_val=None, log=None, *, seed=42,
          groups=None, source_ids=None, test_ids=None, validation_fraction=0.2,
          augment=False, epochs=200, batch_size=128, patience=10,
          vae_epochs=100, output_dir=None, protocol_name="source_holdout", dataset_name="unspecified"):
    if x_val is None or y_val is None:
        raise ValueError("Independent FINAL test data required; never used for early stopping.")
    labels, num_classes = labels_to_int(y_train)
    test_labels, _ = labels_to_int(y_val, num_classes=num_classes)
    x_train = validate_images(x_train, labels)
    x_test = validate_images(x_val, test_labels)
    if set(labels) != set(range(num_classes)):
        raise ValueError("Source data must contain every configured class.")
    supplied_ids = source_ids is not None and test_ids is not None
    if (source_ids is None) != (test_ids is None):
        raise ValueError("Provide both source and test IDs or neither.")
    source_ids = source_ids if supplied_ids else ["source/{}".format(i) for i in range(len(labels))]
    test_ids = test_ids if supplied_ids else ["test/{}".format(i) for i in range(len(test_labels))]
    source_ids, test_ids = check_ids(source_ids, test_ids, len(labels), len(test_labels))
    fit_idx, validation_idx = split_source(labels, validation_fraction, seed, groups)
    if min(epochs, batch_size, patience) < 1:
        raise ValueError("epochs, batch_size and patience must be positive.")
    if output_dir is None:
        if log:
            logpath = Path(log).resolve()
            output_dir = logpath.parent / logpath.stem
        else:
            output_dir = Path(__file__).resolve().parents[1] / "outputs" / ("run_" + uuid.uuid4().hex[:12])
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = dict(status="started", created_utc=datetime.now(timezone.utc).isoformat(),
                  dataset=dataset_name, protocol=protocol_name, seed=seed,
                  model="tritf_ap_paper_v1", num_classes=num_classes,
                  expected_input_representation="tritf_ap_paper_v1",
                  epochs=epochs, batch_size=batch_size, patience=patience,
                  learning_rate=0.0003, monitor="val_loss", augment=augment,
                  vae_epochs=vae_epochs if augment else 0, validation_fraction=validation_fraction,
                  validation_unit="group" if groups is not None else "trial",
                  external_trial_ids_supplied=supplied_ids,
                  source_count=len(labels), fit_original_count=len(fit_idx),
                  validation_count=len(validation_idx), test_count=len(test_labels))

    def save_config():
        (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    save_config()
    split_info = {"fit_indices": fit_idx.tolist(), "validation_indices": validation_idx.tolist(),
                  "fit_ids": source_ids[fit_idx].tolist(), "validation_ids": source_ids[validation_idx].tolist(),
                  "test_ids": test_ids.tolist(), "groups": np.asarray(groups).tolist() if groups is not None else None}
    (output_dir / "splits.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    try:
        import tensorflow as tf
        from sklearn.metrics import accuracy_score, cohen_kappa_score, log_loss
        from paper_model import PAPER_MODEL_VERSION, build_classifier, export_deploy
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        config["tensorflow_version"] = tf.__version__
        x_fit, y_fit = x_train[fit_idx], labels[fit_idx]
        if augment:
            from augmentation import augment_training
            x_fit, y_fit, origins = augment_training(x_fit, y_fit, epochs=vae_epochs,
                                                   batch_size=batch_size, seed=seed, output_dir=output_dir)
            (output_dir / "augmentation_origins.json").write_text(
                json.dumps({"generated_from_trial_ids": source_ids[fit_idx][origins].tolist()}, indent=2), encoding="utf-8")
        config["fit_total_count"] = len(x_fit)
        tf.keras.utils.set_random_seed(seed)
        model = build_classifier(num_classes=num_classes)
        if config["model"] != PAPER_MODEL_VERSION:
            raise RuntimeError("Training configuration and paper model version disagree.")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        config["training_graph_parameters"] = model.count_params()
        save_config()
        checkpoint = output_dir / "best.weights.h5"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.CSVLogger(str(output_dir / "history.csv")),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        history = model.fit(x_fit, y_fit, validation_data=(x_train[validation_idx], labels[validation_idx]),
                            batch_size=batch_size, epochs=epochs, shuffle=True, callbacks=callbacks, verbose=2)
        if not np.isfinite(history.history["loss"]).all() or not np.isfinite(history.history["val_loss"]).all():
            raise ValueError("Nonfinite loss; no final test result will be reported.")
        model.load_weights(str(checkpoint))
        model.save(str(output_dir / "classifier.keras"))
        deployed = export_deploy(model)
        check_x = x_train[validation_idx[:min(4, len(validation_idx))]]
        expected = model(check_x, training=False).numpy()
        actual = deployed(check_x, training=False).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        deployed.save(str(output_dir / "classifier_deploy.keras"))
        config["deployment_parameters"] = deployed.count_params()
        config["fusion_max_abs_error"] = float(np.max(np.abs(expected - actual)))
        config["selected_epoch"] = int(np.argmin(history.history["val_loss"]) + 1)
        # Final test prediction occurs only AFTER checkpoint selection and export verification.
        probabilities = model.predict(x_test, batch_size=batch_size, verbose=0)
        predicted = probabilities.argmax(axis=1)
        metrics = dict(accuracy=float(accuracy_score(test_labels, predicted)),
                       kappa=float(cohen_kappa_score(test_labels, predicted)),
                       log_loss=float(log_loss(test_labels, probabilities, labels=list(range(num_classes)))),
                       n_test=len(test_labels), selected_epoch=config["selected_epoch"])
        with (output_dir / "test_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["trial_id", "y_true", "y_pred"] + ["p_{}".format(i) for i in range(num_classes)])
            writer.writerows([sample_id, int(truth), int(pred)] + prob.tolist()
                             for sample_id, truth, pred, prob in zip(test_ids, test_labels, predicted, probabilities))
        (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        config["status"] = "complete"
        save_config()
        print("Final independent test:", metrics, "Saved to:", output_dir)
        return model
    except Exception as exc:
        config.update(status="failed", error=str(exc))
        save_config()
        raise


if __name__ == "__main__":
    raise SystemExit("Use train_openbmi.py or VAE.py with configured data. No implicit dummy training.")
