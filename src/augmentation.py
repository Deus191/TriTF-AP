"""NEW split-safe adaptation of the archived convolutional VAE.

Caller must supply ONLY the classifier fit split. Posterior samples inherit their
source labels; class preservation is an assumption to evaluate, not a guarantee.
"""
import numpy as np


def augment_training(x, y, epochs=100, batch_size=128, seed=42, output_dir=None):
    import tensorflow as tf
    from tensorflow.keras import layers
    tf.keras.utils.set_random_seed(seed)
    inputs = layers.Input((64, 64, 3))
    h = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    h = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(h)
    h = layers.Flatten()(h)
    h = layers.Dense(16, activation="relu")(h)
    mean, logvar = layers.Dense(2)(h), layers.Dense(2)(h)
    encoder = tf.keras.Model(inputs, [mean, logvar])
    z = layers.Input((2,))
    h = layers.Dense(16 * 16 * 64, activation="relu")(z)
    h = layers.Reshape((16, 16, 64))(h)
    h = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(h)
    h = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(h)
    decoded = layers.Conv2DTranspose(3, 3, padding="same", activation="sigmoid")(h)
    decoder = tf.keras.Model(z, decoded)

    class TrainingVAE(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.encoder, self.decoder = encoder, decoder
            self.loss_meter = tf.keras.metrics.Mean(name="loss")
            self.reconstruction_meter = tf.keras.metrics.Mean(name="reconstruction")
            self.kl_meter = tf.keras.metrics.Mean(name="kl")

        @property
        def metrics(self):
            return [self.loss_meter, self.reconstruction_meter, self.kl_meter]

        def train_step(self, data):
            data, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
            with tf.GradientTape() as tape:
                mu, lv = self.encoder(data, training=True)
                lv = tf.clip_by_value(lv, -20.0, 20.0)
                sample = mu + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(mu))
                reconstruction = self.decoder(sample, training=True)
                bce = tf.keras.losses.binary_crossentropy(data, reconstruction)
                recon_loss = tf.reduce_mean(tf.reduce_sum(bce, axis=(1, 2)))
                kl = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + lv - tf.square(mu) - tf.exp(lv)), axis=1))
                loss = recon_loss + kl
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            for meter, value in zip(self.metrics, (loss, recon_loss, kl)):
                meter.update_state(value)
            return {meter.name: meter.result() for meter in self.metrics}

    if epochs < 1:
        raise ValueError("VAE epochs must be positive.")
    vae = TrainingVAE()
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    if output_dir is not None:
        callbacks.append(tf.keras.callbacks.CSVLogger(str(output_dir / "vae_history.csv")))
    vae.fit(x, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2, callbacks=callbacks)
    mu, lv = encoder.predict(x, batch_size=batch_size, verbose=0)
    rng = np.random.default_rng(seed)
    z_sample = mu + np.exp(0.5 * np.clip(lv, -20, 20)) * rng.normal(size=mu.shape)
    generated = decoder.predict(z_sample, batch_size=batch_size, verbose=0).astype(np.float32)
    if not np.isfinite(generated).all():
        raise ValueError("Nonfinite VAE output.")
    if output_dir is not None:
        encoder.save(str(output_dir / "vae_encoder.keras"))
        decoder.save(str(output_dir / "vae_decoder.keras"))
    return np.concatenate((x, generated)), np.concatenate((y, y)), np.arange(len(x))
