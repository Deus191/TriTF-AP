"""Executable TriTF-AP classifier corresponding to main.tex equations 135-181.

Unspecified choices: spatial dropout=0.25, BN epsilon=1e-3, no convolution
bias before BN. This implementation still requires fresh experiments; historical
weights/results cannot be attributed to it without rerunning. Counts are measured,
never forced to the manuscript's rounded 4.6K value.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

PAPER_MODEL_VERSION = "tritf_ap_paper_v1"


@tf.keras.utils.register_keras_serializable(package="TriTF")
class SimAM(layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        d = tf.square(inputs - tf.reduce_mean(inputs, axis=(1, 2), keepdims=True))
        n = tf.cast(tf.maximum(tf.shape(inputs)[1] * tf.shape(inputs)[2] - 1, 1), inputs.dtype)
        variance = tf.reduce_sum(d, axis=(1, 2), keepdims=True) / n
        return inputs * tf.sigmoid(d / (4.0 * (variance + self.epsilon)) + 0.5)

    def get_config(self):
        return dict(super().get_config(), epsilon=self.epsilon)


@tf.keras.utils.register_keras_serializable(package="TriTF")
class RepDepthwise(layers.Layer):
    def __init__(self, deploy=False, **kwargs):
        super().__init__(**kwargs)
        self.deploy = deploy
        if deploy:
            self.fused = layers.DepthwiseConv2D(3, padding="same", use_bias=True)
        else:
            self.convs = [layers.DepthwiseConv2D(k, padding="same", use_bias=False)
                          for k in ((3, 3), (3, 1), (1, 3))]
            self.norms = [layers.BatchNormalization() for _ in range(4)]

    def build(self, input_shape):
        if self.deploy:
            self.fused.build(input_shape)
        else:
            for conv in self.convs:
                conv.build(input_shape)
            for norm in self.norms:
                norm.build(input_shape)
            self.gates = self.add_weight(name="gates", shape=(4,), initializer="ones")
        super().build(input_shape)

    def call(self, inputs, training=None):
        if self.deploy:
            return self.fused(inputs)
        terms = [conv(inputs) for conv in self.convs] + [inputs]
        return tf.add_n([self.gates[i] * self.norms[i](term, training=training)
                         for i, term in enumerate(terms)])

    def fused_weights(self):
        if self.deploy:
            return self.fused.get_weights()
        channels = self.convs[0].get_weights()[0].shape[2]
        kernel = np.zeros((3, 3, channels, 1), dtype=np.float32)
        bias = np.zeros(channels, dtype=np.float32)
        for i, norm in enumerate(self.norms):
            if i == 3:
                weight = np.zeros_like(kernel)
                weight[1, 1, :, 0] = 1
            else:
                weight = self.convs[i].get_weights()[0]
                height, width = weight.shape[:2]
                weight = np.pad(weight, (((3-height)//2, (3-height)//2),
                                         ((3-width)//2, (3-width)//2), (0, 0), (0, 0)))
            gamma, beta, mean, variance = norm.get_weights()
            scale = gamma / np.sqrt(variance + norm.epsilon)
            gate = float(self.gates.numpy()[i])
            kernel += gate * weight * scale.reshape(1, 1, -1, 1)
            bias += gate * (beta - mean * scale)
        return [kernel, bias]

    def get_config(self):
        return dict(super().get_config(), deploy=self.deploy)


def build_classifier(input_shape=(64, 64, 3), num_classes=2, deploy=False, dropout=0.25):
    if tuple(input_shape) != (64, 64, 3):
        raise ValueError("Paper classifier expects 64x64 RGB TF images, not raw EEG.")
    if num_classes < 2:
        raise ValueError("At least two classes are required.")
    inputs = layers.Input(shape=input_shape, name="tf_image")
    # Separate PW and DW permit folding both the stem and output BN at export.
    x = layers.DepthwiseConv2D(8, padding="same", use_bias=False, name="stem_dw")(inputs)
    x = layers.Conv2D(24, 1, use_bias=deploy, name="stem_pw")(x)
    if not deploy:
        x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(4, name="stem_pool")(x)
    x = layers.SpatialDropout2D(dropout, name="stem_drop")(x)
    x = RepDepthwise(deploy=deploy, name="rep_dw")(x)
    x = layers.Conv2D(32, 1, use_bias=deploy, name="rep_pw")(x)
    if not deploy:
        x = layers.BatchNormalization(name="rep_bn")(x)
    x = layers.ReLU(name="rep_relu")(x)
    x = SimAM(name="simam")(x)
    x = layers.MaxPooling2D(2, name="rep_pool")(x)
    x = layers.SpatialDropout2D(dropout, name="rep_drop")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(72, activation="relu", name="head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="prediction")(x)
    graph = "deploy" if deploy else "training"
    return tf.keras.Model(inputs, outputs, name="TriTF_AP_{}".format(graph))


def export_deploy(model):
    """Fuse BN/gates using running statistics; equivalent only in inference mode."""
    deployed = build_classifier(model.input_shape[1:], model.output_shape[-1], deploy=True,
                                dropout=model.get_layer("stem_drop").rate)
    for layer in deployed.layers:
        if layer.name == "rep_dw":
            layer.fused.set_weights(model.get_layer("rep_dw").fused_weights())
        elif layer.name in ("stem_pw", "rep_pw"):
            weight = model.get_layer(layer.name).get_weights()[0]
            norm = model.get_layer(layer.name.replace("_pw", "_bn"))
            gamma, beta, mean, variance = norm.get_weights()
            scale = gamma / np.sqrt(variance + norm.epsilon)
            layer.set_weights([weight * scale.reshape(1, 1, 1, -1), beta - mean * scale])
        elif layer.weights:
            layer.set_weights(model.get_layer(layer.name).get_weights())
    return deployed
