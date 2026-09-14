"""Label-independent 8-30 Hz Morlet representation and numerical helpers."""
import numpy as np
import pywt


def normalize_trials(data):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError("Expected finite 3D trials.")
    denominator = np.max(np.abs(data), axis=(1, 2), keepdims=True)
    return np.divide(data, denominator, out=np.zeros_like(data), where=denominator > 0)


def cwt_band(signal, fs, wavelet="morl", bins=64, low=8.0, high=30.0):
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or len(signal) < 2 or not np.isfinite(signal).all():
        raise ValueError("CWT input must be a finite 1D signal.")
    if not 0 < low < high < fs / 2 or bins < 2:
        raise ValueError("Invalid frequency range / sampling rate.")
    frequencies = np.linspace(low, high, bins)
    scales = pywt.frequency2scale(wavelet, frequencies / fs)
    coefficients, actual = pywt.cwt(signal, scales, wavelet, sampling_period=1.0 / fs)
    return np.abs(coefficients), actual


def trial_tf_image(trial, fs, wavelet="morl", bipolar=True):
    """trial=(3,T), channel order C3,Cz,C4. No class label argument by design."""
    from PIL import Image
    from matplotlib import colormaps
    trial = np.asarray(trial)
    if trial.ndim != 2 or trial.shape[0] != 3:
        raise ValueError("Exactly C3,Cz,C4 in shape (3,T) are required.")
    derived = np.stack((trial[0] - trial[1], trial[2] - trial[1])) if bipolar else trial
    magnitude = np.concatenate([cwt_band(signal, fs, wavelet)[0] for signal in derived], axis=0)
    maximum = magnitude.max()
    normalized = magnitude / maximum if maximum > 0 else np.zeros_like(magnitude)
    rgb = np.rint(colormaps["jet"](normalized)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize((64, 64), Image.Resampling.BILINEAR)


def fit_ea(data, epsilon=1e-6):
    """Fit on training data only, shape (N,T,C); reuse result for validation/test."""
    data = np.asarray(data, dtype=np.float64)
    covariance = np.einsum("ntc,ntd->cd", data, data) / (data.shape[0] * data.shape[1])
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * (1.0 / np.sqrt(np.maximum(values, epsilon)))) @ vectors.T


def apply_ea(data, transform):
    return np.asarray(data) @ np.asarray(transform).T
