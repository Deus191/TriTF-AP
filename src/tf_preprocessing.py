"""Paper-defined, label-independent TriTF-AP time-frequency representation."""
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


def paper_bipolar_pair(trial):
    """Return the two paper derivations from C3/Cz/C4 input in shape (3,T)."""
    trial = np.asarray(trial, dtype=np.float64)
    if trial.ndim != 2 or trial.shape[0] != 3 or not np.isfinite(trial).all():
        raise ValueError("Exactly finite C3,Cz,C4 input in shape (3,T) is required.")
    return np.stack((trial[0] - trial[1], trial[2] - trial[1]))


def paper_tf_canvas(trial, fs, wavelet="morl"):
    """Build the normalized two-map log-power canvas before RGB conversion."""
    derived = paper_bipolar_pair(trial)
    power = np.concatenate(
        [np.square(cwt_band(signal, fs, wavelet)[0]) for signal in derived], axis=0
    )
    log_power = np.log1p(power)
    maximum = log_power.max()
    return log_power / maximum if maximum > 0 else np.zeros_like(log_power)


def trial_tf_image(trial, fs, wavelet="morl"):
    """Create the paper's 64x64 RGB image; labels cannot affect this transform."""
    from PIL import Image
    from matplotlib import colormaps
    canvas = paper_tf_canvas(trial, fs, wavelet)
    rgb = np.rint(colormaps["jet"](canvas)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize((64, 64), Image.Resampling.BILINEAR)


def fit_ea(data, epsilon=1e-6):
    """Fit on training data only, shape (N,T,C); reuse result for validation/test."""
    data = np.asarray(data, dtype=np.float64)
    covariance = np.einsum("ntc,ntd->cd", data, data) / (data.shape[0] * data.shape[1])
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * (1.0 / np.sqrt(np.maximum(values, epsilon)))) @ vectors.T


def apply_ea(data, transform):
    return np.asarray(data) @ np.asarray(transform).T
