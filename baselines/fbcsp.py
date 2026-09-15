"""Leakage-safe filter-bank CSP with optional feature selection and SVM."""
from functools import partial

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_BANDS = ((8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 30))


def normalized_covariance(trial):
    covariance = trial @ trial.T
    trace = float(np.trace(covariance))
    if trace <= np.finfo(float).eps:
        raise ValueError("Encountered a zero-energy EEG trial")
    return covariance / trace


class CSPBank:
    def __init__(self, sampling_rate=250, bands=DEFAULT_BANDS, components_per_side=1):
        self.sampling_rate = sampling_rate
        self.bands = tuple(tuple(band) for band in bands)
        self.components_per_side = components_per_side
        self.filters_ = None

    def _bandpass(self, data, band):
        low, high = band
        if not 0 < low < high < self.sampling_rate / 2:
            raise ValueError("Invalid filter band {} for {} Hz".format(band, self.sampling_rate))
        sos = butter(4, (low, high), btype="bandpass", fs=self.sampling_rate, output="sos")
        return sosfiltfilt(sos, data, axis=-1).astype(np.float32, copy=False)

    def fit(self, data, labels):
        data = np.asarray(data, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if data.ndim != 3 or len(data) != len(labels):
            raise ValueError("Expected aligned data=(trials, channels, time) and labels")
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError("CSP requires both binary classes")
        n_channels = data.shape[1]
        if self.components_per_side < 1 or 2 * self.components_per_side > n_channels:
            raise ValueError("components_per_side is incompatible with the channel count")
        filters = []
        for band in self.bands:
            filtered = self._bandpass(data, band)
            class_covariances = []
            for label in (0, 1):
                covariances = [normalized_covariance(trial) for trial in filtered[labels == label]]
                class_covariances.append(np.mean(covariances, axis=0))
            composite = class_covariances[0] + class_covariances[1]
            composite += np.eye(n_channels) * 1e-10
            eigenvalues, eigenvectors = eigh(class_covariances[0], composite)
            del eigenvalues
            order = list(range(self.components_per_side)) + list(
                range(n_channels - self.components_per_side, n_channels)
            )
            filters.append(eigenvectors[:, order].T)
        self.filters_ = filters
        return self

    def transform(self, data):
        if self.filters_ is None:
            raise RuntimeError("CSPBank must be fitted before transform")
        data = np.asarray(data, dtype=np.float64)
        features = []
        for band, spatial_filter in zip(self.bands, self.filters_):
            filtered = self._bandpass(data, band)
            projected = np.einsum("kc,nct->nkt", spatial_filter, filtered)
            variances = np.var(projected, axis=-1)
            normalized = variances / np.maximum(variances.sum(axis=1, keepdims=True), 1e-12)
            features.append(np.log(np.maximum(normalized, 1e-12)))
        return np.concatenate(features, axis=1)


class FBCSPSVM:
    def __init__(
        self,
        sampling_rate=250,
        bands=DEFAULT_BANDS,
        components_per_side=1,
        selected_features=8,
        svm_c=1.0,
        svm_kernel="linear",
        seed=42,
    ):
        self.csp = CSPBank(sampling_rate, bands, components_per_side)
        self.selected_features = selected_features
        self.svm_c = svm_c
        self.svm_kernel = svm_kernel
        self.seed = seed
        self.classifier_ = None

    def fit(self, data, labels):
        features = self.csp.fit(data, labels).transform(data)
        k = min(self.selected_features, features.shape[1])
        self.classifier_ = Pipeline(
            [
                (
                    "select",
                    SelectKBest(partial(mutual_info_classif, random_state=self.seed), k=k),
                ),
                ("scale", StandardScaler()),
                (
                    "svm",
                    SVC(
                        C=self.svm_c,
                        kernel=self.svm_kernel,
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        self.classifier_.fit(features, labels)
        return self

    def predict(self, data):
        if self.classifier_ is None:
            raise RuntimeError("FBCSPSVM must be fitted before predict")
        return self.classifier_.predict(self.csp.transform(data))

    @property
    def feature_count(self):
        return len(self.csp.bands) * 2 * self.csp.components_per_side
