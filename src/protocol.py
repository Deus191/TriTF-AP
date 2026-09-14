"""Array/identity checks and leakage-safe split logic; no TensorFlow imports."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def labels_to_int(y, num_classes=None):
    y = np.asarray(y)
    if y.ndim == 2:
        if y.shape[1] < 2 or not np.isin(y, [0, 1]).all() or not np.all(y.sum(axis=1) == 1):
            raise ValueError("Expected strict one-hot labels, one active class per row.")
        inferred = y.shape[1]
        y = y.argmax(axis=1)
    elif y.ndim == 1 and len(y) and np.isfinite(y).all() and np.equal(y, y.astype(int)).all():
        inferred = int(y.max()) + 1
    else:
        raise ValueError("Expected a nonempty integer label vector or one-hot matrix.")
    count = num_classes if num_classes is not None else inferred
    if count < 2 or inferred > count or np.any(y < 0):
        raise ValueError("Labels out of class range.")
    return y.astype(np.int64), count


def validate_images(x, y):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4 or tuple(x.shape[1:]) != (64, 64, 3) or len(x) != len(y) or not len(x):
        raise ValueError("Expected nonempty X=(N,64,64,3), y=(N,...).")
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Images must be finite and normalized to [0,1].")
    return x


def split_source(y, validation_fraction=0.2, seed=42, groups=None):
    labels, _ = labels_to_int(y)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie in (0,1).")
    indices = np.arange(len(labels))
    if groups is None:
        train, val = train_test_split(indices, test_size=validation_fraction, stratify=labels, random_state=seed)
    else:
        groups = np.asarray(groups)
        if len(groups) != len(labels) or len(np.unique(groups)) < 2:
            raise ValueError("Group validation requires at least two source groups.")
        splitter = GroupShuffleSplit(n_splits=64, test_size=validation_fraction, random_state=seed)
        for train, val in splitter.split(indices, labels, groups):
            if set(labels[train]) == set(labels) and set(labels[val]) == set(labels):
                break
        else:
            raise ValueError("Cannot form a group-disjoint validation set covering all classes.")
    if set(labels[train]) != set(labels) or set(labels[val]) != set(labels):
        raise ValueError("Both source splits must cover every class.")
    return train, val


def check_ids(source_ids, test_ids, n_source, n_test):
    source_ids, test_ids = list(map(str, source_ids)), list(map(str, test_ids))
    if len(source_ids) != n_source or len(test_ids) != n_test:
        raise ValueError("IDs must align one-to-one with samples.")
    if len(set(source_ids)) != n_source or len(set(test_ids)) != n_test:
        raise ValueError("Duplicate original trial IDs are not permitted.")
    if set(source_ids) & set(test_ids):
        raise ValueError("Source and test trial identities overlap.")
    return np.asarray(source_ids), np.asarray(test_ids)
