""""""

from sklearn import preprocessing
from typing import Tuple
from typing import Callable

import numpy as np


class Data:
    """Represents data. Is guaranteed to have all training, test data."""

    def __init__(
            self,
            labels: np.ndarray,
            num_classes: int,
            one_hot: bool,
            X: np.ndarray):
        self.X = X
        self.labels = labels
        self.Y = to_one_hot(num_classes, labels) if one_hot else labels
        self.n = X.shape[0]


def read_dataset(
        data_hook: Callable[[np.ndarray, np.ndarray], Tuple],
        dtype: str,
        num_classes: int,
        one_hot: bool,
        path: str,
        shape: Tuple[int, int],
        subset: int) -> Data:
    """Read the dataset in its entirety.

    Returns the training input and the labels. Note that labels are necessarily
    *not* one hot vectors. They are values.
    """
    data = np.memmap(path, dtype=dtype, mode='r', shape=(shape[0], shape[1] + 1))
    if subset > 0:
        data = data[:subset]
    X, labels = data_hook(*block_x_labels(data))
    return Data(labels, num_classes, one_hot, X)


def block_x_labels(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a block of data into X and Y.

    Args:
        block: The block of data to extract X and Y from
        dtype: Datatype of the provided block

    Returns:
        X: the data inputs
        labels: the data outputs
    """
    X = block[:, :-1].astype('float64', copy=False)
    labels = block[:, -1].astype(int, copy=False)
    return X, labels


def to_one_hot(num_classes: int, y: np.ndarray):
    """Convert vector into one hot form."""
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(range(num_classes)))
    return lb.transform(y)


def de_one_hot(X: np.ndarray):
    """Convert one hot vectors back into class labels."""
    return np.argmax(X, axis=1)
