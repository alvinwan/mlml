""""""

from typing import Tuple
from typing import Callable

import numpy as np


def read_dataset(
        data_hook: Callable[[np.ndarray, np.ndarray], Tuple],
        dtype: str,
        path: str,
        shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Read the dataset in its entirety.

    Returns the training input and the labels. Note that labels are necessarily
    *not* one hot vectors. They are values.
    """
    data = np.memmap(path, dtype=dtype, mode='r', shape=(shape[0], shape[1] + 1))
    return data_hook(*block_x_labels(data))


def block_x_labels(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a block of data into X and Y.

    Args:
        block: The block of data to extract X and Y from

    Returns:
        X: the data inputs
        labels: the data outputs
    """
    X = block[:, :-1].astype('float64', copy=False)
    labels = block[:, -1].astype(int, copy=False)
    return X, labels


def to_one_hot(num_classes: int, y: np.ndarray):
    one_hot = np.eye(num_classes)[y]
    if len(one_hot.shape) > 2:
        one_hot.shape = (one_hot.shape[0], one_hot.shape[-1])
    return one_hot


def de_one_hot(X: np.ndarray):
    """Convert one hot vectors back into class labels."""
    return np.argmax(X, axis=1)
