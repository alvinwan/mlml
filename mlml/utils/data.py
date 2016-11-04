""""""

from typing import Tuple
from typing import Callable

import numpy as np


def read_dataset(
        data_hook: Callable[[np.ndarray, np.ndarray], Tuple],
        dtype: str,
        path: str,
        shape: Tuple[int, int],
        num_classes: int=10,
        one_hot: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """Read the dataset in its entirety."""
    data = np.memmap(path, dtype=dtype, mode='r', shape=(shape[0], shape[1] + 1))
    return data_hook(*block_to_x_y(data, num_classes, one_hot))


def block_to_x_y(
        block: np.ndarray,
        num_classes: int=10,
        one_hot: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a block of data into X and Y.

    Args:
        block: The block of data to extract X and Y from
        num_classes: Number of classes
        one_hot: Whether or not to use one hot encodings

    Returns:
        X: the data inputs
        Y: the data outputs
    """
    X = block[:, :-1].astype('float64', copy=False)
    y = np.matrix(block[:, -1].astype(int, copy=False)).T
    if one_hot:
        y = np.eye(num_classes)[y].reshape((X.shape[0], num_classes))
    return X, y


def de_one_hot(X: np.ndarray):
    """Convert one hot vectors back into class labels."""
    return np.argmax(X, axis=1)