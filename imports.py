"""Imports files into data/, in the format required by ssgd.

This module provides several utilities for saving numpy matrices to


"""

import numpy as np

TRAIN_FILEPATH_FORMAT = 'data/{namespace}-{dtype}-{n}-train'
TEST_FILEPATH_FORMAT = 'data/{namespace}-{dtype}-{n}-test'


def save_inputs_labels_as_data(
        namespace: str,
        X_train: np.ndarray,
        labels_train: np.ndarray,
        X_test: np.ndarray,
        labels_test: np.ndarray,
        dtype: str='float64') -> None:
    """Save inputs and labels in a format that ssgd can process.

    Files will be saved using data/{namespace}-train and data/{namespace}-test.

    Args:
        namespace: Name of filset
        X_train: Inputs for training data, formatted as numpy matrix
        labels_train: Labels for training data, single column for labels
        X_test: Inputs for testing data, formatted as numpy matrix
        labels_test: Labels for testing data, single column for labels
        dtype: Type of data in matrix
    """
    train = np.concatenate((X_train, labels_train))
    test = np.concatenate((X_test, labels_test))
    save_inputs_as_data(namespace, train, test, dtype)


def save_inputs_as_data(
        namespace: str,
        train: np.ndarray,
        test: np.ndarray,
        dtype: str='float64') -> None:
    """Save data in a format that ssgd can process.

    Files will be saved using data/{namespace}-train and data/{namespace}-test.

    Args:
        dtype: Type of data in matrix
        namespace: Name of fileset
        train: Inputs and labels for training data, formatted as numpy matrix
        test: Inputs and labels for testing data, formatted as numpy matrix
    """
    train_path = TRAIN_FILEPATH_FORMAT.format(
        namespace=namespace,
        dtype=dtype,
        n=train.shape[0])
    train_fh = np.memmap(train_path, dtype=dtype, mode='w', shape=train.shape)
    train_fh[:] = train[:]
    del train_fh

    test_path = TEST_FILEPATH_FORMAT.format(
        namespace=namespace,
        dtype=dtype,
        n=test.shape[0])
    test_fh = np.memmap(test_path, dtype=dtype, mode='w', shape=test.shape)
    test_fh[:] = test[:]
    del test_fh
