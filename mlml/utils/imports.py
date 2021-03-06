"""Imports files into data/, in the format required by ssgd.

This module provides several utilities for saving numpy matrices to binary. It
also offers utilities for artificially inflating size of data, to exhibit the
capabilities of ssgd.


Usage:
    imports.py (mnist|spam|cifar-10) [options]

Options:
    --dtype=<dtype>             Datatype of generated memmap [default: uint8]
    --percentage=<percentage>   Percentage of data for training [default: 0.8]
"""

import pickle
import docopt
import numpy as np
import scipy.io

from mnist import MNIST
from typing import Tuple

TRAIN_FILEPATH_FORMAT = 'data/{namespace}-{dtype}-{n}-train'
TEST_FILEPATH_FORMAT = 'data/{namespace}-{dtype}-{n}-test'


def save_inputs_labels_as_data(
        namespace: str,
        X_train: np.ndarray,
        labels_train: np.ndarray,
        X_test: np.ndarray,
        labels_test: np.ndarray,
        dtype: str='uint8') -> None:
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
    train = np.concatenate((X_train, labels_train), axis=1)
    test = np.concatenate((X_test, labels_test), axis=1)
    save_inputs_as_data(namespace, train, test, dtype)


def save_inputs_as_data(
        namespace: str,
        train: np.ndarray,
        test: np.ndarray,
        dtype: str='uint8') -> None:
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
    train_fh = np.memmap(train_path, dtype=dtype, mode='w+', shape=train.shape)
    train_fh[:] = train.astype(dtype)[:]
    del train_fh
    print(' * Wrote to', train_path, 'with datatype', dtype)

    test_path = TEST_FILEPATH_FORMAT.format(
        namespace=namespace,
        dtype=dtype,
        n=test.shape[0])
    test_fh = np.memmap(test_path, dtype=dtype, mode='w+', shape=test.shape)
    test_fh[:] = test.astype(dtype)[:]
    del test_fh
    print(' * Wrote to', test_path, 'with datatype', dtype)


def import_mnist(dtype: str):
    """Imports MNIST files held in ./data/ folder.

    Converts all data to binary values and stores in a binary file under the
    data directory.
    """
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    labels_train = np.matrix(labels_train).T
    labels_test = np.matrix(labels_test).T
    save_inputs_labels_as_data('mnist', X_train, labels_train, X_test,
                               labels_test, dtype)


def import_spam(dtype: str, training_data_percentage: float=0.8):
    """Imports spam files held in ./data/ folder.

    Converts all data to binary values and stores in a binary file under the
    data directory.

    Args:
        training_data_percentage: Percentage of training data to keep for
                                  training - remainder used for validation
                                  (float between 0 and 1)
    """
    data = scipy.io.loadmat('data/spam.mat')
    X, y = shuffle(data['Xtrain'], data['ytrain'])
    N = int(np.ceil(X.shape[0] * training_data_percentage))
    X_train, X_test = X[:N], X[N:]
    labels_train, labels_test = y[:N], y[N:]
    save_inputs_labels_as_data('spam', X_train, labels_train, X_test,
                               labels_test, dtype)


def import_cifar_10(dtype: str):
    """Imports CIFAR files held in ./data/ folder.

    Stores in a binary file under the data directory.
    """
    train_paths = ['data/data_batch_%d' % i for i in range(1, 6)]
    X_train, labels_train = None, None
    for train_path in train_paths:
        with open(train_path, 'rb') as f:
            raw = pickle.load(f, encoding='latin1')
            X_train = raw['data'] if X_train is None else \
                np.concatenate((X_train, raw['data']))
            labels_train = raw['labels'] if labels_train is None else \
                np.concatenate((labels_train, raw['labels']))
    labels_train = np.matrix(labels_train).T

    test_path = 'data/test_batch'
    with open(test_path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
        X_test = raw['data']
        labels_test = np.matrix(raw['labels']).T

    save_inputs_labels_as_data('cifar-10', X_train, labels_train, X_test,
                               labels_test, dtype)


def shuffle(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle all X and Y data, in tandem with one another.

    Args:
        X: input data
        Y: output data, or labels
    """
    indices = list(range(X.shape[0]))
    np.random.shuffle(indices)
    return X[indices], Y[indices]


def main():
    """Run the command-line interface."""
    arguments = docopt.docopt(__doc__, version='SSGD 1.0')
    if arguments['mnist']:
        import_mnist(arguments['--dtype'])
    elif arguments['spam']:
        import_spam(arguments['--dtype'], float(arguments['--percentage']))
    elif arguments['cifar-10']:
        import_cifar_10(arguments['--dtype'])


if __name__ == '__main__':
    main()
