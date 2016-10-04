"""Streaming Stochastic Gradient Descent implementation, starter task.

This script assumes that the provided Matlab file contains the following
information:

    X_(train|test)
    y_(train|test)
    num_samples

Usage:
    ssgd.py

Options:
    --epochs=<epochs>  Number of passes over the training data
    --eta0=<eta0>      The initial learning rate
    --damp=<damp>      Amount to multiply each learning rate by
    --buffer=<buffer>  Size of buffer, amount of data loaded into memory
    --train=<train>    Path to train file. (.mat) [default: data/train.mat]
    --test=<test>      Path to test file. (.mat) [default: data/test.mat]
"""

import docopt
import numpy as np
import scipy
import sklearn.metrics

from typing import Tuple


NUM_FEATURES = 10  # CHANGE ME


def main() -> None:
    """Load data and launch training, then evaluate accuracy."""
    arguments = docopt.docopt(__doc__, version='ssgd 1.0')
    model = train(
        train_path=arguments['<train>'],
        epochs=arguments['<epochs>'],
        eta0=arguments['<eta0>'],
        damp=arguments['<damp>'],
        buffer=arguments['<buffer>'])
    output(model, arguments['<test>'])


def output(model: np.ndarray, test_path: str) -> None:
    """Output the test results.

    Args:
        model: The trained model
        test_path: Path to the test file (.mat)
    """
    X_test, y_test = laod_test_dataset(test_path)
    y_hat = model.dot(X_test)
    print('Accuracy:', sklearn.metrics.accuracy_score(y_test, y_hat))


def load_test_dataset(test_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and give test data in memory.

    Args:
        test_path: Path to the test file (.mat)
    """
    test_data = scipy.io.loadmat(test_path)
    return (test_data['X_test'], test_data['Y_test'])


def train(
        train_path: str,
        epochs: int,
        eta0: float,
        damp: float,
        buffer: int) -> np.ndarray:
    """Train using stochastic gradient descent.

    Args:
        epochs: Number of passes over training data
        eta0: Starting learning rate
        damp: Amount to multiply learning rate at each epoch
        buffer: Size of data to load into each block

    Returns:
        The trained model
    """
    n, size_per_sample = 0, 1  # CHANGE ME
    w = np.eye(N=NUM_FEATURES)
    for p in range(epochs):
        permute_train_dataset(train_path)
        for t in range(n):
            if t % (buffer // size_per_sample) == 0:
                X, Y = read_block(p, t)
            x, y = X[t], Y[t]
            w -= eta0*(damp**(p-1))*np.linalg.inv(x.T.dot(x)).dot(x.dot(y))
    return w


def permute_train_dataset(train_path: str) -> None:
    """Permute the data in place.

    Args:
        train_path: Path to the train file (.mat)
    """
    pass


def read_block(p: int, t: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read block of data."""
    return (np.array(), np.array())


if __name__ == '__main__':
    main()
