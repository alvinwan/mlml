"""Streaming Stochastic Gradient Descent implementation, starter task.

In the following, we assume the objective function is L2-regularized least
squares, otherwise known as Ridge Regression.

    (1/2) X^T ||Xw - y||_2^2 + (1/2)reg||w||_2^2

Additionally, we assume all outputs are class labels. Each binary includes both
input and outputs, where the outputs form the last column of each matrix.

Usage:
    ssgd.py closed --n=<n> --d=<d> --train=<train> --test=<test> --nt=<nt> [options]
    ssgd.py gd --n=<n> --d=<d> --train=<train> --test=<test> --nt=<nt> [options]
    ssgd.py sgd --n=<n> --d=<d> --train=<train> --test=<test> --nt=<nt> [options]
    ssgd.py ssgd --n=<n> --d=<d> --buffer=<buffer> --train=<train> --test=<test> --nt=<nt> [options]
    ssgd.py (closed|gd|sgd|ssgd) (mnist|spam) [options]

Options:
    --algo=<algo>       Shuffling algorithm to use [default: external_shuffle]
    --buffer=<num>      Size of memory in megabytes (MB) [default: 5]
    --d=<d>             Number of features
    --damp=<damp>       Amount to multiply learning rate by per epoch [default: 0.99]
    --dtype=<dtype>     The numeric type of each sample [default: float64]
    --epochs=<epochs>   Number of passes over the training data [default: 5]
    --eta0=<eta0>       The initial learning rate [default: 1e-6]
    --iters=<iters>     The number of iterations, used for gd and sgd [default: 1000]
    --n=<n>             Number of training samples
    --k=<k>             Number of classes [default: 10]
    --nt=<nt>           Number of testing samples
    --one-hot=<onehot>  Whether or not to use one hot encoding [default: False]
    --reg=<reg>         Regularization constant [default: 0.1]
    --train=<train>     Path to training data binary [default: data/train]
    --test=<test>       Path to test data [default: data/test]
"""

import docopt
import numpy as np
import scipy
import sklearn.metrics

from blocks import BlockBuffer
from shuffle import shuffle_train
from typing import Tuple


def main() -> None:
    """Load data and launch training, then evaluate accuracy."""
    arguments = preprocess_arguments(docopt.docopt(__doc__, version='ssgd 1.0'))

    if arguments['closed']:
        model = train_closed(
            dtype=arguments['--dtype'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            one_hot=arguments['--one-hot'],
            reg=arguments['--reg'],
            train_path=arguments['--train'])
    elif arguments['gd']:
        model = train_gd(
            damp=arguments['--damp'],
            dtype=arguments['--dtype'],
            eta0=arguments['--eta0'],
            iterations=arguments['--iters'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            one_hot=arguments['--one-hot'],
            reg=arguments['--reg'],
            train_path=arguments['--train'])
    elif arguments['sgd']:
        model = train_sgd(
            damp=arguments['--damp'],
            dtype=arguments['--dtype'],
            eta0=arguments['--eta0'],
            epochs=arguments['--epochs'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            one_hot=arguments['--one-hot'],
            reg=arguments['--reg'],
            train_path=arguments['--train'])
    elif arguments['ssgd']:
        model = train_ssgd(
            algorithm=arguments['--algo'],
            damp=arguments['--damp'],
            dtype=arguments['--dtype'],
            epochs=arguments['--epochs'],
            eta0=arguments['--eta0'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            num_per_block=arguments['--num-per-block'],
            one_hot=arguments['--one-hot'],
            reg=arguments['--reg'],
            train_path=arguments['--train'])
    else:
        raise UserWarning('Invalid algorithm specified.')
    X_test, y_test = read_full_dataset(
        dtype=arguments['--dtype'],
        path=arguments['--test'],
        shape=(arguments['--nt'], arguments['--d']))
    if arguments['--one-hot']:
        y_hat = np.argmax(X_test.dot(model), axis=1)
    else:
        y_hat = np.where(X_test.dot(model) > 0.5, 1, 0)
    print('Accuracy:', sklearn.metrics.accuracy_score(y_test, y_hat))


def preprocess_arguments(arguments) -> dict:
    """Preprocessing arguments dictionary by cleaning numeric values.

    Args:
        arguments: The dictionary of command-line arguments
    """

    if arguments['mnist']:
        arguments['--train'] = 'data/mnist-float64-60000-train'
        arguments['--test'] = 'data/mnist-float64-10000-test'
        arguments['--n'] = 60000
        arguments['--nt'] = 10000
        arguments['--k'] = 10
        arguments['--d'] = 784
        arguments['--one-hot'] = 'true'
    if arguments['spam']:
        arguments['--train'] = 'data/spam-float64-2760-train'
        arguments['--test'] = 'data/spam-float64-690-test'
        arguments['--n'] = 2760
        arguments['--nt'] = 690
        arguments['--k'] = 1
        arguments['--d'] = 55
        arguments['--epochs'] = 20

    arguments['--damp'] = float(arguments['--damp'])
    arguments['--epochs'] = int(arguments['--epochs'])
    arguments['--eta0'] = float(arguments['--eta0'])
    arguments['--iters'] = int(arguments['--iters'])
    arguments['--n'] = int(arguments['--n'])
    arguments['--d'] = int(arguments['--d'])
    arguments['--k'] = int(arguments['--k'])
    arguments['--num-per-block'] = min(
        int((float(arguments['--buffer']) * (10 ** 6)) // 4),
        arguments['--n'])
    arguments['--one-hot'] = arguments['--one-hot'].lower() == 'true'
    arguments['--reg'] = float(arguments['--reg'])
    return arguments


def read_full_dataset(
        dtype: str,
        path: str,
        shape: Tuple[int, int],
        num_classes: int=10,
        one_hot: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """Read the dataset in its entirety."""
    data = np.memmap(path, dtype=dtype, mode='r', shape=(shape[0], shape[1] + 1))
    return block_to_x_y(data, num_classes, one_hot)


def train_closed(
        dtype: str,
        n: int,
        num_classes: int,
        num_features: int,
        one_hot: bool,
        reg: float,
        train_path: str) -> np.ndarray:
    """Compute the closed form solution.

    Args:
        dtype: Data type of numbers in file
        n: Number of training samples
        num_classes: Number of classes
        num_features: Number of features
        limited by the size of the buffer and size of each sample
        one_hot: Whether or not to use one hot encodings
        reg: Regularization constant
        train_path: Path to the training file (binary)

    Returns:
        The trained model
    """
    shape = (n, num_features)
    X, y = read_full_dataset(dtype, train_path, shape, num_classes, one_hot)
    XTX, I, XTy = X.T.dot(X), np.identity(num_features), X.T.dot(y)
    return scipy.linalg.solve(XTX + reg*I, XTy, sym_pos=True)


def train_gd(
        damp: float,
        dtype: str,
        eta0: float,
        iterations: int,
        n: int,
        num_features: int,
        num_classes: int,
        one_hot: bool,
        reg: float,
        train_path: str) -> np.ndarray:
    """Train using gradient descent.

    Args:
        damp: Amount to multiply learning rate at each epoch
        dtype: Data type of numbers in file
        eta0: Starting learning rate
        iterations: Number of iterations to train
        n: Number of training samples
        num_classes: Number of classes
        num_features: Number of features
        one_hot: Whether or not to train with one hot encodings
        limited by the size of the buffer and size of each sample
        reg: Regularization constant
        train_path: Path to the training file (binary)

    Returns:
        The trained model
    """
    shape = (n, num_features)
    X, y = read_full_dataset(dtype, train_path, shape, num_classes, one_hot)
    XTX, XTy, w = X.T.dot(X), X.T.dot(y), np.zeros((num_features, num_classes))
    for i in range(iterations):
        grad = XTX.dot(w) - XTy + 2*reg*w
        alpha = eta0*damp**(i % 100)
        w -= alpha * grad
    return w


def train_sgd(
        damp: float,
        dtype: str,
        eta0: float,
        epochs: int,
        n: int,
        num_classes: int,
        num_features: int,
        one_hot: bool,
        reg: float,
        train_path: str) -> np.ndarray:
    """Train using stochastic gradient descent.

    Args:
        damp: Amount to multiply learning rate at each epoch
        dtype: Data type of numbers in file
        eta0: Starting learning rate
        epochs: Number of passes over training data
        n: Number of training samples
        num_classes: Number of classes
        num_features: Number of features
        one_hot: Whether or not to use one hot encodings
        limited by the size of the buffer and size of each sample
        reg: Regularization constant
        train_path: Path to the training file (binary)

    Returns:
        The trained model
    """
    shape = (n, num_features)
    X, Y = read_full_dataset(dtype, train_path, shape, num_classes, one_hot)
    w = np.zeros((num_features, num_classes))
    for i in range(epochs):
        indices = list(range(X.shape[0]))
        np.random.shuffle(indices)
        for index in indices:
            x, y = np.matrix(X[index]), np.matrix(Y[index])
            grad = x.T.dot(x.dot(w) - y) + 2 * reg * w
            alpha = eta0 * damp ** ((n * i + index) % 1000)
            w -= alpha * grad
    return w


def train_ssgd(
        algorithm: str,
        damp: float,
        dtype: str,
        epochs: int,
        eta0: float,
        n: int,
        num_classes: int,
        num_features: int,
        num_per_block: int,
        one_hot: bool,
        reg: float,
        train_path: str) -> np.ndarray:
    """Train using streaming stochastic gradient descent.

    The shuffling algorithms are described and discussed in shuffle.py.

    As described in the starter task PDF, all samples are read sequentially from
    disk. Again using a buffer, we simply read the next chunk of N samples that
    are needed for sgd to run another block.

    Args:
        algorithm: Shuffling algorithm to use
        damp: Amount to multiply learning rate at each epoch
        dtype: Data type of numbers in file
        epochs: Number of passes over training data
        eta0: Starting learning rate
        n: Number of training samples
        num_classes: Number of classes
        num_features: Number of features
        num_per_block: Number of training samples to load into each block
        one_hot: Whether or not to use one hot encodings
        limited by the size of the buffer and size of each sample
        reg: Regularization constant
        train_path: Path to the training file (binary)

    Returns:
        The trained model
    """
    shape = (num_per_block, num_features + 1)
    w, I = np.zeros((num_features, num_classes)), np.identity(num_features)
    for p in range(epochs):
        shuffle_train(algorithm, dtype, n, num_per_block, num_features,
                      train_path)
        blocks = BlockBuffer(dtype, num_per_block, train_path, shape)
        deblockify = lambda block: block_to_x_y(block, num_classes, one_hot)
        for X, Y in map(deblockify, blocks):
            for i in range(X.shape[0]):
                x, y = np.matrix(X[i]), np.matrix(Y[i]).T
                grad = x.T.dot(x.dot(w) - y) + 2 * reg * w
                alpha = eta0 * damp ** ((n * p + i) % 1000)
                w -= alpha * grad
    return w


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
    X, y = block[:, :-1], np.matrix(block[:, -1].astype(int, copy=False)).T
    if one_hot:
        y = np.eye(num_classes)[y].reshape((X.shape[0], num_classes))
    return X, y


if __name__ == '__main__':
    main()
