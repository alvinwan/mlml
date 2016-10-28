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
    --buffer=<num>      Size of memory in megabytes (MB) [default: 10]
    --d=<d>             Number of features
    --damp=<damp>       Amount to multiply learning rate by per epoch [default: 0.99]
    --dtype=<dtype>     The numeric type of each sample [default: float64]
    --epochs=<epochs>   Number of passes over the training data [default: 5]
    --eta0=<eta0>       The initial learning rate [default: 1e-6]
    --iters=<iters>     The number of iterations, used for gd and sgd [default: 5000]
    --logfreq=<freq>    Number of iterations between log entries. 0 for no log. [default: 1000]
    --momentum=<mom>    Momentum to apply to changes in weight [default: 0.9]
    --n=<n>             Number of training samples
    --k=<k>             Number of classes [default: 10]
    --nt=<nt>           Number of testing samples
    --one-hot=<onehot>  Whether or not to use one hot encoding [default: False]
    --nthreads=<nthr>   Number of threads [default: 1]
    --reg=<reg>         Regularization constant [default: 0.1]
    --step=<step>       Number of iterations between each alpha decay [default: 10000]
    --train=<train>     Path to training data binary [default: data/train]
    --test=<test>       Path to test data [default: data/test]
"""

import datetime
import docopt
import functools
import numpy as np
import scipy
import sklearn.metrics
import time

from utils.blocks import bytes_per_dtype
from utils.blocks import BlockBuffer
from utils.shuffle import shuffle_train
from typing import Tuple


TIME = time.time()
LOG_PATH_FORMAT = 'logs/{algo}/run-{time}.csv'
LOG_HEADER = 'i,Time,Loss,Train Accuracy,Test Accuracy\n'
LOG_ENTRY_FORMAT = '{i},{time},{loss},{train_accuracy},{test_accuracy}\n'


def main() -> None:
    """Load data and launch training, then evaluate accuracy."""
    arguments = preprocess_arguments(docopt.docopt(__doc__, version='ssgd 1.0'))

    X_test, y_test = read_full_dataset(
        dtype=arguments['--dtype'],
        path=arguments['--test'],
        shape=(arguments['--nt'], arguments['--d']))
    if arguments['closed']:
        X, Y, model = train_closed(
            dtype=arguments['--dtype'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            one_hot=arguments['--one-hot'],
            reg=arguments['--reg'],
            train_path=arguments['--train'])
    elif arguments['gd']:
        X, Y, model = train_gd(
            damp=arguments['--damp'],
            dtype=arguments['--dtype'],
            eta0=arguments['--eta0'],
            iterations=arguments['--iters'],
            log_frequency=arguments['--logfreq'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            one_hot=arguments['--one-hot'],
            reg=arguments['--reg'],
            train_path=arguments['--train'],
            X_test=X_test,
            y_test=y_test)
    elif arguments['sgd']:
        X, Y, model = train_sgd(
            damp=arguments['--damp'],
            dtype=arguments['--dtype'],
            eta0=arguments['--eta0'],
            epochs=arguments['--epochs'],
            log_frequency=arguments['--logfreq'],
            momentum=arguments['--momentum'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            one_hot=arguments['--one-hot'],
            reg=arguments['--reg'],
            step=arguments['--step'],
            train_path=arguments['--train'],
            X_test=X_test,
            y_test=y_test)
    elif arguments['ssgd']:
        X, Y, model = train_ssgd(
            algorithm=arguments['--algo'],
            damp=arguments['--damp'],
            dtype=arguments['--dtype'],
            epochs=arguments['--epochs'],
            eta0=arguments['--eta0'],
            log_frequency=arguments['--logfreq'],
            momentum=arguments['--momentum'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            num_per_block=arguments['--num-per-block'],
            num_threads=arguments['--nthreads'],
            one_hot=arguments['--one-hot'],
            reg=arguments['--reg'],
            step=arguments['--step'],
            train_path=arguments['--train'],
            X_test=X_test,
            y_test=y_test)
    else:
        raise UserWarning('Invalid algorithm specified.')
    evaluate_model(model, arguments['--one-hot'], X_test, X, y_test, Y)


def preprocess_arguments(arguments) -> dict:
    """Preprocessing arguments dictionary by cleaning numeric values.

    Args:
        arguments: The dictionary of command-line arguments
    """

    if arguments['mnist']:
        arguments['--dtype'] = 'int8'
        arguments['--train'] = 'data/mnist-%s-60000-train' % arguments['--dtype']
        arguments['--test'] = 'data/mnist-%s-10000-test' % arguments['--dtype']
        arguments['--n'] = 60000
        arguments['--nt'] = 10000
        arguments['--k'] = 10
        arguments['--d'] = 784
        arguments['--one-hot'] = 'true'
    if arguments['spam']:
        arguments['--train'] = 'data/spam-%s-2760-train' % arguments['--dtype']
        arguments['--test'] = 'data/spam-%s-690-test' % arguments['--dtype']
        arguments['--n'] = 2760
        arguments['--nt'] = 690
        arguments['--k'] = 1
        arguments['--d'] = 55

    arguments['--damp'] = float(arguments['--damp'])
    arguments['--epochs'] = int(arguments['--epochs'])
    arguments['--eta0'] = float(arguments['--eta0'])
    arguments['--iters'] = int(arguments['--iters'])
    arguments['--logfreq'] = int(arguments['--logfreq'])
    arguments['--momentum'] = float(arguments['--momentum'])
    arguments['--n'] = int(arguments['--n'])
    arguments['--nthreads'] = int(arguments['--nthreads'])
    arguments['--d'] = int(arguments['--d'])
    arguments['--k'] = int(arguments['--k'])
    arguments['--one-hot'] = arguments['--one-hot'].lower() == 'true'
    arguments['--reg'] = float(arguments['--reg'])
    arguments['--step'] = int(arguments['--step'])

    bytes_total = float(arguments['--buffer']) * (10 ** 6)
    bytes_per_sample = (arguments['--d'] + 1) * bytes_per_dtype(arguments['--dtype'])
    arguments['--num-per-block'] = min(
        int(bytes_total // bytes_per_sample),
        arguments['--n'])
    return arguments


def predict_binary(
        X: np.ndarray,
        model: np.ndarray,
        threshold: float=0.5) -> np.ndarray:
    """Predict for binary classification."""
    return np.where(X.dot(model) > threshold, 1, 0)


def predict_one_hot(
        X: np.ndarray,
        model: np.ndarray) -> np.ndarray:
    """Predict for one hot vectors."""
    return de_one_hot(X.dot(model))


def de_one_hot(X: np.ndarray):
    """Convert one hot vectors back into class labels."""
    return np.argmax(X, axis=1)


def ridgeloss(
        X: np.ndarray,
        w: np.ndarray,
        Y: np.ndarray,
        reg: float):
    """Compute ridge regression loss."""
    A = X.dot(w) - Y
    return np.asscalar(np.linalg.norm(A) + reg * np.linalg.norm(w))


def evaluate_model(
        model: np.ndarray,
        one_hot: bool,
        X_test: np.ndarray,
        X_train: np.ndarray,
        y_test: np.ndarray,
        y_train: np.ndarray) -> Tuple[float, float]:
    """Evaluate the model's accuracy."""
    if one_hot:
        if y_train is not None and y_train.shape[1] > 1:
            y_train = de_one_hot(y_train)  # hacky
        y_train_hat = predict_one_hot(X_train, model)
        y_test_hat = predict_one_hot(X_test, model)
    else:
        y_train_hat = predict_binary(X_train, model)
        y_test_hat = predict_binary(X_test, model)
    train_accuracy = sklearn.metrics.accuracy_score(y_train, y_train_hat)
    test_accuracy = sklearn.metrics.accuracy_score(y_test, y_test_hat)
    print('Train Accuracy:', train_accuracy, 'Test Accuracy:', test_accuracy)
    return train_accuracy, test_accuracy


def read_full_dataset(
        dtype: str,
        path: str,
        shape: Tuple[int, int],
        num_classes: int=10,
        one_hot: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """Read the dataset in its entirety."""
    data = np.memmap(path, dtype=dtype, mode='r', shape=(shape[0], shape[1] + 1))
    return block_to_x_y(data, num_classes, one_hot)


def timeit(f):
    """Times the function that it decorates."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        a = datetime.datetime.now()
        rv = f(*args, **kwargs)
        b = datetime.datetime.now()
        c = b - a
        print('Time (s):', c.total_seconds())
        return rv
    return wrapper


@timeit
def train_closed(
        dtype: str,
        n: int,
        num_classes: int,
        num_features: int,
        one_hot: bool,
        reg: float,
        train_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return X, y, scipy.linalg.solve(XTX + reg*I, XTy, sym_pos=True)


@timeit
def train_gd(
        damp: float,
        dtype: str,
        eta0: float,
        iterations: int,
        log_frequency: int,
        n: int,
        num_features: int,
        num_classes: int,
        one_hot: bool,
        reg: float,
        train_path: str,
        X_test: np.ndarray,
        y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train using gradient descent.

    Args:
        damp: Amount to multiply learning rate at each epoch
        dtype: Data type of numbers in file
        eta0: Starting learning rate
        iterations: Number of iterations to train
        log_frequency: Number of iterations between log entries
        n: Number of training samples
        num_classes: Number of classes
        num_features: Number of features
        one_hot: Whether or not to train with one hot encodings
        limited by the size of the buffer and size of each sample
        reg: Regularization constant
        train_path: Path to the training file (binary)
        X_test: Test input data
        y_test: Test output data

    Returns:
        The trained model
    """
    with open(LOG_PATH_FORMAT.format(time=TIME, algo='gd'), 'w') as f:
        f.write(LOG_HEADER)
        shape = (n, num_features)
        X, Y = read_full_dataset(dtype, train_path, shape, num_classes, one_hot)
        XTX, XTy, w = X.T.dot(X), X.T.dot(Y), np.zeros((num_features, num_classes))
        for i in range(iterations):
            grad = XTX.dot(w) - XTy + 2*reg*w
            alpha = eta0*damp**(i % 100)
            w -= alpha * grad

            if i % log_frequency == 0:
                train_accuracy, test_accuracy = evaluate_model(
                    w, one_hot, X_test, X, y_test, Y)
                f.write(LOG_ENTRY_FORMAT.format(
                    i=i,
                    time=time.time() - TIME,
                    loss=ridgeloss(X, w, Y, reg),
                    train_accuracy=train_accuracy,
                    test_accuracy=test_accuracy))
    return X, Y, w


@timeit
def train_sgd(
        damp: float,
        dtype: str,
        eta0: float,
        epochs: int,
        log_frequency: int,
        momentum: float,
        n: int,
        num_classes: int,
        num_features: int,
        one_hot: bool,
        reg: float,
        step: int,
        train_path: str,
        X_test: np.ndarray,
        y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train using stochastic gradient descent.

    Args:
        damp: Amount to multiply learning rate at each epoch
        dtype: Data type of numbers in file
        eta0: Starting learning rate
        epochs: Number of passes over training data
        log_frequency: Number of iterations between log entries
        momentum: Momentum to apply to changes in w
        n: Number of training samples
        num_classes: Number of classes
        num_features: Number of features
        one_hot: Whether or not to use one hot encodings
        limited by the size of the buffer and size of each sample
        reg: Regularization constant
        step: Number of iterations between each alpha decay
        train_path: Path to the training file (binary)
        X_test: Test input data
        y_test: Test output data

    Returns:
        The trained model
    """
    with open(LOG_PATH_FORMAT.format(time=TIME, algo='sgd'), 'w') as f:
        f.write(LOG_HEADER)
        shape, w_delta = (n, num_features), 0
        X, Y = read_full_dataset(dtype, train_path, shape, num_classes, one_hot)
        w = np.zeros((num_features, num_classes))
        for p in range(epochs):
            indices = list(range(X.shape[0]))
            np.random.shuffle(indices)
            for i, index in enumerate(indices):
                x, y = np.matrix(X[index]), np.matrix(Y[index])
                grad = x.T.dot(x.dot(w) - y) + reg * w
                alpha = eta0 * damp ** ((n * p + i) // step)
                w_delta = alpha * grad + momentum * w_delta
                w -= w_delta

                if (i + p * X.shape[0]) % log_frequency == 0:
                    train_accuracy, test_accuracy = evaluate_model(
                        w, one_hot, X_test, X, y_test, Y)
                    f.write(LOG_ENTRY_FORMAT.format(
                        i=i + p * X.shape[0],
                        time=time.time() - TIME,
                        loss=ridgeloss(X, w, Y, reg),
                        train_accuracy=train_accuracy,
                        test_accuracy=test_accuracy))
        print('=' * 30, '\n * SGD : Epoch {p} finished.'.format(p=p))
    return X, Y, w


@timeit
def train_ssgd(
        algorithm: str,
        damp: float,
        dtype: str,
        epochs: int,
        eta0: float,
        log_frequency: int,
        momentum: float,
        n: int,
        num_classes: int,
        num_features: int,
        num_per_block: int,
        num_threads: int,
        one_hot: bool,
        reg: float,
        step: int,
        train_path: str,
        X_test: np.ndarray,
        y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        log_frequency: Number of iterations between log entries
        momentum: Momentum to apply to changes in w
        n: Number of training samples
        num_classes: Number of classes
        num_features: Number of features
        num_per_block: Number of training samples to load into each block
        one_hot: Whether or not to use one hot encodings
        limited by the size of the buffer and size of each sample
        reg: Regularization constant
        step: Number of iterations between each alpha decay
        train_path: Path to the training file (binary)
        X_test: Test input data
        y_test: Test output data

    Returns:
        The trained model
    """
    with open(LOG_PATH_FORMAT.format(time=TIME, algo='ssgd'), 'w') as f:
        f.write(LOG_HEADER)
        w, I = np.zeros((num_features, num_classes)), np.identity(num_features)
        w_delta = 0
        for p in range(epochs):
            shuffled_path = shuffle_train(
                algorithm, dtype, n, num_per_block, num_features, train_path)
            blocks = BlockBuffer(dtype, n, num_features + 1, num_per_block, shuffled_path)
            deblockify = lambda block: block_to_x_y(block, num_classes, one_hot)
            for b, (X, Y) in enumerate(map(deblockify, blocks)):
                if Y.shape[0] < Y.shape[1]:
                    Y.shape = (X.shape[0], 1)  # hacky
                for i in range(X.shape[0]):
                    x, y = np.matrix(X[i]), np.matrix(Y[i])
                    grad = x.T.dot(x.dot(w) - y) + reg * w
                    alpha = eta0 * damp ** ((n * p + i) // step)
                    w_delta = alpha * grad + momentum * w_delta
                    w -= w_delta

                    if (i + p * X.shape[0]) % log_frequency == 0:
                        train_accuracy, test_accuracy = evaluate_model(
                            w, one_hot, X_test, X, y_test, Y)
                        f.write(LOG_ENTRY_FORMAT.format(
                            i=i + b * num_per_block + p * X.shape[0],
                            time=time.time() - TIME,
                            loss=ridgeloss(X, w, Y, reg),
                            train_accuracy=train_accuracy,
                            test_accuracy=test_accuracy))
            print('='*30, '\n * SGD : Epoch {p} finished.'.format(p=p))
    return X, Y, w


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
