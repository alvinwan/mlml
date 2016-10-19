"""Streaming Stochastic Gradient Descent implementation, starter task.

Usage:
    ssgd.py closed --dtype=<dtype> --n=<n> --num_features=<num_features>
                   --reg=<reg> --train=<train>
    ssgd.py gd
    ssgd.py sgd
    ssgd.py ssgd [--epochs=<epochs>] [--eta0=<eta0>] [--damp=<damp>]
    [--blocknum=<num>] [--train=<train>] [--test=<test>]

Options:
    --algo=<algo>       Shuffling algorithm to use
    --buffer=<num>      Size of memory in megabytes (MB)
    --d=<d>             Number of features
    --damp=<damp>       Amount to multiply learning rate by per epoch
    --dtype=<dtype>     The numeric type of each sample [default: float32]
    --epochs=<epochs>   Number of passes over the training data
    --eta0=<eta0>       The initial learning rate
    --iters=<iters>     The number of iterations, used for gd and sgd
    --n=<n>             Number of training samples
    --reg=<reg>         Regularization constant
    --train=<train>     Path to training data binary [default: data/train]
    --test=<test>       Path to test data [default: data/test]
"""

import docopt
import numpy as np
import sklearn.metrics

from blocks import BlockBuffer
from shuffle import shuffle_train
from typing import Tuple


def main() -> None:
    """Load data and launch training, then evaluate accuracy."""
    arguments = docopt.docopt(__doc__, version='ssgd 1.0')

    algorithm = arguments['--algo']
    damp = float(arguments['--damp'])
    dtype = arguments['--dtype']
    epochs = int(arguments['--epochs'])
    eta0 = float(arguments['--eta0'])
    iterations = int(arguments['--iters'])
    n = int(arguments['--n'])
    num_features = int(arguments['--d'])
    num_per_block = int((float(arguments['--buffer']) * (10 ** 6)) // 4)
    reg = float(arguments['--reg'])
    train_path = arguments['--train']

    if arguments['closed']:
        model = train_closed(
            dtype=dtype,
            n=n,
            num_features=num_features,
            reg=reg,
            train_path=train_path)
    elif arguments['gd']:
        model = train_gd(
            damp=damp,
            dtype=dtype,
            eta0=eta0,
            iterations=iterations,
            n=n,
            num_features=num_features,
            reg=reg,
            train_path=train_path)
    elif arguments['sgd']:
        model = train_sgd(
            damp=damp,
            dtype=dtype,
            eta0=eta0,
            iterations=iterations,
            n=n,
            num_features=num_features,
            reg=reg,
            train_path=train_path)
    elif arguments['ssgd']:
        model = train_ssgd(
            algorithm=algorithm,
            damp=damp,
            dtype=dtype,
            epochs=epochs,
            eta0=eta0,
            n=n,
            num_features=num_features,
            num_per_block=num_per_block,
            reg=reg,
            train_path=train_path)
    else:
        raise UserWarning('Invalid algorithm specified.')
    X_test, y_test = read_full_dataset(arguments['--test'])
    y_hat = np.round(model.dot(X_test))
    print('Accuracy:', sklearn.metrics.accuracy_score(y_test, y_hat))


def read_full_dataset(
        dtype: str,
        path: str,
        shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Read the dataset in its entirety."""
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)
    return block_to_x_y(data)


def train_closed(
        dtype: str,
        n: int,
        num_features: int,
        reg: float,
        train_path: str) -> np.ndarray:
    """Compute the closed form solution.

    Returns:
        The trained model
    """
    X, y = read_full_dataset(dtype, train_path, (n, num_features))
    XTX, I, XTy = X.T.dot(X), np.identity(num_features), X.T.dot(y)
    return np.linalg.solve(XTX + reg*I, XTy)


def train_gd(
        damp: float,
        dtype: str,
        eta0: float,
        iterations: int,
        n: int,
        num_features: int,
        reg: float,
        train_path: str) -> np.ndarray:
    """Train using gradient descent.

    Returns:
        The trained model
    """
    X, y = read_full_dataset(dtype, train_path, (n, num_features))
    XTX, XTy, w = X.T.dot(X), X.T.dot(y), np.zeros((num_features, 1))
    for i in range(iterations):
        grad = XTX.dot(w) - XTy + 2*reg*w
        alpha = eta0*damp**(i % 100)
        w += alpha * grad
    return w


def train_sgd(
        damp: float,
        dtype: str,
        eta0: float,
        iterations: int,
        n: int,
        num_features: int,
        reg: float,
        train_path: str) -> np.ndarray:
    """Train using stochastic gradient descent.

    Returns:
        The trained model
    """
    X, y = read_full_dataset(dtype, train_path, (n, num_features))
    w = np.zeros((num_features, 1))
    for i in range(iterations):
        x, y = np.matrix(X[i]), np.asscalar(y[i])
        grad = x.T.dot(x.dot(w) - y) + 2 * reg * w
        alpha = eta0 * damp ** (i % 100)
        w += alpha * grad
    return w


def train_ssgd(
        algorithm: str,
        damp: float,
        dtype: str,
        epochs: int,
        eta0: float,
        n: int,
        num_features: int,
        num_per_block: int,
        reg: float,
        train_path: str) -> np.ndarray:
    """Train using streaming stochastic gradient descent.

    The shuffling algorithms are described and discussed in shuffle.py.

    As described in the starter task PDF, all samples are read sequentially from
    disk. Again using a buffer, we simply read the next chunk of N samples that
    are needed for sgd to run another block.

    Returns:
        The trained model
    """
    w, I = np.eye(N=num_features), np.identity(num_features)
    for p in range(epochs):
        shuffle_train(algorithm, dtype, n, num_per_block, train_path)
        blocks = BlockBuffer(dtype, num_per_block, train_path)
        for X, Y in map(block_to_x_y, blocks):
            for i in range(X.shape[0]):
                x, y = X[i], Y[i]
                grad = np.linalg.inv(x.T.dot(x) + reg*I).dot(x.dot(y))
                w -= eta0*(damp**(p-1))*grad
    return w


def block_to_x_y(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a block of data into X and Y.

    Args:
        block: The block of data to extract X and Y from

    Returns:
        X: the data inputs
        Y: the data outputs
    """
    return block[:, :-1], block[:, -1]


if __name__ == '__main__':
    main()
