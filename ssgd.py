"""Streaming Stochastic Gradient Descent implementation, starter task.

Shuffling Mechanism
---

We use an external sorting algorithm by first assigning random indices to
each sample. Then, sort by indices, where the external merge algorithm takes
O(nlogn) time.

Take each block of N samples, and sort in-memory. Save each block to disk. For
each of k blocks, buffer N/k samples. Then perform a (k-1)-way merge using k
file buffers, where we load the next N/k samples any time a buffer empties.


Reading Mechanism
---

As described in the starter task PDF, all samples are read sequentially from
disk. Again using a buffer, we simply read the next chunk of N samples that
are needed for sgd to run another block.


Binary File
---

By default, each entry in the binary file is assumed to be a


Usage:
    ssgd.py [--epochs=<epochs>] [--eta0=<eta0>] [--damp=<damp>]
    [--blocknum=<num>] [--train=<train>] [--test=<test>]

Options:
    --buffer=<num>      Size of memory in megabytes (MB)
    --d=<d>             Number of features
    --damp=<damp>       Amount to multiply learning rate by per epoch
    --dtype=<dtype>     The numeric type of each sample [default: float32]
    --epochs=<epochs>   Number of passes over the training data
    --eta0=<eta0>       The initial learning rate
    --n=<n>             Number of training samples
    --reg=<reg>         Regularization constant
    --train=<train>     Path to training data binary [default: data/train]
    --test=<test>       Path to test data [default: data/test.mat]
"""

import docopt
import numpy as np
import scipy
import sklearn.metrics

from blocks import BlockBuffer
from blocks import BlockScope
from blocks import BlockWriter
from typing import Tuple


def main() -> None:
    """Load data and launch training, then evaluate accuracy."""
    arguments = docopt.docopt(__doc__, version='ssgd 1.0')
    model = train(
        damp=float(arguments['--damp']),
        dtype=arguments['--dtype'],
        epochs=int(arguments['--epochs']),
        eta0=float(arguments['--eta0']),
        n=int(arguments['--n']),
        num_features=int(arguments['--d']),
        num_per_block=int((float(arguments['--buffer']) * (10**6)) // 4),
        reg=float(arguments['--reg']),
        train_path=arguments['--train'])
    X_test, y_test = load_test_dataset(arguments['--test'])
    y_hat = np.round(model.dot(X_test))
    print('Accuracy:', sklearn.metrics.accuracy_score(y_test, y_hat))


def load_test_dataset(test_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and give test data in memory.

    Args:
        test_path: Path to the test file (.mat)

    Returns:
        A tuple containing the test input, then the test output
    """
    data = scipy.io.loadmat(test_path)
    return data['Xtest'], data['ytest']


def block_to_x_y(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a block of data into X and Y.

    Args:
        block: The block of data to extract X and Y from

    Returns:
        X: the data inputs
        Y: the data outputs
    """
    return block[:, :-1], block[:, -1]


def train(
        damp: float,
        dtype: str,
        epochs: int,
        eta0: float,
        n: int,
        num_features: int,
        num_per_block: int,
        reg: float,
        train_path: str) -> np.ndarray:
    """Train using stochastic gradient descent.

    Args:
        damp: Amount to multiply learning rate at each epoch
        dtype: Data type of numbers in file
        epochs: Number of passes over training data
        eta0: Starting learning rate
        n: Number of training samples
        num_features: Number of features
        num_per_block: Number of training samples to load into each block
        limited by the size of the buffer and size of each sample
        reg: The regularization constant
        train_path: Path to the training file (.csv)

    Returns:
        The trained model
    """
    w, I = np.eye(N=num_features), np.identity(num_features)
    for p in range(epochs):
        permute_train_dataset(dtype, n, num_per_block, train_path)
        blocks = BlockBuffer(dtype, num_per_block, train_path)
        for X, Y in map(block_to_x_y, blocks):
            for i in range(X.shape[0]):
                x, y = X[i], Y[i]
                grad = np.linalg.inv(x.T.dot(x) + reg*I).dot(x.dot(y))
                w -= eta0*(damp**(p-1))*grad
    return w


def permute_train_dataset(
        dtype: str,
        n: int,
        num_per_block: int,
        train_path: str):
    """Permute the data, and save the shuffled data to a new file named .tmp.

    See the docstring at the top of this file for a description of the
    shuffling mechanism used here, which takes advantage of spatial locality
    by loading data from disk sequentially, wherever possible.

    *Note* May cause issues if num_per_block is not evenly divided by the
    number of buffers. This means some entry in the block will be left empty.

    Args:
        dtype: Data type of samples in file
        n: Number of total samples
        num_per_block: Number of training samples to load into each block
        train_path: Path to the train file (binary)
    """
    num_buffers = n / num_per_block
    with BlockScope('float64', 'samplesort', num_per_block) as scope:
        writer = BlockWriter(dtype, num_per_block, train_path)
        blocks = BlockBuffer(dtype, num_per_block, train_path)
        buffers = []

        for i, block in enumerate(blocks):
            scope.write_block(i, np.random.shuffle(block))
            buffer = scope.get_block_buffer(i)
            buffer.num_per_block = num_per_block // num_buffers
            buffers.append(buffer)

        while buffers:
            current_block, remaining_buffers = [], []
            for buffer in buffers:
                block = buffer.read_block()
                if len(block) > 0:
                    current_block.extend(block)
                    remaining_buffers.append(buffer)
            current_block = np.matrix(current_block)
            np.random.shuffle(current_block)
            writer.write(current_block)
            buffers = remaining_buffers


if __name__ == '__main__':
    main()
