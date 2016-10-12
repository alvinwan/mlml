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
    --epochs=<epochs>   Number of passes over the training data
    --eta0=<eta0>       The initial learning rate
    --damp=<damp>       Amount to multiply learning rate by per epoch
    --buffer=<num>      Size of memory in megabytes (MB)
    --dtype=<dtype>     The numeric type of each sample [default: float32]
    --d=<d>             Number of features
    --n=<n>             Number of training samples
    --train=<train>     Path to training data binary [default: data/train]
    --test=<test>       Path to test data [default: data/test.mat]
"""

import docopt
import numpy as np
import scipy
import sklearn.metrics

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


def train(
        damp: float,
        dtype: str,
        epochs: int,
        eta0: float,
        n: int,
        num_features: int,
        num_per_block: int,
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
        train_path: Path to the training file (.csv)

    Returns:
        The trained model
    """
    w = np.eye(N=num_features)
    for p in range(epochs):
        permute_train_dataset(dtype, n, train_path, num_per_block)
        blocks = BlockBuffer(dtype, num_per_block, train_path)
        for X, Y in blocks:
            for i in range(X.shape[0]):
                x, y = X[i], Y[i]
                w -= eta0*(damp**(p-1))*np.linalg.inv(x.T.dot(x)).dot(x.dot(y))
    return w


def permute_train_dataset(
        dtype: str,
        n: int,
        train_path: str,
        num_per_block: int) -> int:
    """Permute the data, and save the shuffled data to a new file named .tmp.

    See the docstring at the top of this file for a description of the
    shuffling mechanism used here, which takes advantage of spatial locality
    by loading data from disk sequentially, wherever possible.

    Args:
        dtype: Data type of samples in file
        n: Number of training samples
        num_per_block: Number of training samples to load into each block
        train_path: Path to the train file (binary)

    Returns:
        The number of training examples
    """
    indices = list(range(n))
    np.random.shuffle(indices)

    blocks = BlockBuffer(dtype, num_per_block, train_path)
    for X, Y in blocks:
        ...
    return n


class BlockBuffer:
    """File buffer that buffers blocks of data at once."""

    def __init__(self, dtype: str, num_per_block: int, path: str):
        """Initialize file handler but do not buffer data.

        Args:
            dtype: Data type of numbers in file
            num_per_block: Number of samples per block
            path: Path to the file to buffer
        """
        self.block = 0
        self.handler = np.memmap(path, dtype=dtype, mode='r')
        self.num_per_block = num_per_block

    def __next__(self) -> np.ndarray:
        """Buffer and return the next block of data.

        Returns:
            The next buffered block of data, as a numpy matrix
        """
        block = self.read_block(self.block)
        self.block += 1
        if len(block) == 0:
            raise StopIteration
        return block

    def read_block(self, block: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read block of data from shuffled data.

        Note that even though the entire I/O buffer is run through, only data
        from the current block is saved in memory and returned to the main sgd
        loop for training.

        Args:
            block: Index of the block of data to read into memory

        Returns:
            A tuple containing training inputs and outputs
        """
        raw = self.handler[block * self.num_per_block:
                           (block + 1) * self.num_per_block]
        return raw[:, :-1], raw[:, -1]

    def __iter__(self):
        return self


if __name__ == '__main__':
    main()
