"""Streaming Stochastic Gradient Descent implementation, starter task.

Shuffling Mechanism
---

The shuffling mechanism is linear w.r.t. the number of samples, taking linear
extra space. This was programmed considering that sequential access to disk
is relatively fast and that random accesses are slow.

First, we consider the number of samples (n). Then, randomly shuffle a list of
*indices*. Let us assume that memory can hold at most some finite number of
samples (N). Take the first N indices from the shuffled list of indices, and
then make one pass over all samples, using a buffer, to pick samples that are
included in the set of N indices. The buffer ensures that we are not holding
the entire file in memory. We now have N randomly-selected samples in memory.
Write these N onto disk, and repeat for each chunk of N samples, for all n.
With each write, we process another set of N indices from our n, finally
resulting in a complete shuffling of all n samples.

The function below implements this by reading from and writing to files.


Reading Mechanism
---

As described in the starter task PDF, all samples are read sequentially from
disk. Again using a buffer, we simply read the next chunk of N samples that
are needed for sgd to run another block.


CSV File Format
---

Each csv must contain the number of samples at the top of the file. Each line
is then formatted with two comma-separated values, where the first is a list
of values and the second is a single integer.


Usage:
    ssgd.py

Options:
    --epochs=<epochs>  Number of passes over the training data
    --eta0=<eta0>      The initial learning rate
    --damp=<damp>      Amount to multiply learning rate by per epoch
    --blocknum=<num>   Number of samples memory can hold, at maximum
    --train=<train>    Path to train file. (.csv) [default: data/train.csv]
    --test=<test>      Path to test file. (.csv) [default: data/test.csv]
"""

import docopt
import json
import numpy as np
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
        num_per_block=arguments['<num>'])
    output(model, arguments['<test>'])


def output(model: np.ndarray, test_path: str) -> None:
    """Output the test results.

    Args:
        model: The trained model
        test_path: Path to the test file (.csv)
    """
    X_test, y_test = load_test_dataset(test_path)
    y_hat = model.dot(X_test)
    print('Accuracy:', sklearn.metrics.accuracy_score(y_test, y_hat))


def load_test_dataset(test_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and give test data in memory.

    Args:
        test_path: Path to the test file (.csv)

    Returns:
        A tuple containing the test input, then the test output
    """
    X_test, Y_test = [], []
    with open(test_path) as f:
        next(f)
        for line in f.readlines():
            x, y = line.split(',')
            X_test.append(json.loads(x))
            Y_test.append(y)
    return np.array(X_test), np.array(Y_test)


def train(
        train_path: str,
        epochs: int,
        eta0: float,
        damp: float,
        num_per_block: int) -> np.ndarray:
    """Train using stochastic gradient descent.

    Args:
        train_path: Path to the training file (.csv)
        epochs: Number of passes over training data
        eta0: Starting learning rate
        damp: Amount to multiply learning rate at each epoch
        num_per_block: Number of training samples to load into each block,
        limited by the size of the buffer and size of each sample

    Returns:
        The trained model
    """
    w = np.eye(N=NUM_FEATURES)
    for p in range(epochs):
        n = permute_train_dataset(train_path, num_per_block)
        for t in range(n):
            if t % num_per_block == 0:
                X, Y = read_block(train_path, t // num_per_block, num_per_block)
            x, y = X[t % num_per_block], Y[t % num_per_block]
            w -= eta0*(damp**(p-1))*np.linalg.inv(x.T.dot(x)).dot(x.dot(y))
    return w


def permute_train_dataset(train_path: str, num_per_block: int) -> int:
    """Permute the data, and save the shuffled data to a new file named .tmp.

    See the docstring at the top of this file for a description of the
    shuffling mechanism used here, which takes advantage of spatial locality
    by loading data from disk sequentially, wherever possible.

    Args:
        train_path: Path to the train file (.csv)

    Returns:
        The number of training examples
    """
    with open(train_path) as f:
        n = int(next(f))
        indices = list(range(n))
        np.random.shuffle(indices)

    with open(train_path + '.tmp', 'w') as tmp:
        for i in range(np.ceil(n / num_per_block)):
            block_indices = set(indices[i*num_per_block:(i+1)*num_per_block])
            with open(train_path) as f:
                for i, line in enumerate(f.readlines()):
                    if i in block_indices:
                        tmp.write(line)

    return n


def read_block(train_path: str, block: int, num_per_block: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Read block of data from shuffled data.

    Note that even though the entire I/O buffer is run through, only data
    from the current block is saved in memory and returned to the main sgd
    loop for training.

    Args:
        train_path: Path to the training file (.csv)
        block: Index of the block of data to read into memory
        num_per_block: Number of data points per block (to load into memory)

    Returns:
        A tuple containing training inputs and outputs
    """
    X_train, Y_train = [], []
    with open(train_path + '.tmp') as f:
        next(f)
        for i, line in enumerate(f.readlines()):
            if block * num_per_block < i < (block + 1) * num_per_block:
                x, y = line.split(',')
                X_train.append(json.loads(x))
                Y_train.append(y)
    return np.array(X_train), np.array(Y_train)


if __name__ == '__main__':
    main()
