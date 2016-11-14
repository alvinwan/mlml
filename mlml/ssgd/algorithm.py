"""Streaming Stochastic Gradient Descent implementation"""

import numpy as np

from typing import Tuple
from typing import Callable

from mlml.algorithm import Algorithm
from mlml.logging import Logger
from mlml.logging import LOG_PATH_FORMAT
from mlml.logging import LOG_HEADER
from mlml.logging import StandardLogger
from mlml.logging import TIME
from mlml.loss import Loss
from mlml.loss import RidgeRegression
from mlml.model import RegressionModel
from mlml.model import Model
from mlml.ssgd.blocks import BlockBuffer
from mlml.ssgd.shuffle import shuffle_train
from mlml.utils.data import block_x_labels
from mlml.utils.data import read_dataset
from mlml.utils.data import to_one_hot
from mlml.utils.data import Data


class SSGD(Algorithm):
    """Streaming Stochastic Gradient Descent"""

    @classmethod
    def from_arguments(
            cls,
            arguments: dict,
            X_test: np.ndarray,
            y_test: np.ndarray,
            logger: Logger=StandardLogger(),
            loss: Loss=RidgeRegression(0.1)) -> Tuple[Data, Model]:
        return SSGD().train(
            algorithm=arguments['--algo'],
            damp=arguments['--damp'],
            data_hook=arguments['--data-hook'],
            dtype=arguments['--dtype'],
            epochs=arguments['--epochs'],
            eta0=arguments['--eta0'],
            logger=StandardLogger(),
            log_frequency=arguments['--logfreq'],
            loss=loss,
            momentum=arguments['--momentum'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            num_per_block=arguments['--num-per-block'],
            one_hot=arguments['--one-hot'],
            simulated=arguments['--simulated'],
            step=arguments['--step'],
            subset=arguments['--subset'],
            train_path=arguments['--train'],
            X_test=X_test,
            labels_test=y_test)

    def train(
            self,
            algorithm: str,
            damp: float,
            data_hook: Callable[[np.ndarray, np.ndarray], Tuple],
            dtype: str,
            epochs: int,
            eta0: float,
            logger: Logger,
            log_frequency: int,
            loss: Loss,
            momentum: float,
            n: int,
            num_classes: int,
            num_features: int,
            num_per_block: int,
            one_hot: bool,
            simulated: bool,
            step: int,
            subset: int,
            train_path: str,
            X_test: np.ndarray,
            labels_test: np.ndarray) -> Tuple[Data, Model]:
        """Train using streaming stochastic gradient descent.

        The shuffling algorithms are described and discussed in shuffle.py.

        As described in the starter task PDF, all samples are read sequentially from
        disk. Again using a buffer, we simply read the next chunk of N samples that
        are needed for sgd to run another block.

        If memory constraints are simulated, the below function will load the entire
        dataset into memory, to evaluate train accuracy.

        Args:
            algorithm: Shuffling algorithm to use
            damp: Amount to multiply learning rate at each epoch
            data_hook: Processes data for given scenario
            dtype: Data type of numbers in file
            epochs: Number of passes over training data
            eta0: Starting learning rate
            logger: The logging utility
            log_frequency: Number of iterations between log entries
            loss: Loss function
            momentum: Momentum to apply to changes in w
            n: Number of training samples
            num_classes: Number of classes
            num_features: Number of features
            num_per_block: Number of training samples to load into each block
            one_hot: Whether or not to use one hot encodings
            limited by the size of the buffer and size of each sample
            step: Number of iterations between each alpha decay
            train_path: Path to the training file (binary)
            X_test: Test input data
            labels_test: Test output data

        Returns:
            The trained model
        """
        train = None
        if simulated:
            shape = (n, num_features)
            train = read_dataset(
                data_hook, dtype, num_classes, one_hot, train_path, shape, 10)
        f = open(LOG_PATH_FORMAT.format(time=TIME, algo='ssgd'), 'w')
        f.write(LOG_HEADER)
        model = RegressionModel.initialize_zero(num_features, num_classes)
        w_delta = iteration = 0
        for p in range(epochs):
            shuffled_path = shuffle_train(
                algorithm, dtype, n, num_per_block, num_features, train_path)
            blocks = BlockBuffer(
                dtype, n, num_features + 1, num_per_block, shuffled_path)
            for X, labels in map(block_x_labels, blocks):
                X, labels = data_hook(X, labels)
                Y = to_one_hot(num_classes, labels) if one_hot else labels
                for i in range(X.shape[0]):
                    grad = loss.gradient(model, X[i], Y[i])
                    alpha = eta0 * damp ** (iteration // step)
                    w_delta = alpha * grad + momentum * w_delta
                    model.w -= w_delta

                    if simulated:
                        logger.iteration(
                            iteration, f, log_frequency, loss, model, train.X,
                            train.Y, train.labels, X_test, labels_test)
                    else:
                        logger.iteration(
                            iteration, f, log_frequency, loss, model, X, Y,
                            labels, X_test, labels_test)
                    iteration += 1
            logger.epoch(p)
        f.close()
        if train is None:
            train = Data(labels, num_classes, one_hot, X)
        return train, model