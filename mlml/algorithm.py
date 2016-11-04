import time
from typing import Callable
from typing import Tuple

import numpy as np

from mlml.ssgd.blocks import BlockBuffer
from mlml.ssgd.shuffle import emulate_external_shuffle
from mlml.ssgd.shuffle import shuffle_train
from mlml.utils.data import block_x_labels
from mlml.utils.data import read_dataset
from mlml.utils.data import to_one_hot
from mlml.logging import Logger
from mlml.logging import StandardLogger
from mlml.loss import Loss
from mlml.loss import RidgeRegression
from mlml.model import RegressionModel
from mlml.model import Model

TIME = time.time()
LOG_PATH_FORMAT = 'logs/{algo}/run-{time}.csv'
LOG_HEADER = 'i,Time,Loss,Train Accuracy,Test Accuracy\n'
LOG_ENTRY_FORMAT = '{i},{time},{loss},{train_accuracy},{test_accuracy}\n'


class Algorithm:
    """General structure and API for an algorithm."""

    @classmethod
    def from_arguments(
            cls,
            arguments: dict,
            X_test: np.ndarray,
            y_test: np.ndarray):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError


class ClosedForm(Algorithm):
    """Run the closed form solution, if one exists."""

    @classmethod
    def from_arguments(
            cls,
            arguments: dict,
            X_test: np.ndarray,
            y_test: np.ndarray):
        return ClosedForm().train(
            data_hook=arguments['--data-hook'],
            dtype=arguments['--dtype'],
            loss=RidgeRegression(arguments['--reg']),
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            one_hot=arguments['--one-hot'],
            train_path=arguments['--train'])

    def train(
            self,
            data_hook: Callable[[np.ndarray, np.ndarray], Tuple],
            dtype: str,
            loss: Loss,
            n: int,
            num_classes: int,
            num_features: int,
            one_hot: bool,
            train_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the closed form solution.

        Args:
            data_hook: Processes data for given scenario
            dtype: Data type of numbers in file
            loss: Loss function
            n: Number of training samples
            num_classes: Number of classes
            num_features: Number of features
            limited by the size of the buffer and size of each sample
            one_hot: Whether or not to use one hot encodings
            train_path: Path to the training file (binary)

        Returns:
            The trained model
        """
        shape = (n, num_features)
        data = read_dataset(
            data_hook, dtype, num_classes, one_hot, train_path, shape)
        return data.X, data.Y, loss.closed_form(data.X, data.Y)


class GD(Algorithm):
    """Gradient Descent Algorithm"""

    @classmethod
    def from_arguments(
            cls,
            arguments: dict,
            X_test: np.ndarray,
            y_test: np.ndarray):
        return GD().train(
            damp=arguments['--damp'],
            data_hook=arguments['--data-hook'],
            dtype=arguments['--dtype'],
            eta0=arguments['--eta0'],
            iterations=arguments['--iters'],
            loss=RidgeRegression(arguments['--reg']),
            logger=StandardLogger(),
            log_frequency=arguments['--logfreq'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            one_hot=arguments['--one-hot'],
            step=arguments['--step'],
            train_path=arguments['--train'],
            X_test=X_test,
            labels_test=y_test)

    def train(
            self,
            damp: float,
            data_hook: Callable[[np.ndarray, np.ndarray], Tuple],
            dtype: str,
            eta0: float,
            iterations: int,
            logger: Logger,
            loss: Loss,
            log_frequency: int,
            n: int,
            num_features: int,
            num_classes: int,
            one_hot: bool,
            step: int,
            train_path: str,
            X_test: np.ndarray,
            labels_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Model]:
        """Train using gradient descent.

        Args:
            damp: Amount to multiply learning rate at each epoch
            data_hook: Processes data for given scenario
            dtype: Data type of numbers in file
            eta0: Starting learning rate
            iterations: Number of iterations to train
            logger: The logging utility
            log_frequency: Number of iterations between log entries
            loss: Loss function
            n: Number of training samples
            num_classes: Number of classes
            num_features: Number of features
            one_hot: Whether or not to train with one hot encodings
            limited by the size of the buffer and size of each sample
            step: Number of iterations between each alpha decay
            train_path: Path to the training file (binary)
            X_test: Test input data
            labels_test: Test output data

        Returns:
            The trained model
        """
        f = open(LOG_PATH_FORMAT.format(time=TIME, algo='gd'), 'w')
        f.write(LOG_HEADER)
        shape = (n, num_features)
        data = read_dataset(
            data_hook, dtype, num_classes, one_hot, train_path, shape)
        loss.pre_hook(data.X, data.Y)
        model = RegressionModel.initialize_zero(num_features, num_classes)
        for i in range(iterations):
            grad = loss.gradient(model)
            alpha = eta0 * damp ** (i // step)
            model.w -= alpha * grad
            logger.iteration(
                i, f, log_frequency, loss, model, data.X, data.Y,
                data.labels, X_test, labels_test)
        f.close()
        return data.X, data.Y, model


class SGD(Algorithm):
    """Stochastic Gradient Descent Algorithm"""

    @classmethod
    def from_arguments(
            cls,
            arguments: dict,
            X_test: np.ndarray,
            y_test: np.ndarray):
        return SGD().train(
            damp=arguments['--damp'],
            data_hook=arguments['--data-hook'],
            dtype=arguments['--dtype'],
            eta0=arguments['--eta0'],
            epochs=arguments['--epochs'],
            logger=StandardLogger(),
            log_frequency=arguments['--logfreq'],
            loss=RidgeRegression(arguments['--reg']),
            momentum=arguments['--momentum'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            num_per_block=arguments['--num-per-block'],
            one_hot=arguments['--one-hot'],
            step=arguments['--step'],
            train_path=arguments['--train'],
            X_test=X_test,
            labels_test=y_test)

    def train(
            self,
            damp: float,
            data_hook: Callable[[np.ndarray, np.ndarray], Tuple],
            dtype: str,
            eta0: float,
            epochs: int,
            logger: Logger,
            log_frequency: int,
            loss: Loss,
            momentum: float,
            n: int,
            num_classes: int,
            num_features: int,
            num_per_block: int,
            one_hot: bool,
            step: int,
            train_path: str,
            X_test: np.ndarray,
            labels_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Model]:
        """Train using stochastic gradient descent.

        Args:
            damp: Amount to multiply learning rate at each epoch
            data_hook: Processes data for given scenario
            dtype: Data type of numbers in file
            eta0: Starting learning rate
            epochs: Number of passes over training data
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
        f = open(LOG_PATH_FORMAT.format(time=TIME, algo='sgd'), 'w')
        f.write(LOG_HEADER)
        shape, w_delta, iteration = (n, num_features), 0, 0
        data = read_dataset(
            data_hook, dtype, num_classes, one_hot, train_path, shape)
        model = RegressionModel.initialize_zero(num_features, num_classes)
        for p in range(epochs):
            indices = np.arange(0, data.X.shape[0])
            emulate_external_shuffle(num_per_block, indices)
            for i, index in enumerate(indices):
                grad = loss.gradient(model, data.X[index], data.Y[index])
                alpha = eta0 * damp ** (iteration // step)
                w_delta = alpha * grad + momentum * w_delta
                model.w -= w_delta
                logger.iteration(
                    iteration, f, log_frequency, loss, model, data.X,
                    data.Y, data.labels, X_test, labels_test)
                iteration += 1
            logger.epoch(p)
        f.close()
        return data.X, data.Y, model


class SSGD(Algorithm):
    """Streaming Stochastic Gradient Descent"""

    @classmethod
    def from_arguments(
            cls,
            arguments: dict,
            X_test: np.ndarray,
            y_test: np.ndarray):
        return SSGD().train(
            algorithm=arguments['--algo'],
            damp=arguments['--damp'],
            data_hook=arguments['--data-hook'],
            dtype=arguments['--dtype'],
            epochs=arguments['--epochs'],
            eta0=arguments['--eta0'],
            logger=StandardLogger(),
            log_frequency=arguments['--logfreq'],
            loss=RidgeRegression(arguments['--reg']),
            momentum=arguments['--momentum'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            num_per_block=arguments['--num-per-block'],
            one_hot=arguments['--one-hot'],
            simulated=arguments['--simulated'],
            step=arguments['--step'],
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
            train_path: str,
            X_test: np.ndarray,
            labels_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Model]:
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
        if simulated:
            shape = (n, num_features)
            data = read_dataset(
                data_hook, dtype, num_classes, one_hot, train_path, shape)
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
                            iteration, f, log_frequency, loss, model, data.X,
                            data.Y, data.labels, X_test, labels_test)
                    else:
                        logger.iteration(
                            iteration, f, log_frequency, loss, model, X, Y,
                            labels, X_test, labels_test)
                    iteration += 1
            logger.epoch(p)
        f.close()
        return data.X, data.Y, model