"""Class of non-specialized algorithms for comparison"""

from typing import Callable
from typing import Tuple

import numpy as np

from mlml.ssgd.shuffle import emulate_external_shuffle
from mlml.utils.data import read_dataset
from mlml.utils.data import Data
from mlml.logging import Logger
from mlml.logging import StandardLogger
from mlml.logging import LOG_PATH_FORMAT
from mlml.logging import LOG_HEADER
from mlml.logging import TIME
from mlml.loss import Loss
from mlml.loss import RidgeRegression
from mlml.model import RegressionModel
from mlml.model import Model


class Algorithm:
    """General structure and API for an algorithm."""

    @classmethod
    def from_arguments(
            cls,
            arguments: dict,
            X_test: np.ndarray,
            y_test: np.ndarray,
            logger: Logger=StandardLogger(),
            loss: Loss=RidgeRegression(0.1)) -> Tuple[Data, Model]:
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
            y_test: np.ndarray,
            logger: Logger=StandardLogger(),
            loss: Loss=RidgeRegression(0.1)) -> Tuple[Data, Model]:
        return ClosedForm().train(
            data_hook=arguments['--data-hook'],
            dtype=arguments['--dtype'],
            loss=loss,
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
            train_path: str) -> Tuple[Data, Model]:
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
        return data, loss.closed_form(data.X, data.Y)


class GD(Algorithm):
    """Gradient Descent Algorithm"""

    @classmethod
    def from_arguments(
            cls,
            arguments: dict,
            X_test: np.ndarray,
            y_test: np.ndarray,
            logger: Logger=StandardLogger(),
            loss: Loss=RidgeRegression(0.1)) -> Tuple[Data, Model]:
        return GD().train(
            damp=arguments['--damp'],
            data_hook=arguments['--data-hook'],
            dtype=arguments['--dtype'],
            eta0=arguments['--eta0'],
            iterations=arguments['--iters'],
            loss=loss,
            logger=logger,
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
            labels_test: np.ndarray) -> Tuple[Data, Model]:
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
        train = read_dataset(
            data_hook, dtype, num_classes, one_hot, train_path, shape)
        loss.pre_hook(train.X, train.Y)
        model = RegressionModel.initialize_zero(num_features, num_classes)
        for i in range(iterations):
            grad = loss.gradient(model)
            alpha = eta0 * damp ** (i // step)
            model.w -= alpha * grad
            logger.iteration(
                i, f, log_frequency, loss, model, train.X, train.Y,
                train.labels, X_test, labels_test)
        f.close()
        return train, model


class SGD(Algorithm):
    """Stochastic Gradient Descent Algorithm"""

    @classmethod
    def from_arguments(
            cls,
            arguments: dict,
            X_test: np.ndarray,
            y_test: np.ndarray,
            logger: Logger=StandardLogger(),
            loss: Loss=RidgeRegression(0.1)) -> Tuple[Data, Model]:
        return SGD().train(
            damp=arguments['--damp'],
            data_hook=arguments['--data-hook'],
            dtype=arguments['--dtype'],
            eta0=arguments['--eta0'],
            epochs=arguments['--epochs'],
            logger=logger,
            log_frequency=arguments['--logfreq'],
            loss=loss,
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
            labels_test: np.ndarray) -> Tuple[Data, Model]:
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
        train = read_dataset(
            data_hook, dtype, num_classes, one_hot, train_path, shape)
        model = RegressionModel.initialize_zero(num_features, num_classes)
        for p in range(epochs):
            indices = np.arange(0, train.X.shape[0])
            emulate_external_shuffle(num_per_block, indices)
            for i, index in enumerate(indices):
                grad = loss.gradient(model, train.X[index], train.Y[index])
                alpha = eta0 * damp ** (iteration // step)
                w_delta = alpha * grad + momentum * w_delta
                model.w -= w_delta
                logger.iteration(
                    iteration, f, log_frequency, loss, model, train.X,
                    train.Y, train.labels, X_test, labels_test)
                iteration += 1
            logger.epoch(p)
        f.close()
        return train, model
