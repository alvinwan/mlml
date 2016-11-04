from typing import Callable
from typing import Tuple

from mlml.utils.data import read_full_dataset
from mlml.utils.data import block_to_x_y
from mlml.ssgd.blocks import BlockBuffer
from mlml.ssgd.shuffle import emulate_external_shuffle
from mlml.ssgd.shuffle import shuffle_train

import numpy as np
import scipy
import sklearn
import sklearn.metrics
import time


TIME = time.time()
LOG_PATH_FORMAT = 'logs/{algo}/run-{time}.csv'
LOG_HEADER = 'i,Time,Loss,Train Accuracy,Test Accuracy\n'
LOG_ENTRY_FORMAT = '{i},{time},{loss},{train_accuracy},{test_accuracy}\n'


# Temporary functions

def ridgeloss(
        X: np.ndarray,
        w: np.ndarray,
        Y: np.ndarray,
        reg: float):
    """Compute ridge regression loss."""
    A = X.dot(w) - Y
    return np.asscalar(np.linalg.norm(A) + reg * np.linalg.norm(w))


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
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            one_hot=arguments['--one-hot'],
            reg=arguments['--reg'],
            train_path=arguments['--train'])

    def train(
            self,
            data_hook: Callable[[np.ndarray, np.ndarray], Tuple],
            dtype: str,
            n: int,
            num_classes: int,
            num_features: int,
            one_hot: bool,
            reg: float,
            train_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the closed form solution.

        Args:
            data_hook: Processes data for given scenario
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
        X, y = read_full_dataset(
            data_hook, dtype, train_path, shape, num_classes, one_hot)
        XTX, I, XTy = X.T.dot(X), np.identity(num_features), X.T.dot(y)
        return X, y, scipy.linalg.solve(XTX + reg * I, XTy, sym_pos=True)


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
            log_frequency=arguments['--logfreq'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            one_hot=arguments['--one-hot'],
            reg=arguments['--reg'],
            step=arguments['--step'],
            train_path=arguments['--train'],
            X_test=X_test,
            y_test=y_test)

    def train(
            self,
            damp: float,
            data_hook: Callable[[np.ndarray, np.ndarray], Tuple],
            dtype: str,
            eta0: float,
            iterations: int,
            log_frequency: int,
            n: int,
            num_features: int,
            num_classes: int,
            one_hot: bool,
            reg: float,
            step: int,
            train_path: str,
            X_test: np.ndarray,
            y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Train using gradient descent.

        Args:
            damp: Amount to multiply learning rate at each epoch
            data_hook: Processes data for given scenario
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
            step: Number of iterations between each alpha decay
            train_path: Path to the training file (binary)
            X_test: Test input data
            y_test: Test output data

        Returns:
            The trained model
        """
        with open(LOG_PATH_FORMAT.format(time=TIME, algo='gd'), 'w') as f:
            f.write(LOG_HEADER)
            shape = (n, num_features)
            X, Y = read_full_dataset(
                data_hook, dtype, train_path, shape, num_classes, one_hot)
            XTX, XTy = X.T.dot(X), X.T.dot(Y)
            w = np.zeros((num_features, num_classes))
            for i in range(iterations):
                grad = XTX.dot(w) - XTy + reg * w
                alpha = eta0 * damp ** (i // step)
                w -= alpha * grad

                if log_frequency and i % log_frequency == 0:
                    train_accuracy, test_accuracy = evaluate_model(
                        w, one_hot, X_test, X, y_test, Y)
                    f.write(LOG_ENTRY_FORMAT.format(
                        i=i,
                        time=time.time() - TIME,
                        loss=ridgeloss(X, w, Y, reg),
                        train_accuracy=train_accuracy,
                        test_accuracy=test_accuracy))
        return X, Y, w


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
            log_frequency=arguments['--logfreq'],
            momentum=arguments['--momentum'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            num_per_block=arguments['--num-per-block'],
            one_hot=arguments['--one-hot'],
            reg=arguments['--reg'],
            step=arguments['--step'],
            train_path=arguments['--train'],
            X_test=X_test,
            y_test=y_test)

    def train(
            self,
            damp: float,
            data_hook: Callable[[np.ndarray, np.ndarray], Tuple],
            dtype: str,
            eta0: float,
            epochs: int,
            log_frequency: int,
            momentum: float,
            n: int,
            num_classes: int,
            num_features: int,
            num_per_block: int,
            one_hot: bool,
            reg: float,
            step: int,
            train_path: str,
            X_test: np.ndarray,
            y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Train using stochastic gradient descent.

        Args:
            damp: Amount to multiply learning rate at each epoch
            data_hook: Processes data for given scenario
            dtype: Data type of numbers in file
            eta0: Starting learning rate
            epochs: Number of passes over training data
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
        with open(LOG_PATH_FORMAT.format(time=TIME, algo='sgd'), 'w') as f:
            f.write(LOG_HEADER)
            shape, w_delta, index = (n, num_features), 0, 0
            w = np.zeros((num_features, num_classes))
            X, Y = read_full_dataset(
                data_hook, dtype, train_path, shape, num_classes, one_hot)
            for p in range(epochs):
                indices = np.arange(0, X.shape[0])
                emulate_external_shuffle(num_per_block, indices)
                for i, random_index in enumerate(indices):
                    index += 1
                    x, y = np.matrix(X[random_index]), np.matrix(
                        Y[random_index])
                    grad = x.T.dot(x.dot(w) - y) + reg * w
                    alpha = eta0 * damp ** ((n * p + i) // step)
                    w_delta = alpha * grad + momentum * w_delta
                    w -= w_delta

                    if log_frequency and index % log_frequency == 0:
                        train_accuracy, test_accuracy = evaluate_model(
                            w, one_hot, X_test, X, y_test, Y)
                        f.write(LOG_ENTRY_FORMAT.format(
                            i=index,
                            time=time.time() - TIME,
                            loss=ridgeloss(X, w, Y, reg),
                            train_accuracy=train_accuracy,
                            test_accuracy=test_accuracy))
                print('=' * 30, '\n * SGD : Epoch {p} finished.'.format(p=p))
        return X, Y, w


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
            log_frequency=arguments['--logfreq'],
            momentum=arguments['--momentum'],
            n=arguments['--n'],
            num_classes=arguments['--k'],
            num_features=arguments['--d'],
            num_per_block=arguments['--num-per-block'],
            one_hot=arguments['--one-hot'],
            reg=arguments['--reg'],
            simulated=arguments['--simulated'],
            step=arguments['--step'],
            train_path=arguments['--train'],
            X_test=X_test,
            y_test=y_test)

    def train(
            self,
            algorithm: str,
            damp: float,
            data_hook: Callable[[np.ndarray, np.ndarray], Tuple],
            dtype: str,
            epochs: int,
            eta0: float,
            log_frequency: int,
            momentum: float,
            n: int,
            num_classes: int,
            num_features: int,
            num_per_block: int,
            one_hot: bool,
            reg: float,
            simulated: bool,
            step: int,
            train_path: str,
            X_test: np.ndarray,
            y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        if simulated:
            shape = (n, num_features)
            X_train, Y_train = read_full_dataset(
                data_hook, dtype, train_path, shape, num_classes, one_hot)
        with open(LOG_PATH_FORMAT.format(time=TIME, algo='ssgd'), 'w') as f:
            f.write(LOG_HEADER)
            w, I = np.zeros((num_features, num_classes)), np.identity(
                num_features)
            w_delta = index = 0
            for p in range(epochs):
                shuffled_path = shuffle_train(
                    algorithm, dtype, n, num_per_block, num_features,
                    train_path)
                blocks = BlockBuffer(dtype, n, num_features + 1, num_per_block,
                                     shuffled_path)
                deblockify = lambda block: block_to_x_y(block, num_classes,
                                                        one_hot)
                for X, Y in map(deblockify, blocks):
                    X, Y = data_hook(X, Y)
                    if Y.shape[0] < Y.shape[1]:
                        Y.shape = (X.shape[0], 1)  # hacky
                    for i in range(X.shape[0]):
                        index += 1
                        x, y = np.matrix(X[i]), np.matrix(Y[i])
                        grad = x.T.dot(x.dot(w) - y) + reg * w
                        alpha = eta0 * damp ** ((n * p + i) // step)
                        w_delta = alpha * grad + momentum * w_delta
                        w -= w_delta

                        if log_frequency and index % log_frequency == 0:
                            if not simulated:
                                X_train, Y_train = X, Y
                            train_accuracy, test_accuracy = evaluate_model(
                                w, one_hot, X_test, X_train, y_test, Y_train)
                            f.write(LOG_ENTRY_FORMAT.format(
                                i=index,
                                time=time.time() - TIME,
                                loss=ridgeloss(X_train, w, Y_train, reg),
                                train_accuracy=train_accuracy,
                                test_accuracy=test_accuracy))
                print('=' * 30, '\n * SGD : Epoch {p} finished.'.format(p=p))
        return X_train, Y_train, w