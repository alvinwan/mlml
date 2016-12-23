import datetime
import functools
import numpy as np
import time

from mlml.model import Model
from mlml.loss import Loss

TIME = time.time()
LOG_PATH_FORMAT = 'logs/{algo}/run-{time}.csv'
LOG_HEADER = 'i,Time,Loss,Train Accuracy,Test Accuracy\n'
LOG_ENTRY_FORMAT = '{i},{time},{loss},{train_accuracy},{test_accuracy}\n'


def timeit(f):
    """Times the function that it decorates."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        a = datetime.datetime.now()
        rv = f(*args, **kwargs)
        b = datetime.datetime.now()
        c = b - a
        print(' * Time (s):', c.total_seconds())
        return rv
    return wrapper


class Logger:

    def iteration(
            self,
            iteration: int,
            f,
            log_frequency: int,
            loss: Loss,
            model: Model,
            X_train: np.ndarray,
            y_train: np.ndarray,
            labels_train: np.ndarray,
            X_test: np.ndarray,
            labels_test: np.ndarray):
        raise NotImplementedError

    def epoch(self, epoch: int):
        raise NotImplementedError


class StandardLogger(Logger):

    def iteration(
            self,
            iteration: int,
            f,
            log_frequency: int,
            loss_function: Loss,
            model: Model,
            X_train: np.ndarray,
            y_train: np.ndarray,
            labels_train: np.ndarray,
            X_test: np.ndarray,
            labels_test: np.ndarray):
        if log_frequency and iteration % log_frequency == 0:
            train_accuracy = model.accuracy(X_train, labels_train)
            test_accuracy = model.accuracy(X_test, labels_test)
            print('Train:', train_accuracy, 'Test:', test_accuracy,
                  'Loss:', loss_function(model, X_train, y_train))
            f.write(LOG_ENTRY_FORMAT.format(
                i=iteration,
                time=time.time() - TIME,
                loss=loss_function(model, X_train, y_train),
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy))

    def epoch(self, epoch: int):
        print('=' * 30, '\n * SGD : Epoch {p} finished.'.format(p=epoch))