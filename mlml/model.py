"""Common models."""

import numpy as np
import sklearn.metrics

from mlml.utils.data import de_one_hot


class Model:
    """General interface for a model to train."""

    @classmethod
    def initialize_zero(cls, num_features, num_classes):
        raise NotImplementedError

    def predict(self, X: np.ndarray):
        raise NotImplementedError

    def accuracy(self, X: np.ndarray, labels: np.ndarray):
        return sklearn.metrics.accuracy_score(self.predict(X), labels)


class RegressionModel(Model):
    """Represents the model for a regression problem."""

    def __init__(self, w: np.ndarray):
        self.w = w

    @classmethod
    def initialize_zero(cls, num_features, num_classes):
        return RegressionModel(np.zeros((num_features, num_classes)))

    def predict(self, X: np.ndarray):
        """Run prediction using current model.

        If a multi-class label, we take the most likely of all classes. If a
        binary classifier, then we determine if our value is above the
        threshold.
        """
        y_hat = X.dot(self.w)
        if y_hat.shape[1] > 1:
            return de_one_hot(y_hat)
        return np.where(y_hat > 0.5, [1], [0])
