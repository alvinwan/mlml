"""This files contains several common loss functions.

In the following, we use the following conventions:

    d : the number of features for each sample
    k : the number of classes
    n : the number of samples
"""

import numpy as np
import scipy

from mlml.model import Model
from mlml.model import RegressionModel


class Loss:
    """General interface for a loss."""

    def closed_form(self, X: np.ndarray, Y: np.ndarray) -> Model:
        """Run closed form solution and return trained model."""
        raise NotImplementedError

    def gradient(
            self,
            model: Model,
            X: np.ndarray=None,
            Y: np.ndarray=None) -> np.ndarray:
        """Compute gradient of loss with respect to parameters."""
        raise NotImplementedError

    def pre_hook(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Run before the main algorithm loops are run."""
        raise NotImplementedError

    def __call__(self, model: Model, X: np.ndarray, Y: np.ndarray) -> float:
        """Evaluate the loss of the provided model."""
        raise NotImplementedError


class RidgeRegression(Loss):
    """Least squares loss with L2 regularization.

    (1/2) X^T ||Xw - y||_2^2 + (1/2)reg||w||_2^2
    """

    def __init__(self, reg: float):
        self.reg = reg
        self.XTX = None
        self.XTy = None

    def closed_form(self, X: np.ndarray, Y: np.ndarray) -> RegressionModel:
        """Closed form solution for Ridge Regression

        Args:
            X: nxd, row-major design matrix
            Y: nxk, output

        Returns:
            Optional model
        """
        num_features = X.shape[1]
        XTX, I, XTY = X.T.dot(X), np.identity(num_features), X.T.dot(Y)
        return RegressionModel(
            scipy.linalg.solve(XTX + self.reg * I, XTY, sym_pos=True))

    def gradient(
            self,
            model: RegressionModel,
            X: np.ndarray=None,
            Y: np.ndarray=None) -> np.ndarray:
        """Compute the gradient for Ridge Regression. Taking the gradient of
        the objective function specified above, with respect to w, we have the
        following.

        dL/dw = X^T (Xw - y) + reg w

        Args:
            model: The model for that we are training
            X: nxd, row-major design matrix
            Y: nxk, output

        Returns:
            gradient of the loss with respect to the model parameters
        """
        if X is None or Y is None:
            return self.XTX.dot(model.w) - self.XTy + self.reg * model.w
        X, Y = np.matrix(X), np.matrix(Y)
        return X.T.dot(X.dot(model.w) - Y) + self.reg * model.w

    def pre_hook(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Pre-compute dot products to speed up gradient calculation."""
        self.XTX, self.XTy = X.T.dot(X), X.T.dot(Y)

    def __call__(
            self,
            model: RegressionModel,
            X: np.ndarray,
            Y: np.ndarray) -> float:
        """Loss function for Ridge Regression.

        Args:
            model: contains w, dxk vector
            X: nxd, row-major design matrix
            Y: nxk, output

        Returns:
            A scalar value representing the loss
        """
        return np.asscalar(np.linalg.norm(X.dot(model.w) - Y) +
                           self.reg * np.linalg.norm(model.w))