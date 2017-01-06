"""Common kernel functions."""

import numpy as np


class RBF:

    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, Xi: np.ndarray, Xj: np.ndarray) -> np.ndarray:
        Xi = Xi.reshape((Xi.shape[0], -1))
        Xj = Xj.reshape((Xj.shape[0], -1))
        Xi_norms = (np.linalg.norm(Xi, axis=1) ** 2)[:, np.newaxis]
        Xj_norms = (np.linalg.norm(Xj, axis=1) ** 2)[:, np.newaxis]
        return Xi_norms - 2 * Xi.dot(Xj.T) + Xj_norms.T