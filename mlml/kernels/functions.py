"""Common kernel functions."""

import numpy as np


class RBF:

    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, Xi: np.ndarray, Xj: np.ndarray) -> np.ndarray:
        return np.exp(-(np.linalg.norm(Xj - Xi) / (2 * self.sigma ** 2)))