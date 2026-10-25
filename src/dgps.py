"""Regression DGPs sharing the generate(rng) interface from Lecture 6."""

from abc import ABC, abstractmethod

import numpy as np


class RegressionDGP(ABC):
    """Gaussian predictors with varying number of features, fixed beta, 
    fixed noise distribution


    Parameters satisfy 0 < p < n and 0 <= rho < 1. generate() takes a
    NumPy Generator and returns X, y, beta_true, covariance in a dictionary.
    Array shapes are (n, p), (n,), (p,), (p, p), respectively.
    """

    def __init__(self, n, gamma, r2):
        self.n,  self.r2 = n, r2
        self.p = int(gamma*n)
        self.beta = np.sqrt(r2/self.p)*np.ones(self.p)
           
    @abstractmethod
    def generate(self, rng):
         """Return Design matrix and variables from the regression model."""

class NormalDGP(RegressionDGP):
    def generate(self, rng):
        X = rng.normal(size=(self.n, self.p)) 
        return {"X": X, "y": X @ self.beta + rng.normal(size=self.n),
                "beta_true": self.beta}

DGP_CLASSES = {"normal": NormalDGP}
