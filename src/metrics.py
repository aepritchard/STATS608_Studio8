"""Losses and one helper for Monte Carlo uncertainty in a paired ratio."""

import numpy as np


def evaluate(beta_hat, data):
    """Realized squared losses, without averaging or dividing by p.

    Prediction loss concerns the conditional mean at a new predictor with
    the same covariance; it does not include new response noise.
    """
    difference = beta_hat - data["beta_true"]
    return {"coefficient_l1": np.sum(np.abs(difference)),
            "coefficient_error": float(difference @ difference)}

