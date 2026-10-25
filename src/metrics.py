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

def summarize_risk(results):
    """
    Return one row per (n_sim, gamma) scenario with the MSE and MCSE
    """
    summary = results.groupby(["nsim", "gamma"], as_index=False).agg(
        mse_hat=("coefficient_error", "mean"),
        se_mse=("coefficient_error", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
    )
    # For n_sim = 1, set SE to 0
    summary["se_mse"] = summary["se_mse"].fillna(0.0)
    summary = summary.rename(columns={"nsim": "n_sim"})
    return summary