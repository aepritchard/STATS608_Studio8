import pickle
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# Open the file in binary read mode ('rb')
with open('results/your_file.pkl', 'rb') as file:
    data = pickle.load(file)

def theoretical_risk(gamma, r2=5):
    gamma = np.asarray(gamma, dtype=float)
    if np.any(gamma == 1):
        raise ValueError("gamma must not equal 1")
    risk_below_one = gamma / (1 - gamma)
    risk_above_one = r2 * (1 - 1 / gamma) + 1 / (gamma - 1)
    return np.where(gamma < 1, risk_below_one, risk_above_one)

def make_figure_subplot(n_sim, ax):
    # Get data for n_sim
    data_for_n_sim = data[data["n_sim"] == n_sim]
    gamma = data_for_n_sim["gamma"].to_numpy()
    mse_hat = data_for_n_sim["mse_hat"].to_numpy()
    se_mse = data_for_n_sim["se_mse"].to_numpy()

    # Change transparency and markersize for n_sim = 1
    alpha = 1
    markersize = 12
    if n_sim == 1:
        alpha = 0.5
        markersize = 10

        ax.plot(gamma, mse_hat, linestyle="none", marker="o",
                alpha = alpha, markersize = markersize)
        # Run non parametric regression (e.g. LOWESS w/ frac =0.1 and it=0)
        # For gamma in [0.1, 1]
        data_below_one = (gamma >= 0.1) & (gamma < 1)
        fit_below_one = lowess(mse_hat[data_below_one], gamma[data_below_one], frac=0.1, it=0)
        ax.plot(fit_below_one[:, 0], fit_below_one[:, 1], color="blue", label="LOWESS")

        # For gamma in [1,10]
        data_above_one = (gamma > 1) & (gamma <= 10)
        fit_above_one = lowess(mse_hat[data_above_one], gamma[data_above_one], frac=0.1, it=0)
        ax.plot(fit_above_one[:, 0], fit_above_one[:, 1], color="red", label="LOWESS")

    else:
        # Plot the points in the scatterplot w/ error bars 2x standard error
        ax.errorbar(gamma, mse_hat, yerr=2 * se_mse, linestyle="none", marker="o",
                    alpha=alpha, markersize=markersize, capsize=2)

    # Plot vertical asymptote at one
    ax.axvline(1, color="gray", linestyle="--")
    
    # Plot theoretical risk curve
    gamma_grid_below_one = np.linspace(0.1, 0.999, 500)
    gamma_grid_above_one = np.linspace(1.001, 10, 500)
    ax.plot(gamma_grid_below_one, theoretical_risk(gamma_grid_below_one),
            color="black", label="Theoretical risk")
    ax.plot(gamma_grid_above_one, theoretical_risk(gamma_grid_above_one),
            color="black")

    # Change to log scale
    ax.set_xscale("log")
    ax.set_xlim(0.1, 10)
    ax.set_title(f"n_sim = {n_sim}")
    ax.set_xlabel("gamma")
    ax.set_ylabel("Estimated risk")
    ax.legend()

# Make figure
plt.savefig("figure.pdf", bbox_inches='tight')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, n_sim in zip(axes, [1, 50, 1000]):
    make_figure_subplot(n_sim, ax)
fig.tight_layout()
