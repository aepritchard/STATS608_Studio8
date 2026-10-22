"""Specify the simulation design in one place."""
import numpy as np


def make_scenarios():
    n=200
    r2 = 5
    scenarios = {}
    distribution = "normal"
    nsims = [1,50,1000]
    gammas = [enumerate(np.logspace(.1,10,num = 5000)),enumerate(np.logspace(.1,10,num = 100)), [.2,.5, .8,2,5]]
    #make the nsim=1 ones
    for i in range(len(nsims)):
        for gamma in gammas[i]:
            scenarios[f"nsim={nsims[i]}_gamma={gamma}"] = {
                "nsim": nsims[i], "n":n, "r2" : r2, "gamma": gamma, "error_distribution": distribution
            }
    return scenarios
