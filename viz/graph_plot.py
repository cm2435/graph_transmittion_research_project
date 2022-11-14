import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path


def plot_saturation(
    saturation_fraction_mean: np.ndarray,
    saturation_fraction_std: np.ndarray,
    timesteps: np.array = None,
    graph_type: str = "fully_connected",
    save_filename: bool = False
):

    timesteps = [x for x in range(
        len(saturation_fraction_mean))] if timesteps is None else timesteps

    def logistic(x, k, x0):
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))


    from scipy.optimize import curve_fit
    p, cov = curve_fit(logistic, timesteps, saturation_fraction_mean)
    fig = plt.figure(figsize=(12, 8))
    plt.plot(timesteps, saturation_fraction_mean, label=f"{graph_type}, saturation")
    plt.plot(timesteps, logistic(timesteps, *p), label="logistic")
    plt.fill_between(timesteps, saturation_fraction_mean+saturation_fraction_std,
                     saturation_fraction_mean-saturation_fraction_std, alpha=0.3, label="1.Std across all runs")
    plt.xlabel(f'Number of timesteps')
    plt.ylabel(f'Fraction saturated')
    plt.legend()
    from datetime import datetime
    # Move this to somewhere else
    plt.title(f"{graph_type}, plotted on {str(datetime.now())}")
    # plt.show()
    if save_filename:
        fig.savefig(
            str(Path(__file__).parents[1] / "data" / "figures") + f'/{graph_type}.png')


if __name__ == "__main__":
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    means = np.array([x for x in range(100)])
    stds = np.array([random.randint(0, 20) for x in range(100)])
    timesteps = [x for x in range(100)]

    plot_saturation(timesteps, means, stds)
