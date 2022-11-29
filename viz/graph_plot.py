import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

def plot_hist(convergence_steps : np.ndarray, saturation_fractions : np.ndrarray):
    hist = True
    if hist:
        from scaling_hypotheses.hypotheses import Logistic
        from scipy.optimize import curve_fit
        import inspect
        logi = Logistic()
        ff = logi.fit_func()
        argStrings = (inspect.getfullargspec(ff).args)
        argStrings.pop(0)
        numArgs = len(argStrings)# The first argument is the x-data
        found = 0
        foundParams = np.array([])
        for i in range(0, len(convergence_steps)):
            times = np.array([x for x in range(0, convergence_steps[i])])
            sats = saturation_fractions[i]
            assert(len(times) == len(sats))
            try:
                p, pcov = curve_fit(ff, times, sats)
                found += 1
                foundParams = np.concatenate((foundParams, p), axis=0)
            except RuntimeError as e:
                pass
        split = np.split(foundParams, numArgs)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(numArgs)
        bins = 100
        for i in range(numArgs):
            ax.flat[i].hist(np.log10(split[i]), bins = bins)
            ax.flat[i].set(xlabel=argStrings[i])
        plt.show()


def plot_saturation(
    saturation_fraction_mean: np.ndarray,
    saturation_fraction_std: np.ndarray,
    timesteps: np.array = None,
    graph_type: str = "fully_connected",
    save_filename: bool = False
):

    timesteps = np.array([x for x in range(
        len(saturation_fraction_mean))] if timesteps is None else timesteps)

    def logistic(x, k, x0, A):
        return A / (1.0 + np.exp(-k * (x - x0)))
    grad = np.gradient(saturation_fraction_mean)
    
    def main_curve(axis):
        from scipy.optimize import curve_fit
        fig = plt.figure(figsize=(12, 8))
        axis.plot(timesteps, saturation_fraction_mean, label=f"{graph_type}, saturation")
        ax2 = axis.twinx()
        ax2.plot(timesteps, grad, label="d/dt saturation")
        ax2.set(ylabel="Gradient")
        ax2.legend()

        try:
            p, cov = curve_fit(logistic, timesteps, saturation_fraction_mean)
            axis.plot(timesteps, logistic(timesteps, *p), label="logistic")
        except RuntimeError as e:
            print(f"{e}")

        axis.fill_between(timesteps, saturation_fraction_mean+saturation_fraction_std,
                        saturation_fraction_mean-saturation_fraction_std, alpha=0.3, label="1.Std across all runs")
        axis.set(xlabel=f'Number of timesteps')
        axis.set(ylabel=f'Fraction saturated')
        axis.legend()

    def gradient_curves(axis):
        axis.set(xlabel='Saturation', ylabel='Gradient')
        axis.plot(saturation_fraction_mean, grad)
    fig, axes = plt.subplots(1, 2)

    fig.set_size_inches(8, 4.5)
    main_curve(axes.flat[0])
    gradient_curves(axes.flat[1])
    fig.tight_layout()
    #ßfig.subplots_adjust(top=0.1, left=0.1)
    from datetime import datetime
    # Move this to somewhere else
    fig.suptitle(f"{graph_type}, plotted on {str(datetime.now())}")
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
