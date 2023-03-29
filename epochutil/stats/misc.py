import numpy as np
import matplotlib.pyplot as plt


def filter_nans(x):
    x = np.array(x)
    return x[~np.isnan(x)]

def plot_ecdf(x, **plot_args):
    x = filter_nans(x)
    x = np.sort(x)
    y = np.arange(len(x))/float(len(x))
    plt.plot(x, y, **plot_args)

