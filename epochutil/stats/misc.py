import matplotlib.pyplot as plt

def plot_ecdf(x, **plot_args):
    x = filter_nans(x)
    x = np.sort(x)
    y = np.arange(len(x))/float(len(x))
    plt.plot(x, y, **plot_args)

