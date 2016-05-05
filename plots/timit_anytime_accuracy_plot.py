import numpy as np
import matplotlib.pyplot as plt
import os


# fig_dir = os.getcwd()
fig_dir = "/Users/crankshaw/Dropbox/Apps/ShareLaTeX/velox-centipede/osdi_2016/figs"

digits_accuracies = [0.5000, 0.4079, 0.3293, 0.2623, 0.2158, 0.1807, 0.1477, 0.1132, 0.0834, 0.0555, 0.0286]

def plot():
    data = np.loadtxt("../results/timit_anytime_results.csv", delimiter=",")
    accs = 1 - np.mean(data, axis=0)
    # errs = np.std(data, axis=0) / np.sqrt(len(data[:,0]))
    fig, ax = plt.subplots()
    # ax.errorbar(range(1,len(accs) + 1), accs, yerr=errs)
    ax.plot(range(1,len(accs) + 1), accs)
    ax.set_xlim((0, 9.2))
    ax.set_ylabel("Test Error")
    ax.set_xlabel("Number of features included")
    fig.set_size_inches((4, 2))

    fig.savefig(fig_dir + '/' + 'timit_anytime_acc' +'.pdf',bbox_inches='tight')


if __name__=='__main__':
    plot()
