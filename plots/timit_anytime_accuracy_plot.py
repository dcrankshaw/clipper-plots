import numpy as np
import matplotlib.pyplot as plt
import os


fig_dir = os.getcwd()
# fig_dir = "/Users/crankshaw/Dropbox/Apps/ShareLaTeX/velox-centipede/osdi_2016/figs"

def plot():
    data = np.loadtxt("../results/timit_anytime_results.csv", delimiter=",")
    accs = np.mean(data, axis=0)
    errs = np.std(data, axis=0) / np.sqrt(len(data[:,0]))
    fig, ax = plt.subplots()
    ax.errorbar(range(1,len(accs) + 1), accs, yerr=errs)
    ax.set_xlim((0, 9))
    fig.savefig(fig_dir + '/' + 'timit_anytime_acc' +'.pdf',bbox_inches='tight')


if __name__=='__main__':
    plot()
