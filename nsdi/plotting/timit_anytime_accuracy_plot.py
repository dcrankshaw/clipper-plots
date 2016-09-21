import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set_style("white")
sns.set_context("paper", font_scale=1.0,)
# fig_dir = os.getcwd()
# fig_dir = "/Users/crankshaw/ModelServingPaper/osdi_2016/figs"
fig_dir = "/Users/giuliozhou/Research/RISE/ModelServingPaper/nsdi_2017/figs"

digits_accuracies = [0.5000, 0.4079, 0.3293, 0.2623, 0.2158, 0.1807, 0.1477, 0.1132, 0.0834, 0.0555, 0.0286]
# matplotlib.rcParams['font.family'] = "Times New Roman"
# matplotlib.rcParams['font.size'] = 9
width=1.5
height=0.7
nbins = 6
def plot_digits():
    digits_errors = [0.5000, 0.4079, 0.3293, 0.2623, 0.2158, 0.1807, 0.1477, 0.1132, 0.0834, 0.0555, 0.0286]
    # accs = 1 - np.mean(data, axis=0)
    digits_errors.reverse()
    accs = 1 - np.array(digits_errors)
    # errs = np.std(data, axis=0) / np.sqrt(len(data[:,0]))
    fig, ax = plt.subplots()
    ax.locator_params(tight=True, nbins=nbins)
    # ax.errorbar(range(1,len(accs) + 1), accs, yerr=errs)
    ax.plot(range(0,len(accs)), accs)
    # ax.set_xlim((0, 9.2))
    ax.set_ylabel("Accuracy")
    # ax.set_ylim((0, 0.55))
    ax.set_ylim((0.45, 1.0))
    # ax.set_xlabel("Available Predictions")
    ax.set_xlabel("Stragglers")
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    fig.set_size_inches((width, height))
    # plt.tight_layout()

    fig.savefig(fig_dir + '/' + 'digits_anytime_acc' +'.pdf',bbox_inches='tight')

def plot_timit():
    colors = sns.cubehelix_palette(3, start=.75, rot=-.75)
    data = np.loadtxt("../results/timit_anytime_results.csv", delimiter=",")
    accs = np.mean(data, axis=0)
    accs = accs[::-1]
    # errs = np.std(data, axis=0) / np.sqrt(len(data[:,0]))
    # fig, ax = plt.subplots(figsize=(3,1.5))
    fig = plt.figure(figsize=(4,1.5))
    ax = plt.gca()
    ax.locator_params(tight=True,nbins=nbins)
    # ax.errorbar(range(1,len(accs) + 1), accs, yerr=errs)
    # ax.plot(range(1,len(accs) + 1), accs)
    ax.plot(range(len(accs)), accs, color=colors[1])
    ax.set_xlim((0, 8.2))
    yd,yu = ax.get_ylim()
    # ax.set_ylim(
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Stragglers")
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # fig.set_size_inches((width, height))
    # plt.tight_layout()

    fig.savefig(fig_dir + '/' + 'timit_anytime_acc' +'.pdf',bbox_inches='tight')


if __name__=='__main__':
    plot_timit()
    # plot_digits()
