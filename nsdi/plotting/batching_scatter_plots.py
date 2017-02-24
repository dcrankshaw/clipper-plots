from __future__ import print_function
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import seaborn as sns
import statsmodels.formula.api as smf
import utils

"""
NOTES:
    For this experiment, we ran the adaptive batching search
    algorithm from its initialization with batch size 1 for
    1000 samples for each model with a latency objective of
    20ms. The scatter plots show the batch size and measured
    latency of each sample. We then fit a 99-percentile quantile
    regression model and plot this as well. The AIMD Decrease points
    are the samples where the adaptive algorithm decreased the
    maximum batch size, and we can see that these are exactly
    the points above the latency objective.

"""


fig_dir = utils.NSDI_FIG_DIR


# sns.set_style("darkgrid")
sns.set_style("darkgrid")
sns.set_context("paper", font_scale=0.75,)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def find_backtrack_points(batches, lats, k=False):
    bt_batches = []
    bt_lats = []
    idx = []
    for i in range(len(batches) - 1):
        if batches[i] > batches[i + 1]:
            if k:
                bt_batches.append(batches[i])
                bt_lats.append(lats[i])
                idx.append(i)
            else:
                if batches[i] > 50:
                    bt_batches.append(batches[i])
                    bt_lats.append(lats[i])
                    idx.append(i)
    return (bt_batches, bt_lats)

def process_batch(results, name, ax, title=None):
    splits = [(r["batch_size"], r["latency"]) for r in results[name]["measurements"]]
    batches, latencies = zip(*splits)
    df = pd.DataFrame({"batch_size": batches, "latencies": latencies})
    
    # Fit the quantile regression line
    quantiles = [0.99]
    mod = smf.quantreg('latencies ~ batch_size', df)
    def fit_model(q):
        model_fit = mod.fit(q=q)
        return [q, model_fit.params['Intercept'], model_fit.params['batch_size']] + model_fit.conf_int().ix['batch_size'].tolist()    
    models = [fit_model(cur_q) for cur_q in quantiles]
    models = pd.DataFrame(models, columns=['q', 'a', 'b','lb','ub']) 
    x_max_lim = np.max(batches)*1.2
    if x_max_lim > 10:
        x_max_lim = 1600
    x = np.arange(0, x_max_lim, 1)
    # colors = sns.cubehelix_palette(4, start=2.8, rot=-0.1)
    colors=sns.color_palette("cubehelix", 5)
    bt_batches, bt_lats = find_backtrack_points(batches, latencies, name=="kernel_svm")
    lw=1
    ax.scatter(batches, latencies, color=colors[0], s=4, label="Samples")
    ax.scatter(bt_batches, bt_lats, color=colors[3], s=20, label="AIMD Decrease")
    get_y = lambda a, b: a + b * x
    for i in range(models.shape[0]):
        y = get_y(models.a[i], models.b[i])
        max_idx = len(y)
        if x_max_lim > 10:
            for iii in range(len(y)):
                if y[iii] >= 29000:
                    max_idx = iii
                    break
        ax.plot(x[:max_idx], y[:max_idx], linestyle='-', color=colors[1], linewidth=lw, label="P%d Regression Line" % int(models.q[i]*100))
    ax.plot([0, x_max_lim], np.ones(2)*20000, color=colors[2], linestyle="dashed", linewidth=lw, label = "SLO")
    ax.set_xlim((0, x_max_lim))
    ax.set_ylim((0, max(np.max(latencies)*1.3, 35000)))   


if __name__=="__main__":
    results = {}
    raw_results_fname = os.path.abspath("../results/batching_scatter_plots_raw_logs.txt")




    with open(raw_results_fname, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if "GREPTHIS" in line:
                splits = line.strip().split("XXXXXX")
                res = json.loads(splits[1].strip())
                results[res["name"]] = res

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 2))
    axs = axs.flatten()
    for (i,n) in enumerate(results.keys()):
        ax = axs[i]
        # print(ax)
        process_batch(results, n, ax)
        title = "%s) %s" % (chr(i + ord('a')), utils.name_map[n])

        ax.text(0.5, 0.9, title, fontdict={"weight": "bold",},
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                # backgroundcolor="white"
                )

    legend = axs[1].legend(frameon=True, bbox_to_anchor=(-1.21, 1.05, 3.42, .102), loc=3,
            ncol=4, mode="expand", borderaxespad=0.1, fontsize=8,)


    # if title is None:
    axs[0].set_ylabel("Latency ($\mu$s)")
    axs[3].set_ylabel("Latency ($\mu$s)")
    axs[3].set_xlabel("Batch Size")
    axs[4].set_xlabel("Batch Size")
    axs[5].set_xlabel("Batch Size")
    fig.subplots_adjust(hspace=0.3)

    fname = "%s/batching_scatterplots.pdf" % (fig_dir)
    plt.savefig(fname, bbox_inches='tight')
    print(fname)
