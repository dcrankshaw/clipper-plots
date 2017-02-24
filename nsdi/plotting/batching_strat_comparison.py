from __future__ import print_function
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import seaborn as sns
# import statsmodels.formula.api as smf
import utils

sns.set_style("darkgrid")
sns.set_context("paper", font_scale=1.0,)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# sns.set_palette("Paired")
# sns.set_style("whitegrid", {"axes.grid": "False"})
# sns.set_style("white")
# sns.set_context("paper", font_scale=1.5,)

fig_dir = utils.NSDI_FIG_DIR
results_dir = os.path.abspath("../results/batching_strat_comparison")

name_map = {
        "noop": "No-Op",
        "logistic_reg": "Log Regression\n(SKLearn)",
        "linear_svm": "Linear SVM\n(SKLearn)",
        "spark_svm": "Linear SVM\n(PySpark)",
        "kernel_svm": "Kernel SVM\n(SKLearn)",
        "rf_d16": "Random Forest\n(SKlearn)",
        }

def load_results():
    results_files = utils.get_results_files(results_dir)
    results = {}
    for name in results_files:
        strat = name.split("_")[1]
        with open(os.path.join(results_dir, name)) as f:
            if strat == "aimd":
                strat = "Adaptive"
            elif strat == "learned":
                strat = "Quantile Regression"
            elif strat == "static":
                strat = "No Batching"
            results[strat] = json.load(f)
    return results

def plot_thrus(ax, results, figsize, colors):
    # fig, ax = plt.subplots(figsize=figsize)
    sns.despine()
    width = 1.0
    space = 0.7
    num_bars = len(results)
    offset = 0
    for strat in ["Adaptive", "Quantile Regression", "No Batching"]:
        thrus = [(m["name"].split(":")[0], m["rate"]) for m in results[strat]["meters"] if "model_thruput" in m["name"]]
        model_names, rates = zip(*thrus)
        model_names = [name_map[m] for m in model_names]
        cur_rect = ax.bar(np.arange(len(rates))*width*(num_bars+ space) + width*offset, rates, color=colors[offset], width=width, label=strat)
        if strat == "Adaptive":
            utils.barchart_label(ax, cur_rect, 7, rot=40, ha="left")
        else:
            utils.barchart_label(ax, cur_rect, 7, rot=40, ha="left")
        if offset == 0:
            ax.set_xticks(np.arange(len(rates))*width*(num_bars + space) + width*(num_bars/2.0))
            ax.set_xticklabels(model_names, rotation=0, ha="center")
        offset += 1
    ax.set_ylim(0, 70000)
    ax.set_xlim(-0.3, ax.get_xlim()[1])
    ax.set_ylabel("Throughput\n(qps)")
    ax.locator_params(nbins=4, axis="y")

    ax.legend(frameon=True, bbox_to_anchor=(0.0, 1.07, 1.0, .097), loc=3,
                ncol=3, mode="expand", borderaxespad=0.05, fontsize=7,)

def plot_latencies(ax, results, figsize, colors):
    # fig, ax = plt.subplots(figsize=figsize)
    sns.despine()
    width = 1
    space = 0.7
    num_bars = len(results)
    offset = 0

    for strat in ["Adaptive", "Quantile Regression", "No Batching"]:
        p99 = [(m["name"].split(":")[0], m["p99"]) for m in results[strat]["histograms"] if "model_latency" in m["name"]]
        model_names, lats = zip(*p99)
        model_names = [name_map[m] for m in model_names]
        cur_rect = ax.bar(np.arange(len(lats))*width*(num_bars+ space) + width*offset, lats, color=colors[offset], width=width, label=strat)
        if strat == "Adaptive":
            utils.barchart_label(ax, cur_rect, 7, rot=40, ha="left")
        else:
            utils.barchart_label(ax, cur_rect, 7, rot=40, ha="left")
        if offset == 0:
            ax.set_xticks(np.arange(len(lats))*width*(num_bars + space) + width*(num_bars/2.0) - 1.3)
            ax.set_xticklabels(model_names, rotation=30, ha="center")
        offset += 1
    ax.set_ylim(0, 50000)
    ax.set_xlim(-0.3, ax.get_xlim()[1])
    ax.set_ylabel("P99 Latency\n($\mu$s)")
    ax.locator_params(nbins=4, axis="y")
    # ax.legend(frameon=True, bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3,
    #             ncol=3, mode="expand", borderaxespad=0.05, fontsize=7,)
    #
    # fname = "%s/batching_strategy_comp_lat.pdf" % (fig_dir)
    # plt.savefig(fname, bbox_inches='tight')
    # print(fname)

def plot_batch_sizes(ax, results, figsize, colors):
    # fig, ax = plt.subplots(figsize=figsize)
    sns.despine()
    width = 1
    space = 0.7
    num_bars = len(results)
    offset = 0

    for strat in ["Adaptive", "Quantile Regression", "No Batching"]:
        mean_b = [(m["name"].split(":")[0], m["mean"]) for m in results[strat]["histograms"] if "model_batch" in m["name"]]
        model_names, bs = zip(*mean_b)
        model_names = [name_map[m] for m in model_names]
        cur_rect = ax.bar(np.arange(len(bs))*width*(num_bars+ space) + width*offset, bs, color=colors[offset], width=width, label=strat)
        if strat == "Adaptive":
            utils.barchart_label(ax, cur_rect, 6, hmult=1.20, rot=12)
        else:
            utils.barchart_label(ax, cur_rect, 6, rot=12)
        if offset == 0:
            ax.set_xticks(np.arange(len(bs))*width*(num_bars + space) + width*(num_bars/2.0))
            ax.set_xticklabels(model_names, rotation=0, ha="center")
        offset += 1
    ax.set_ylim(0, 1800)
    ax.set_xlim(-0.3, ax.get_xlim()[1])
    ax.set_ylabel("Batch Size")
    ax.locator_params(nbins=3, axis="y")
    # ax.legend(frameon=True, bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3,
    #             ncol=3, mode="expand", borderaxespad=0.05, fontsize=7,)
    #
    # fname = "%s/batching_strategy_comp_batch_size.pdf" % (fig_dir)
    # plt.savefig(fname, bbox_inches='tight')
    # print(fname)

if __name__=='__main__':
    results = load_results()
    figsize = (4.0,2.0)
    fig, (ax_thru, ax_lat) = plt.subplots(nrows=2, figsize=figsize, sharex=True)
    # colors = sns.color_palette("Set1", n_colors=8, desat=.5)
    # colors = sns.color_palette("cubehelix", n_colors=3)
    colors = sns.cubehelix_palette(3, start=.75, rot=-.75)
    plot_thrus(ax_thru, results, figsize, colors)
    plot_latencies(ax_lat, results, figsize, colors)
    # plot_batch_sizes(ax_batch, results, figsize, colors)
    fig.subplots_adjust(hspace=0.3)


    fname = "%s/batching_strategy_comp_all.pdf" % (fig_dir)
    plt.savefig(fname, bbox_inches='tight')
    print(fname)


