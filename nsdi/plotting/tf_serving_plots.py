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
sns.set_context("paper", font_scale=0.9,)


fig_dir = utils.NSDI_FIG_DIR
results_dir = os.path.abspath("../results/tf_serving_comparison")

name_map = {
        "tf_serving": "TensorFlow Serving",
        "cpp_rpc": "Clipper TF-C++",
        "python": "Clipper TF-Python",
        }

# def plot_thrus(results, figsize, colors):
def plot_thrus(figsize, colors):
    fig, ax = plt.subplots(figsize=figsize)
    sns.despine()
    width = 1.0
    space = 0.7
    # num_bars = len(results)
    num_bars = 3
    offset = 0
    thruput = {
        "MNIST": {"tf_serving": 22795.805, "cpp_rpc": 21390.58408, "python": 19869.90862},
        "Cifar10": {"tf_serving": 5228, "cpp_rpc": 5346.845183, "python": 4461.848557},
        "ImageNet": {"tf_serving": 57.35, "cpp_rpc": 53.71576, "python": 46.99235537}
        }
    for system in ["tf_serving", "cpp_rpc", "python"]:
        thrus = [(key, thruput[key][system]) for key in ["MNIST", "Cifar10", "ImageNet"]]
        model_names, rates = zip(*thrus)
        cur_rect = ax.bar(np.arange(len(rates))*width*(num_bars+ space) + width*offset, rates,
                          color=colors[offset], width=width, label=name_map[system])
        utils.barchart_label(ax, cur_rect, 6, rot=10)
        if offset == 0:
            ax.set_xticks(np.arange(len(rates))*width*(num_bars + space) + width*(num_bars/2.0))
            ax.set_xticklabels(model_names, rotation=0, ha="center")
        offset += 1
    ax.set_ylim(0, 28000)
    ax.set_xlim(-0.3, ax.get_xlim()[1])
    ax.set_ylabel("Throughput (qps)")
    ax.legend(frameon=True, bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3,
                ncol=3, mode="expand", borderaxespad=0.05, fontsize=7,)
    fname = "%s/tf_serving_thruput.pdf" % (fig_dir)
    plt.savefig(fname, bbox_inches='tight')
    print(fname)

# def plot_latencies(results, figsize, colors):
def plot_latencies(figsize, colors):
    fig, ax = plt.subplots(figsize=figsize)
    sns.despine()
    width = 1
    space = 0.7
    # num_bars = len(results)
    num_bars = 3
    offset = 0
    latencies = {
        "MNIST": {"tf_serving": 43661.75, "cpp_rpc": 47821.65, "python": 45134.22536},
        "Cifar10": {"tf_serving": 47298, "cpp_rpc": 47833.8615, "python": 56240.97845},
        "ImageNet": {"tf_serving": 554837, "cpp_rpc": 595054.541, "python": 663444.01029}
        }
    for system in ["tf_serving", "cpp_rpc", "python"]:
        lat = [(key, latencies[key][system] / 1000.) for key in ["MNIST", "Cifar10", "ImageNet"]]
        model_names, lats = zip(*lat)
        cur_rect = ax.bar(np.arange(len(lats))*width*(num_bars+ space) + width*offset, lats,
                          color=colors[offset], width=width, label=name_map[system])
        utils.barchart_label(ax, cur_rect, 6, rot=10)
        if offset == 0:
            ax.set_xticks(np.arange(len(lats))*width*(num_bars + space) + width*(num_bars/2.0))
            ax.set_xticklabels(model_names, rotation=0, ha="center")
        offset += 1
    ax.set_ylim(0, 850)
    ax.set_xlim(-0.3, ax.get_xlim()[1])
    ax.legend(frameon=True, bbox_to_anchor=(0.0, 1.02, 1.0, .102), loc=3,
                ncol=3, mode="expand", borderaxespad=0.05, fontsize=7,)

    ax.set_ylabel("Mean Latency (ms)")
    fname = "%s/tf_serving_lat.pdf" % (fig_dir)
    plt.savefig(fname, bbox_inches='tight')
    print(fname)

if __name__=='__main__':
    # results = load_results()
    figsize = (5.2,1)
    colors = sns.color_palette("Set1", n_colors=8, desat=.5)
    # plot_thrus(results, figsize, colors)
    plot_thrus(figsize, colors)
    plot_latencies(figsize, colors)
    # plot_latencies(results, figsize, colors)
    # plot_batch_sizes(results, figsize, colors)
