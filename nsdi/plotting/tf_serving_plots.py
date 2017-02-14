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


# fig_dir = utils.NSDI_FIG_DIR
fig_dir = os.path.abspath(".")
results_dir = os.path.abspath("../results/tf_serving_comparison")

name_map = {
        "tf_serving": "TensorFlow Serving",
        "cpp_rpc": "Clipper TF-C++",
        "python": "Clipper TF-Python",
        }

def plot_thrus_latency(figsize, colors):
    fig, big_axes = plt.subplots(figsize=figsize , nrows=3, ncols=1)
    # fix, ax = plt.subplots(figsize=figsize, nrows=3, ncols=2)
    sns.despine()
    width = 1.0
    space = 0.7
    num_bars = 3
    thruput = {
        "MNIST": {"tf_serving": 22920.62122, "cpp_rpc": 21705.43172, "python": 19109.23245},
        "CIFAR-10": {"tf_serving": 5293.848952, "cpp_rpc": 5297.627025, "python": 4448.834138},
        "ImageNet": {"tf_serving": 56.84785542, "cpp_rpc": 53.39, "python": 46.78688182}
        }
    latencies = {
        "MNIST": {"tf_serving": 43647.37351, "cpp_rpc": 47147.81, "python": 53556.25},
        "CIFAR-10": {"tf_serving": 47036.4517, "cpp_rpc": 48278.62, "python": 56495.01501},
        "ImageNet": {"tf_serving": 561807.9518, "cpp_rpc": 598744.5983, "python": 683217.75}
        }
    p99 = {
        "MNIST": {"tf_serving": 44204, "cpp_rpc": 47899.781, "python": 55176.174},
        "CIFAR-10": {"tf_serving": 51544, "cpp_rpc": 48976.383, "python": 60299.221},
        "ImageNet": {"tf_serving": 574258, "cpp_rpc": 618484.563, "python": 714437.516}
        }

    yranges = {
        "MNIST": [28000, 70.0],
        "CIFAR-10": [7000, 70.0],
        "ImageNet": [70, 850.0]
        }
    titles = ["a) MNIST", "b) CIFAR-10", "c) ImageNet"]
    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title(titles[row-1], fontsize=8, fontweight='bold')
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False
    ax = [[fig.add_subplot(3, 2, 2 * i + j + 1) for j in range(2)] for i in range(3)]

    for (i,m) in enumerate(["MNIST", "CIFAR-10", "ImageNet"]):
        ax_thru = ax[i][0]
        ax_lat = ax[i][1]
        cur_thru = thruput[m]
        cur_lat = latencies[m]
        cur_p99 = p99[m]
        for (offset,system) in enumerate(["tf_serving", "cpp_rpc", "python"]):
            err = (cur_p99[system] - cur_lat[system]) / 1000.0
            cur_rect = ax_thru.bar(width*offset, cur_thru[system],
                                   color=colors[offset], width=width, label=name_map[system])
            utils.barchart_label(ax_thru, cur_rect, 6)
            cur_rect = ax_lat.bar(width*offset, cur_lat[system] / 1000.0,
                                  color=colors[offset], width=width, label=name_map[system])
            ax_lat.errorbar(width*offset + (width / 2), cur_lat[system] / 1000.0,
                            yerr=[(0,), (err,)], ecolor='r', capthick=0.3, elinewidth=0.3)
            utils.barchart_label(ax_lat, cur_rect, 6, hmult=1.12)
        ax_thru.get_xaxis().set_visible(False)
        ax_thru.set_ylabel("Throughput")
        ax_lat.set_ylabel("Mean Lat.\n(ms)")
        ax_thru.set_ylim(0, yranges[m][0])
        ax_lat.get_xaxis().set_visible(False)
        ax_lat.set_ylim(0, yranges[m][1])

        ax_thru.locator_params(nbins=4, axis="y")
        ax_lat.locator_params(nbins=4, axis="y")
    legend = ax[0][0].legend(frameon=True, bbox_to_anchor=(-0.4, 1.3, 2.83, .102), loc=3,
                ncol=3, mode="expand", borderaxespad=0.1, fontsize=8,)
    plt.subplots_adjust(wspace=0.42, hspace=0.30, bottom=0.04, left=0.18, right=0.94, top=0.88)
    fname = "%s/tf_serving_latency_thruput.pdf" % (fig_dir)
    plt.savefig(fname, bbox_inches='tight')
    print(fname)

def plot_tf_vs_clipper_cpp(figsize, colors):
    fig, big_axes = plt.subplots(figsize=figsize , nrows=3, ncols=1)
    # fix, ax = plt.subplots(figsize=figsize, nrows=3, ncols=2)
    sns.despine()
    width = 1.0
    space = 0.7
    num_bars = 2
    thruput = {
        "MNIST": {"tf_serving": 22920.62122, "cpp_rpc": 21705.43172, "python": 19109.23245},
        "CIFAR-10": {"tf_serving": 5293.848952, "cpp_rpc": 5297.627025, "python": 4448.834138},
        "ImageNet": {"tf_serving": 56.84785542, "cpp_rpc": 53.39, "python": 46.78688182}
        }
    latencies = {
        "MNIST": {"tf_serving": 43647.37351, "cpp_rpc": 47147.81, "python": 53556.25},
        "CIFAR-10": {"tf_serving": 47036.4517, "cpp_rpc": 48278.62, "python": 56495.01501},
        "ImageNet": {"tf_serving": 561807.9518, "cpp_rpc": 598744.5983, "python": 683217.75}
        }
    p99 = {
        "MNIST": {"tf_serving": 44204, "cpp_rpc": 47899.781, "python": 55176.174},
        "CIFAR-10": {"tf_serving": 51544, "cpp_rpc": 48976.383, "python": 60299.221},
        "ImageNet": {"tf_serving": 574258, "cpp_rpc": 618484.563, "python": 714437.516}
        }

    yranges = {
        "MNIST": [28000, 70.0],
        "CIFAR-10": [7000, 70.0],
        "ImageNet": [70, 850.0]
        }
    titles = ["a) MNIST", "b) CIFAR-10", "c) ImageNet"]
    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title(titles[row-1], fontsize=8, fontweight='bold')
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False
    ax = [[fig.add_subplot(3, 2, 2 * i + j + 1) for j in range(2)] for i in range(3)]

    for (i,m) in enumerate(["MNIST", "CIFAR-10", "ImageNet"]):
        ax_thru = ax[i][0]
        ax_lat = ax[i][1]
        cur_thru = thruput[m]
        cur_lat = latencies[m]
        cur_p99 = p99[m]
        for (offset,system) in enumerate(["tf_serving", "cpp_rpc"]):
            err = (cur_p99[system] - cur_lat[system]) / 1000.0
            cur_rect = ax_thru.bar(width*offset, cur_thru[system],
                                   color=colors[offset], width=width, label=name_map[system])
            utils.barchart_label(ax_thru, cur_rect, 6)
            cur_rect = ax_lat.bar(width*offset, cur_lat[system] / 1000.0,
                                  color=colors[offset], width=width, label=name_map[system])
            ax_lat.errorbar(width*offset + (width / 2), cur_lat[system] / 1000.0,
                            yerr=[(0,), (err,)], ecolor='r', capthick=0.3, elinewidth=0.3)
            utils.barchart_label(ax_lat, cur_rect, 6, hmult=1.12)
        ax_thru.get_xaxis().set_visible(False)
        ax_thru.set_ylabel("Throughput")
        ax_lat.set_ylabel("Mean Lat.\n(ms)")
        ax_thru.set_ylim(0, yranges[m][0])
        ax_lat.get_xaxis().set_visible(False)
        ax_lat.set_ylim(0, yranges[m][1])

        ax_thru.locator_params(nbins=4, axis="y")
        ax_lat.locator_params(nbins=4, axis="y")
    legend = ax[0][0].legend(frameon=True, bbox_to_anchor=(-0.4, 1.3, 2.83, .102), loc=3,
                ncol=2, mode="expand", borderaxespad=0.1, fontsize=8,)
    plt.subplots_adjust(wspace=0.42, hspace=0.30, bottom=0.04, left=0.18, right=0.94, top=0.88)
    fname = "%s/tf_serving_latency_thruput.pdf" % (fig_dir)
    plt.savefig(fname, bbox_inches='tight')
    print(fname)



if __name__=='__main__':
    figsize = (4.5,3)
    colors = sns.cubehelix_palette(3, start=.75, rot=-.75)
    # colors = sns.color_palette("cubehelix", n_colors=3)
    # plot_thrus_latency(figsize, colors)
    plot_tf_vs_clipper_cpp(figsize, colors)
