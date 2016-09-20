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
        "cpp_rpc": "Clipper C++ MW",
        "python": "Clipper Python MW",
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
                                  color=colors[offset], width=width, label=name_map[system],
                                  yerr=[(0,), (err,)], ecolor='k', capsize=5)
            utils.barchart_label(ax_lat, cur_rect, 6, hmult=1.12)
        ax_thru.get_xaxis().set_visible(False)
        ax_thru.set_ylabel("Throughput")
        ax_lat.set_ylabel("Mean Lat.\n(ms)")
        ax_thru.set_ylim(0, yranges[m][0])
        ax_lat.get_xaxis().set_visible(False)
        ax_lat.set_ylim(0, yranges[m][1])

        ax_thru.locator_params(nbins=4, axis="y")
        ax_lat.locator_params(nbins=4, axis="y")
    legend = ax[0][0].legend(frameon=True, bbox_to_anchor=(-0.4, 1.2, 2.83, .102), loc=3,
                ncol=3, mode="expand", borderaxespad=0.1, fontsize=8,)
    plt.subplots_adjust(wspace=0.42, hspace=0.30, bottom=0.04, left=0.18, right=0.94, top=0.88)
    # plt.show()
    fname = "%s/tf_serving_latency_thruput.pdf" % (fig_dir)
    plt.savefig(fname, bbox_inches='tight')
    print(fname)


# def plot_thrus(results, figsize, colors):
def plot_thrus(figsize, colors):
    # fig, ax = plt.subplots(figsize=figsize)
    fig, (ax_mnist, ax_cifar, ax_imagenet) = plt.subplots(figsize=figsize, ncols=3)
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
        cur_rect = ax_mnist.bar(width*offset, rates[0],
                          color=colors[offset], width=width, label=name_map[system])
        utils.barchart_label(ax_mnist, cur_rect, 6, rot=10)
        cur_rect = ax_cifar.bar(width*offset, rates[1],
                          color=colors[offset], width=width, label=name_map[system])
        utils.barchart_label(ax_cifar, cur_rect, 6, rot=10)
        cur_rect = ax_imagenet.bar(width*offset, rates[2],
                          color=colors[offset], width=width, label=name_map[system])
        utils.barchart_label(ax_imagenet, cur_rect, 6, rot=10)
        if offset == 0:
            ax_mnist.set_xticks(np.arange(len(rates))*width*(num_bars + space) + width*(num_bars/2.0))
            ax_mnist.set_xticklabels([model_names[0]], rotation=0, ha="center")
            ax_cifar.set_xticks(np.arange(len(rates))*width*(num_bars + space) + width*(num_bars/2.0))
            ax_cifar.set_xticklabels([model_names[1]], rotation=0, ha="center")
            ax_imagenet.set_xticks(np.arange(len(rates))*width*(num_bars + space) + width*(num_bars/2.0))
            ax_imagenet.set_xticklabels([model_names[2]], rotation=0, ha="center")
        offset += 1
    ax_mnist.set_ylim(0, 30000)
    ax_mnist.set_ylabel("Throughput (qps)")
    ax_cifar.set_ylim(0, 7000)
    ax_imagenet.set_ylim(0, 70)
    legend = ax_cifar.legend(frameon=True, bbox_to_anchor=(-1.21, 1.05, 3.5, .102), loc=3,
                ncol=4, mode="expand", borderaxespad=0.1, fontsize=8,)

    fname = "%s/tf_serving_thruput.pdf" % (fig_dir)
    plt.savefig(fname, bbox_inches='tight')
    print(fname)

# def plot_latencies(results, figsize, colors):
def plot_latencies(figsize, colors):
    # fig, ax = plt.subplots(figsize=figsize)
    fig, (ax_mnist, ax_cifar, ax_imagenet) = plt.subplots(figsize=figsize, ncols=3)
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
        cur_rect = ax_mnist.bar(width*offset, lats[0],
                          color=colors[offset], width=width, label=name_map[system])
        utils.barchart_label(ax_mnist, cur_rect, 6, rot=10)
        cur_rect = ax_cifar.bar(width*offset, lats[1],
                          color=colors[offset], width=width, label=name_map[system])
        utils.barchart_label(ax_cifar, cur_rect, 6, rot=10)
        cur_rect = ax_imagenet.bar(width*offset, lats[2],
                          color=colors[offset], width=width, label=name_map[system])
        utils.barchart_label(ax_imagenet, cur_rect, 6, rot=10)
        if offset == 0:
            ax_mnist.set_xticks(np.arange(len(lats))*width*(num_bars + space) + width*(num_bars/2.0))
            ax_mnist.set_xticklabels([model_names[0]], rotation=0, ha="center")
            ax_cifar.set_xticks(np.arange(len(lats))*width*(num_bars + space) + width*(num_bars/2.0))
            ax_cifar.set_xticklabels([model_names[1]], rotation=0, ha="center")
            ax_imagenet.set_xticks(np.arange(len(lats))*width*(num_bars + space) + width*(num_bars/2.0))
            ax_imagenet.set_xticklabels([model_names[2]], rotation=0, ha="center")
        offset += 1
    ax_mnist.set_ylim(0, 60)
    ax_mnist.set_ylabel("Mean Latency (ms)")
    ax_cifar.set_ylim(0, 60)
    ax_imagenet.set_ylim(0, 800)
    legend = ax_cifar.legend(frameon=True, bbox_to_anchor=(-1.21, 1.05, 3.5, .102), loc=3,
                ncol=4, mode="expand", borderaxespad=0.1, fontsize=8,)

    fig.subplots_adjust(hspace=0.3)
    fname = "%s/tf_serving_lat.pdf" % (fig_dir)
    plt.savefig(fname, bbox_inches='tight')
    print(fname)

if __name__=='__main__':
    # figsize = (5.2,1)
    figsize = (4.5,3)
    colors = sns.color_palette("cubehelix", n_colors=3)
    plot_thrus_latency(figsize, colors)
    # plot_thrus(results, figsize, colors)
    # plot_thrus(figsize, colors)
    # plot_latencies(figsize, colors)
    # plot_latencies(results, figsize, colors)
    # plot_batch_sizes(results, figsize, colors)
