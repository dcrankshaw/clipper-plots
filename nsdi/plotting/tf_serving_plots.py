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
# fig_dir = os.path.abspath(".")
fig_dir = os.path.expanduser("~/model-serving/clipper_paper/ModelServingPaper/nsdi_2017/figs2")
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

def plot_tf_vs_clipper_latency_breakdown(figsize, colors):
    fig, big_axes = plt.subplots(figsize=figsize , nrows=3, ncols=1)
    # fix, ax = plt.subplots(figsize=figsize, nrows=3, ncols=2)
    sns.despine()
    width = 1.0
    space = 0.7
    num_bars = 3

    thruput = {
        "MNIST": {"tf_serving": 22920.62122, "cpp_rpc": 22269.9215161283, "python": 19537.55802702},
        "CIFAR-10": {"tf_serving": 5293.848952, "cpp_rpc": 5472.43992468725, "python": 4571.24833224948},
        "ImageNet": {"tf_serving": 56.84785542, "cpp_rpc": 52.5174842517487, "python": 47.911036839022}
        }
    latencies = {
        "MNIST": {
            "tf_serving": {"total": 43647.37351},
            "cpp_rpc": {
                "total": 45951.8576325,
                "read": 1015.040495,
                "queue": 20964.37991,
                "predict": 22126.5715,
                "copy": 812.5615
            },
            "python": {
                "total": 52383.07,
                "read": 1051.548707,
                "queue": 24084.43692,
                "predict": 24582.0,
                "copy": 1556.0
            }
        },
        "CIFAR-10": {
            "tf_serving": {"total": 47036.4517},
            "cpp_rpc": {
                "total": 46754.26,
                "read": 1016.433575,
                "queue": 21350.17605,
                "predict": 22445.95,
                "copy": 887.2065
            },
            "python": {
                "total": 55976.57,
                "read": 1165.6867,
                "queue": 25770.02299,
                "predict": 26559.0,
                "copy": 1377.0
            }
        },
        "ImageNet": {
            "tf_serving": {"total": 561807.9518},
            "cpp_rpc": {
                "total": 608976.06,
                "read": 11717.84932,
                "queue": 274628.2358,
                "predict": 295214.296,
                "copy": 9165.353
            },
            "python": {
                "total": 667533.19,
                "read": 13376.78127,
                "queue": 301313.7111,
                "predict": 306706.0,
                "copy": 26906.0
            }
        }
    }


    p99 = {
        "MNIST": {
            "tf_serving": {"total": 44204},
            "cpp_rpc": {
                "total": 47536.592,
                "read": 1873.915,
                "queue": 21753.514,
                "predict": 22591,
                "copy": 1014.0
            },
            "python": {
                "total": 53720.212,
                "read": 1912.396,
                "queue": 24990.99,
                "predict": 25089,
                "copy": 1824.0
            }
        },
        "CIFAR-10": {
            "tf_serving": {"total": 51544},
            "cpp_rpc": {
                "total": 47504.959,
                "read": 1821.141,
                "queue": 21929.408,
                "predict": 22843.0,
                "copy": 1013.0
            },
            "python": {
                "total": 58613.286,
                "read": 2062.318,
                "queue": 27065.443,
                "predict": 27580.0,
                "copy": 1665.0,
            }
        },
        "ImageNet": {
            "tf_serving": {"total": 574258},
            "cpp_rpc": {
                "total": 617552.208,
                "read": 17150.503,
                "queue": 282130.765,
                "predict": 299284.0,
                "copy": 11681.0
            },
            "python": {
                "total": 687146.717,
                "read": 18427.674,
                "queue": 315596.227,
                "predict": 314844.0,
                "copy": 32563.0
            }
        }
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

        #first plot tf_serving with no breakdown
        offset = 0
        system = "tf_serving"
        err = (cur_p99[system]["total"] - cur_lat[system]["total"]) / 1000.0
        cur_rect = ax_thru.bar(width*offset, cur_thru[system],
                                color=colors[offset], width=width, label=name_map[system])
        utils.barchart_label(ax_thru, cur_rect, 7)
        cur_rect = ax_lat.bar(width*offset, cur_lat[system]["total"] / 1000.0,
                                color=colors[offset], width=width, label=name_map[system])
        ax_lat.errorbar(width*offset + (width / 2), cur_lat[system]["total"] / 1000.0,
                        yerr=[(0,), (err,)], ecolor='r', capthick=0.3, elinewidth=0.3)
        utils.barchart_label(ax_lat, cur_rect, 7, hmult=1.12)


        for system in ["cpp_rpc", "python"]:
            offset += 1
            cur_rect = ax_thru.bar(width*offset, cur_thru[system],
                                color=colors[offset], width=width, label=name_map[system])
            utils.barchart_label(ax_thru, cur_rect, 7)
            cur_bottom = 0.0
            patterns = ('o', '//', '\\', 'x', 'o', 'O', '.', "*")
            # for (stage_num, stage) in enumerate(["read", "queue", "predict", "copy", "total"]):
            for (stage_num, stage) in enumerate(["queue", "predict", "total"]):
                if stage != "total":
                    cur_rect = ax_lat.bar(width*offset, cur_lat[system][stage] / 1000.0,
                                        color=colors[offset], width=width, bottom=cur_bottom,
                                        # hatch=patterns[stage_num]
                                        )
                    
                    utils.barchart_label(ax_lat, cur_rect, 7, hmult=0.4, bottom = cur_bottom, label=stage)


                    cur_bottom += cur_lat[system][stage] / 1000.0
                    print(cur_bottom)


                else:
                    cur_rect = ax_lat.bar(width*offset, cur_lat[system][stage] / 1000.0 - cur_bottom,
                                        color=colors[offset], width=width, bottom=cur_bottom)
                    err = (cur_p99[system][stage] - cur_lat[system][stage]) / 1000.0
                    ax_lat.errorbar(width*offset + (width / 2), cur_lat[system][stage] / 1000.0,
                                    yerr=[(0,), (err,)], ecolor='r', capthick=0.3, elinewidth=0.3)
                    cur_rect.set_label(name_map[system])
                    utils.barchart_label(ax_lat, cur_rect, 7, hmult=1.12, bottom = cur_bottom)
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


if __name__=='__main__':
    figsize = (4.5,3)
    # figsize = (4*4.5,4*3)
    # colors = sns.cubehelix_palette(3, start=.75, rot=-.75)
    colors = sns.cubehelix_palette(3, start=.2, rot=-.6)
    colors.reverse()
    # colors = sns.color_palette("cubehelix", n_colors=3)
    # plot_thrus_latency(figsize, colors)
    # plot_tf_vs_clipper_cpp(figsize, colors)
    plot_tf_vs_clipper_latency_breakdown(figsize, colors)
