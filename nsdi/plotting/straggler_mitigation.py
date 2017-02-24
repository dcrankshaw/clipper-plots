import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import seaborn as sns
import utils
# sns.set_style("white")
sns.set_style("darkgrid")
sns.set_context("paper", font_scale=1.2,)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

"""
    NOTES:
    This experiment examines the increase in mean and p99 latency
    when increasing the size of an ensemble. Each element in the
    ensemble in this experiment was a random forest of depth 16 with
    10 trees trained in Scikit-Learn. The experiment compares two
    latencies. The Clipper latencies are the latency of responses
    returned from Clipper, regardless of whether all the components
    of the ensemble have returned in time. The "Blocking" latency
    measures the latency of the last component in the ensemble to
    return a prediction, and is a measure of the prediction latency
    if Clipper did not return predictions early to meet latency objectives.
    Notably, the prediction load (requested throughput) was set to ensure
    that the blocking latency for an ensemble with one model was well within
    the latency objective. In all experiments the latency objective was
    set to 20ms.


"""

fig_dir = utils.NSDI_FIG_DIR
# fig_dir = os.path.abspath(".")

log_loc = os.path.abspath("../results/straggler_mitigation")
# colors = sns.color_palette("Set1", n_colors=8, desat=.5)
# colors = sns.cubehelix_palette(2, start=0.1, rot=-0.4, dark=0.1, light=0.5)
colors = sns.cubehelix_palette(2, start=1.2, rot=0.5, dark=0.0, light=0.4)
# colors = sns.color_palette('deep', 2)

def extract_results(i, df, fname):
    with open(os.path.join(log_loc, fname), "r") as f:
        ensemble_size = int(fname.split("_")[2])
        res = json.load(f)
        clipper_p99 = [m["p99"] for m in res["histograms"] if "prediction_latency" in m["name"]][0] / 1000.0
        blocking_p99 = [m["p99"] for m in res["histograms"] if "straggler_blocking_prediction_latency" in m["name"]][0] / 1000.0
        clipper_mean = [m["mean"] for m in res["histograms"] if "prediction_latency" in m["name"]][0] / 1000.0
        blocking_mean = [m["mean"] for m in res["histograms"] if "straggler_blocking_prediction_latency" in m["name"]][0] / 1000.0
        in_time_mean = 100 - ([m["mean"] for m in res["histograms"] if "in_time_predictions" in m["name"]][0] / float(ensemble_size) * 100)
        in_time_p99 = 100 - ([m["p99"] for m in res["histograms"] if "in_time_predictions" in m["name"]][0] / float(ensemble_size) * 100)
        if ensemble_size < 18:
            df.loc[i] = [ensemble_size, clipper_mean, clipper_p99, blocking_mean, blocking_p99, in_time_mean, in_time_p99]


def plot_line(cur_col, ax, label, color, marker, ls="-"):
    cur_col.plot(y="mean", yerr="std", ax=ax, marker=marker, color=color, ls=ls, label=label)

def plot_straggler_mitigation():
    results_files = utils.get_results_files(log_loc)
    df = pd.DataFrame(columns=("ensemble_size", "clipper_mean_lat", "clipper_p99", "blocking_mean_lat", "blocking_p99", "in_time_mean", "in_time_p99"))
    for (i,r) in enumerate(results_files):
        extract_results(i, df, r)
    df.sort_values("ensemble_size", inplace=True)
    f = {'clipper_mean_lat':['mean','std'],
         'clipper_p99':['mean','std'],
         'blocking_mean_lat':['mean','std'],
         'blocking_p99':['mean','std'],
         'in_time_mean':['mean','std'],
         'in_time_p99':['mean','std'],
         }
    tgs = df.groupby("ensemble_size").agg(f)
    # print tgs.index.values
    # print tgs["clipper_p99","mean"].values
    tgs.columns.get_level_values(0)
    # fig, (ax_lat, ax_in_time) = plt.subplots(nrows=2, sharex=False, figsize=(4,4), gridspec_kw = {'height_ratios':[2, 1.5]})
    # fig, (ax_lat, ax_in_time) = plt.subplots(ncols=2, sharex=False, figsize=(6,2))
    fig = plt.figure(figsize=(4,1.5))
    ax_lat = plt.gca()
    plot_line(tgs["clipper_p99"], ax_lat, "Straggler Mitigation P99", colors[0], "o")
    plot_line(tgs["clipper_mean_lat"], ax_lat, "Straggler Mitigation Mean", colors[0], "o", ls="--")
    plot_line(tgs["blocking_p99"], ax_lat, "Stragglers P99", colors[1], "v")
    plot_line(tgs["blocking_mean_lat"], ax_lat, "Stragglers Mean", colors[1], "v", ls="--")
    # plot_line(tgs["in_time_mean"], ax_in_time, "P99", colors[1], None)
    # plot_line(tgs["in_time_p99"], ax_in_time, "Mean", colors[1], None, ls="--")
    print(tgs["blocking_mean_lat"]["mean"])
    ax_lat.legend(loc=0, ncol=2, prop={'size':8})
    ax_lat.set_ylim(0, 300)
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_xlabel("Size of ensemble ")
    fname = "%s/straggler_mitigation_lat.pdf" % (fig_dir)
    fig.savefig(fname, bbox_inches='tight')
    print(fname)

    fig = plt.figure(figsize=(4,1.5))
    ax_in_time = plt.gca()
    plot_line(tgs["in_time_mean"], ax_in_time, "P99", colors[1], None)
    plot_line(tgs["in_time_p99"], ax_in_time, "Mean", colors[1], None, ls="--")
    ax_in_time.set_ylabel("% Ensemble Missing")
    ax_in_time.set_xlabel("Size of ensemble")
    ax_in_time.set_ylim(0, 100)
    ax_in_time.legend(loc=0, prop={'size':8})
    fig.subplots_adjust(wspace=0.3)
    fname = "%s/straggler_mitigation_in_time.pdf" % (fig_dir)
    fig.savefig(fname, bbox_inches='tight')
    print(fname)
    # fname = "%s/straggler_mitigation.pdf" % (fig_dir)
    # plt.savefig(fname, bbox_inches='tight')
    # print(fname)
    # plt.subplots_adjust(hspace=0.6)
    # extent = ax_lat.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fname = "%s/straggler_mitigation_lat.pdf" % (fig_dir)
    # fig.savefig(fname, bbox_inches=extent.expanded(1.35, 1.6))
    # print(fname)
    # extent = ax_in_time.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fname = "%s/straggler_mitigation_in_time.pdf" % (fig_dir)
    # fig.savefig(fname, bbox_inches=extent.expanded(1.35, 1.7))
    # print(fname)


def plot_ensemble_accuracy():
    # accuracies = [0.7825, 0.7598, 0.8394, 0.8592, 0.8748, 0.8848, 0.8924, 0.8954, 0.8982, 0.9007, 0.9059, 0.9088, 0.911, 0.9111, 0.9117, 0.9139]
    trials = np.array([[0.9146, 0.9117, 0.9113, 0.9094, 0.9082, 0.9051, 0.9014, 0.8976, 0.8965, 0.8863, 0.8787, 0.8708, 0.8656, 0.8467, 0.7788, 0.8173], [0.9165, 0.9159, 0.9128, 0.9109, 0.9084, 0.9067, 0.9062, 0.9028, 0.8992, 0.8916, 0.8877, 0.881, 0.8656, 0.8451, 0.7805, 0.8005], [0.9133, 0.9121, 0.9134, 0.913, 0.9124, 0.907, 0.9043, 0.9008, 0.8961, 0.8904, 0.8843, 0.8748, 0.856, 0.8255, 0.7524, 0.7984], [0.9117, 0.9119, 0.9108, 0.9075, 0.9046, 0.9032, 0.899, 0.8947, 0.8922, 0.8916, 0.8847, 0.8755, 0.8588, 0.8274, 0.7595, 0.7838], [0.9142, 0.9126, 0.9101, 0.9086, 0.9077, 0.9031, 0.8993, 0.8959, 0.8906, 0.8866, 0.8748, 0.8705, 0.8589, 0.8316, 0.7336, 0.7572], [0.9129, 0.9105, 0.9077, 0.9055, 0.9029, 0.8992, 0.8999, 0.8938, 0.8877, 0.8837, 0.8739, 0.865, 0.8457, 0.819, 0.7419, 0.7649]])
    trials = np.flip(trials, axis=1)

    means = np.mean(trials, axis=0)
    errs = np.std(trials, axis=0)
    fig = plt.figure(figsize=(4,1.5))
    ax_acc = plt.gca()
    # ax_acc.plot(range(1,len(accuracies) + 1), accuracies, color=colors[1])
    ax_acc.errorbar(range(1,len(means) + 1), means, yerr=errs, color=colors[1])
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlabel("Size of ensemble")
    ax_acc.set_ylim(0.75, 1)
    # ax_acc.legend(loc=0, prop={'size':8})
    fig.subplots_adjust(wspace=0.3)
    fname = "%s/straggler_mitigation_ensemble_acc.pdf" % (fig_dir)
    fig.savefig(fname, bbox_inches='tight')
    print(fname)

    
if __name__=="__main__":
    plot_straggler_mitigation()
    plot_ensemble_accuracy()
