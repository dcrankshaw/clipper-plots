import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import seaborn as sns
import utils
sns.set_style("white")
sns.set_context("paper", font_scale=1.8,)

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

log_loc = os.path.abspath("../results/straggler_mitigation")
colors = sns.color_palette("Set1", n_colors=8, desat=.5)

def extract_results(i, df, fname):
    with open(os.path.join(log_loc, fname), "r") as f:
        ensemble_size = int(fname.split("_")[2])
        res = json.load(f)
        clipper_p99 = [m["p99"] for m in res["histograms"] if "prediction_latency" in m["name"]][0]
        blocking_p99 = [m["p99"] for m in res["histograms"] if "straggler_blocking_prediction_latency" in m["name"]][0]
        clipper_mean = [m["mean"] for m in res["histograms"] if "prediction_latency" in m["name"]][0]
        blocking_mean = [m["mean"] for m in res["histograms"] if "straggler_blocking_prediction_latency" in m["name"]][0]
        if ensemble_size < 18:
            df.loc[i] = [ensemble_size, clipper_mean, clipper_p99, blocking_mean, blocking_p99]


def plot_line(cur_col, ax, label, color, ls="-"):
    cur_col.plot(y="mean", yerr="std", ax=ax, color=color, ls=ls, label=label)

def plot_straggler_mitigation():
    results_files = utils.get_results_files(log_loc)
    df = pd.DataFrame(columns=("ensemble_size", "clipper_mean_lat", "clipper_p99", "blocking_mean_lat", "blocking_p99"))
    for (i,r) in enumerate(results_files):
        extract_results(i, df, r)
    df.sort_values("ensemble_size", inplace=True)
    f = {'clipper_mean_lat':['mean','std'], 'clipper_p99':['mean','std'], 'blocking_mean_lat':['mean','std'], 'blocking_p99':['mean','std'],}
    tgs = df.groupby("ensemble_size").agg(f)
    # print tgs.index.values
    # print tgs["clipper_p99","mean"].values
    tgs.columns.get_level_values(0)
    fig, ax = plt.subplots()
    plot_line(tgs["clipper_p99"], ax, "Clipper P99", colors[0])
    plot_line(tgs["clipper_mean_lat"], ax, "Clipper Mean", colors[0], ls="--")
    plot_line(tgs["blocking_p99"], ax, "Blocking P99", colors[1])
    plot_line(tgs["blocking_mean_lat"], ax, "Blocking Mean", colors[1], ls="--")
    ax.legend(loc=0, ncol=2)
    ax.set_ylim(0, 250000)
    ax.set_ylabel("Latency ($\mu$s)")
    ax.set_xlabel("Size of ensemble")
    fname = "%s/straggler_mitigation.pdf" % (fig_dir)
    plt.savefig(fname, bbox_inches='tight')
    print(fname)

    
if __name__=="__main__":
    plot_straggler_mitigation()
