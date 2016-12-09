from __future__ import print_function
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import seaborn as sns
import utils
sns.set_style("white")
sns.set_context("paper", font_scale=1.0,)

"""
NOTES: Both models are linear SVMs, one trained in Spark,
on in Scikit-Learn. The batch wait timeout (x-axis) refers
to the maximum amount of time to delay sending a batch if the
batch is not at the maximum batch size. The delay time is measured
from time the earliest request in the batch was recieved (recv_time) to
the current time. So the spin loop looks like:
    while batch_size < current_max_batch_size
            && time::now - first_req.recv_time < delay_time {
            // spin
    }

"""

fig_dir = utils.NSDI_FIG_DIR
# fig_dir = "test"
log_loc = os.path.abspath("../results/batch_wait_time")

def extract_results(fname):
    with open(os.path.join(log_loc, fname), "r") as f:
        wait_time = int(fname.split("_")[2])*1000
        res = json.load(f)
        model_latency = [m["mean"] for m in res["histograms"] if "model_latency" in m["name"]][0]
        model_batch_size = [m["mean"] for m in res["histograms"] if "model_batch_size" in m["name"]][0]
        model_thruput = [m["rate"] for m in res["meters"] if "model_thruput" in m["name"]][0]
        return [wait_time, model_latency, model_batch_size, model_thruput]


def plot_wait_times():
    results = utils.get_results_files(log_loc)
    spark_results_files = [r for r in results if "spark" in r]
    sklearn_results_files = [r for r in results if "sklearn" in r]
    sklearn_df = pd.DataFrame(columns=("wait_time", "model_latency", "model_batch_size", "model_thruput"))
    spark_df = pd.DataFrame(columns=("wait_time", "model_latency", "model_batch_size", "model_thruput"))
    for (i,r) in enumerate(spark_results_files):
        spark_df.loc[i] = extract_results(r)
    for (i,r) in enumerate(sklearn_results_files):
        sklearn_df.loc[i] = extract_results(r)
    spark_df.sort_values("wait_time", inplace=True)
    sklearn_df.sort_values("wait_time", inplace=True)
    print(sklearn_df["model_thruput"])
    colors = sns.color_palette("deep", n_colors=4)
    colors[1] = colors[2]

    # fig, (ax_thru, ax_lat, ax_batch) = plt.subplots(nrows=3,figsize=(2.5,3), sharex=True)
    fig, (ax_thru, ax_lat, ax_batch) = plt.subplots(nrows=3,figsize=(3.0,2.0), sharex=True)

    ax_thru.plot(spark_df["wait_time"], spark_df["model_thruput"], marker="o", ms=4.5,  label="Spark SVM", color=colors[0])
    ax_thru.plot(sklearn_df["wait_time"], sklearn_df["model_thruput"], marker="^", ms=4.5, label="Scikit-Learn SVM", color=colors[1])
    ax_thru.set_ylabel("Thru.", rotation=0, ha='right', va='center')
    ax_thru.legend(loc=0)
    ax_thru.set_xlim(0, 4000)
    ax_thru.set_ylim(0, 20000)
    ax_thru.locator_params(nbins=3, axis="y")

    ax_lat.plot(spark_df["wait_time"], spark_df["model_latency"], marker="o", ms=4.5, label="Spark SVM", color=colors[0])
    ax_lat.plot(sklearn_df["wait_time"], sklearn_df["model_latency"], marker="^", ms=4.5, label="Scikit-Learn SVM", color=colors[1])
    ax_lat.set_ylabel("Lat.\n($\mu$s)", rotation=0, ha='right', va='center')
    ax_lat.set_xlim(0, 4000)
    ax_lat.set_ylim(0, 4000)
    ax_lat.locator_params(nbins=3, axis="y", )

    ax_batch.plot(spark_df["wait_time"], spark_df["model_batch_size"], marker="o", ms=4.5, label="Spark SVM", color=colors[0])
    ax_batch.plot(sklearn_df["wait_time"], sklearn_df["model_batch_size"], marker="^", ms=4.5, label="Scikit-Learn SVM", color=colors[1])
    ax_batch.set_ylabel("Batch\nSize", rotation=0, ha='right', va='center')
    ax_batch.set_xlabel("Batch Wait Timeout ($\mu$s)")
    ax_batch.set_xlim(0, 4000)
    ax_batch.set_ylim(0, 100)
    ax_batch.locator_params(nbins=3, axis="y")
    ax_batch.locator_params(nbins=4, axis="x")
    fname = "%s/batch_timeouts.pdf" % (fig_dir)
    fig.subplots_adjust(hspace=0.25)
    # fig.subplots_adjust(hspace=0.5)
    plt.savefig(fname, bbox_inches='tight')
    print(fname)

if __name__=="__main__":
    plot_wait_times()
