

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
sns.set_context("paper", font_scale=1.3,)

"""
NOTES:
    Scaling prediction throughput as a function of replicas. The
    first replica is co-located with Clipper running in c67.millennium,
    the next three replicas are on c66, c68, c69 respectively. Each replicas
    is running on a GPU. Machines are connected with a 10Gbps Ethernet
    connection.
"""

fig_dir = utils.NSDI_FIG_DIR
# fig_dir = "test"
log_loc = os.path.abspath("../results/gpu_scaling_10g_network")

def is_replica_thru(n):
    return "model_thruput" in n and ("127" in n or "192" in n)

def get_all_thruputs(res, name):
    agg_thru = [m["rate"] for m in res["meters"] if name in m["name"]][0]
    rep_thrus = [m["rate"] for m in res["meters"] if is_replica_thru(m["name"])]
    print(rep_thrus)
    return (agg_thru, np.array(rep_thrus))

def extract_results(fname, model_name):
    with open(os.path.join(log_loc, fname), "r") as f:
        wait_time = int(fname.split("_")[2])*1000
        res = json.load(f)
        (agg_thru, rep_thrus) = get_all_thruputs(res, model_name)
        num_reps = len(rep_thrus)
        return (num_reps, agg_thru, np.mean(rep_thrus))

def plot_gpu_scaling():
    results = utils.get_results_files(log_loc)
    print(results)
    num_reps, agg_thrus, mean_thrus = zip(*[extract_results(r,  "conv:model_thruput") for r in results])

    colors = sns.color_palette("bright", n_colors=4, desat=.5)
    fig, ax_thru = plt.subplots(figsize=(4,4))
    ax_thru.plot(num_reps, agg_thrus, color=colors[0], label="Aggregate Throughput")
    ax_thru.scatter(num_reps, agg_thrus, color=colors[0])
    ax_thru.plot(num_reps, mean_thrus, color=colors[2], linestyle="dashed", label="Mean Throughput")
    ax_thru.set_ylabel("Throughput (qps)")
    ax_thru.set_xlabel("Number of Replicas")
    ax_thru.legend(loc=0)
    ax_thru.set_xlim(0.8, 4.2)
    ax_thru.set_ylim(0, 80000)
    fname = "%s/gpu_replicas_scaling.pdf" % (fig_dir)
    plt.savefig(fname, bbox_inches='tight')
    print(fname)

if __name__=="__main__":
    plot_gpu_scaling()

    

