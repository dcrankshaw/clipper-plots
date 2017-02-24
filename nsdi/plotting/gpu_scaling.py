

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

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

"""
NOTES:
    Scaling prediction throughput as a function of replicas. The
    first replica is co-located with Clipper running in c67.millennium,
    the next three replicas are on c66, c68, c69 respectively. Each replicas
    is running on a GPU. Machines are connected with a 10Gbps Ethernet
    connection.

    Note that the input here is MNIST, so 784*8 bytes * 8 bits/byte = 50,176 bits
"""

fig_dir = utils.NSDI_FIG_DIR
# fig_dir = os.path.abspath(".")
log_loc = os.path.abspath("../results/gpu_scaling_10g_network")


# from https://docs.google.com/spreadsheets/d/1SmgxZdN1yqUohDCSdFPAZ3Za1HX9_LCfKI-YAOppSdk/edit?usp=sharing
one_gbps_results = ([1,2,3,4], [19727.83464, 35456.12888, 38720.54293, 38523.7034], [
np.mean([19727.83464,]),
np.mean([19866.25684, 15589.87152]),
np.mean([20025.47685, 9330.474972, 9364.58981]),
np.mean([19822.53465, 6238.995755, 6231.073998, 6231.098988])])

one_gbps_mean_lats = np.array([46454015.80909533, 51679662.88655156, 73385925.91755837, 99837800.23115273]) / 1000.0
one_gbps_p99_lats = np.array([49428749.5, 62190651.0, 104841473.75, 160191884.0]) / 1000.0

def is_replica_thru(n):
    return "model_thruput" in n and ("127" in n or "192" in n)

def get_all_thruputs(res, name):
    agg_thru = [m["rate"] for m in res["meters"] if "%s:model_thruput" % name in m["name"]][0]
    rep_thrus = [m["rate"] for m in res["meters"] if is_replica_thru(m["name"])]
    mean_lat = [m["mean"] for m in res["histograms"] if "%s:model_lat" % name in m["name"]][0]
    p99_lat = [m["p99"] for m in res["histograms"] if "%s:model_lat" % name in m["name"]][0]
    return (agg_thru, np.array(rep_thrus), mean_lat / 1000.0, p99_lat / 1000.0)

def extract_results(fname, model_name):
    with open(os.path.join(log_loc, fname), "r") as f:
        res = json.load(f)
        (agg_thru, rep_thrus, mean_lat, p99_lat) = get_all_thruputs(res, model_name)
        num_reps = len(rep_thrus)
        return (num_reps, agg_thru, np.mean(rep_thrus), mean_lat, p99_lat)

def plot_gpu_scaling():
    results = utils.get_results_files(log_loc)
    print(results)
    num_reps, agg_thrus, mean_thrus, mean_lats, p99_lats = zip(*[extract_results(r,  "conv") for r in results])
    slow_net_num_reps, slow_net_agg_thrus, slow_net_mean_thrus = one_gbps_results

    vol = 7

    # colors = sns.color_palette("bright", n_colors=4, desat=.5)
    colors = sns.cubehelix_palette(4, start=0.5, rot=-0.75, dark=0.1, light=0.6)
    fig, (ax_thru, ax_lat) = plt.subplots(nrows=2, sharex=True, figsize=(4,2.2))
    ax_thru.plot(num_reps, np.array(agg_thrus) / 10000.0, color=colors[0], marker="o", ms=vol, label="Agg 10Gbps")
    ax_thru.scatter(num_reps, np.array(agg_thrus) / 10000.0, color=colors[0], marker="o", s=vol)
    ax_thru.plot(slow_net_num_reps, np.array(slow_net_agg_thrus) / 10000.0, color=colors[2], marker="d", ms=vol, label="Agg 1Gbps")
    ax_thru.scatter(slow_net_num_reps, np.array(slow_net_agg_thrus) / 10000.0, color=colors[2], marker="d", s=vol)
    ax_thru.plot(num_reps, np.array(mean_thrus) / 10000.0, color=colors[0], linestyle="dashed", label="Mean 10Gbps")
    ax_thru.plot(slow_net_num_reps, np.array(slow_net_mean_thrus) / 10000.0, color=colors[2], linestyle="dashdot", label="Mean 1Gbps")

    ax_lat.plot(num_reps, np.array(mean_lats) / 1000.0, color=colors[0], linestyle="dashed", label="Mean 10Gbps")
    ax_lat.plot(num_reps, np.array(p99_lats)  / 1000.0, color=colors[0], marker="o", ms=vol, label="P99 10Gbps")
    ax_lat.plot(num_reps, np.array(one_gbps_mean_lats) / 1000.0, color=colors[2], linestyle="dashdot", label="Mean 1Gbps")
    ax_lat.plot(num_reps, np.array(one_gbps_p99_lats) / 1000.0 , color=colors[2], marker="d", ms=vol, label="P99 1Gbps")





    # ax_thru.plot((0,4.2), np.ones(2)*19929.846938776, color=colors[3], linestyle="dotted", label = "Network bandwidth (1Gbps)")
    ax_thru.set_ylabel("Throughput\n(10K qps)")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_xlabel("Number of Replicas")
    ax_thru.legend(loc=0, ncol=2)
    ax_lat.legend(loc=0, ncol=2)
    ax_thru.set_xlim(0.8, 4.2)
    ax_thru.set_ylim(0, 11)
    ax_lat.locator_params(nbins=3)
    ax_thru.locator_params(nbins=3)
    ax_lat.xaxis.set_ticks(range(1, 5))
    fig.subplots_adjust(hspace=0.12)
    fname = "%s/gpu_replicas_scaling.pdf" % (fig_dir)
    plt.savefig(fname, bbox_inches='tight')
    print(fname)

if __name__=="__main__":
    plot_gpu_scaling()

    

