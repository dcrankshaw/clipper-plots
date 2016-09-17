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
sns.set_context("paper", font_scale=1.2)

fig_dir = utils.NSDI_FIG_DIR
# fig_dir = "test/"

log_loc = os.path.abspath("../results/feedback_caching")

def get_results(fname):
    res_fname = fname + "_results.json"
    conf_fname = fname + "_config.json"
    with open(os.path.join(log_loc, res_fname), "r") as rf, open(os.path.join(log_loc, conf_fname), "r") as cf:
        results = json.load(rf)
        conf = json.load(cf)
        cache_off = conf["clipper_conf"]["salt_cache"]
        window_size = conf["clipper_conf"]["window_size"]
        update_cache_rate = [rc["ratio"] for rc in results["ratio_counters"] if rc["name"] == "update_cache_hits"][0]
        update_thruput = [m["rate"] for m in results["meters"] if m["name"] == "update_thruput"][0]
    return (cache_off, window_size, update_cache_rate, update_thruput)


results_files = []
for name in os.listdir(log_loc):
    if "results" in name:
        results_files.append(name)

exp_names = [f.split("_results")[0] for f in results_files]


extracted_results = [get_results(e) for e in exp_names]


extracted_results.sort()
cache_state, windows, hit_rate, cache_on_thrus = zip(*extracted_results[:3])
cache_state, windows, hit_rate, cache_off_thrus = zip(*extracted_results[3:])
colors = sns.cubehelix_palette(2, start=2.8, rot=-0.1)
fig, ax = plt.subplots(figsize=(5,2.))
sns.despine()
width = 1
space = 0.5
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
r1 = ax.bar(np.arange(len(cache_off_thrus))*width*2.5,  cache_on_thrus, width=width, color=colors[0], label="caching")
r2 = ax.bar(np.arange(len(cache_off_thrus))*width*2.5 + 1,  cache_off_thrus, width=width, color=colors[1], label="no caching")
autolabel(r1)
autolabel(r2)

plt.xticks(np.arange(len(cache_off_thrus))*width*2.5 + 1, ["Window=%d" % w for w in windows])
ax.legend(loc=0)
# ax.set_xlabel("Update Window Size")
ax.set_ylabel("Update Throughput")
fname = "%s/caching_for_feedback_thruput.pdf" % (fig_dir)
plt.savefig(fname, bbox_inches='tight')
print(fname)
