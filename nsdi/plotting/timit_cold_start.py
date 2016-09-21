
from __future__ import print_function
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import os, json
import seaborn as sns
import utils

sns.set_style("white")
sns.set_context("paper", font_scale=1.0,)
# matplotlib.rcParams['font.family'] = "Times New Roman"
# matplotlib.rcParams['font.size'] = 13
nbins=4

colors = sns.xkcd_palette(["amber", "faded green", "dusty purple"])
# fig_dir = os.getcwd()
# fig_dir = "/Users/crankshaw/ModelServingPaper/osdi_2016/figs"
# fig_dir = "/Users/giuliozhou/Research/RISE/ModelServingPaper/nsdi_2017/figs"
fig_dir = utils.NSDI_FIG_DIR
results = []
for i in range(1,9,2):
    fn = os.path.abspath("../results/timit/timit_user_eval_%d_to_%d.json" % (i, i + 1))
    with open(fn) as f:
        r = json.load(f)
        results += r
    
def acc(c, i, n):
    return (c - i)/float(n)

def user_acc(u):
    dialect_acc = acc(u['dialect']['corr'][0], u['dialect']['inserts'][0], u['dialect']['num'][0])
    general_acc = acc(u['general']['corr'][0], u['general']['inserts'][0], u['general']['num'][0])
    learned_acc = []
    for i in range(len(u['learned']['corr'])):
        learned_acc.append(acc(u['learned']['corr'][i], u['learned']['inserts'][i], u['learned']['num'][i]))
    return (dialect_acc, general_acc, learned_acc)

dr_accs = []
gen_accs = []
learned_accs = []

for u in results:
    d, g, l = user_acc(u)
    dr_accs.append(d)
    gen_accs.append(g)
    learned_accs.append(l)
dr_accs = np.array(dr_accs)
gen_accs = np.array(gen_accs)
learned_accs = np.array(learned_accs)
cs = 3
el = 1

learned_ys = np.mean(learned_accs, axis=0)
num_train_examples = len(learned_accs)
se_div = np.sqrt(num_train_examples)
learned_ys_errors = np.std(learned_accs, axis=0, ddof=1)
fig, ax = plt.subplots(figsize=(2.2,1.0))
(l1, caps1, _) = ax.errorbar(range(9), 1-learned_ys, yerr = learned_ys_errors/se_div, color=colors[0],label="Clipper", capsize=cs, elinewidth=el)
(l3, caps3, _) = ax.errorbar(range(9), 1-np.ones(9)*np.mean(dr_accs), yerr=np.ones(9)*np.std(dr_accs, ddof=1)/se_div, color=colors[2], label="dialect", capsize=cs, elinewidth=el)
(l2, caps2, _) = ax.errorbar(range(9), 1-np.ones(9)*np.mean(gen_accs), yerr=np.ones(9)*np.std(gen_accs, ddof=1)/se_div, color=colors[1], label="gen", capsize=cs, elinewidth=el)
# ax.set_title("average error across users")

dashes = [9, 3]  # 10 points on, 5 off, 100 on, 5 off
l2.set_dashes(dashes)
dashes = [3, 3]  # 10 points on, 5 off, 100 on, 5 off
l3.set_dashes(dashes)

ax.set_xlabel("Updates")
ax.set_ylabel("Error")
cap_width = 1
for cap in caps1:
    # cap.set_color('red')
    cap.set_markeredgewidth(cap_width)
for cap in caps2:
    # cap.set_color('red')
    cap.set_markeredgewidth(cap_width)
for cap in caps3:
    # cap.set_color('red')
    cap.set_markeredgewidth(cap_width)

# ax.set_ylim((0.275, 0.38))
ax.set_ylim((0.21, 0.38))
ax.set_xlim((0,8.2))
ax.locator_params(tight=True, nbins=nbins)
ax.legend(loc=3,ncol=2, fontsize='small', handlelength=2.4)
# sharelatex_path = os.path.expanduser("~/Dropbox/Apps/ShareLaTeX/velox-centipede/vldb_2016/figs")
# git_path = os.path.expanduser("~/ModelServingPaper/vldb_2016/figs")

# fig.set_size_inches(3.0, 1.8)
fname = os.path.join(fig_dir, "timit_cold_start_err.pdf")
plt.savefig(fname, bbox_inches='tight')

print(fname)
