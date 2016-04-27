
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os, json

# fig_dir = os.getcwd()
fig_dir = "/Users/crankshaw/Dropbox/Apps/ShareLaTeX/velox-centipede/osdi_2016/figs"
results = []
for i in range(1,9,2):
    fn = os.path.abspath("../results/timit_user_eval_%d_to_%d.json" % (i, i + 1))
    with open(fn) as f:
        r = json.load(f)
        results += r
    print len(results)
    
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
cs = 10
el = 2

learned_ys = np.mean(learned_accs, axis=0)
num_train_examples = len(learned_accs)
se_div = np.sqrt(num_train_examples)
print learned_ys
learned_ys_errors = np.std(learned_accs, axis=0, ddof=1)
fig, ax = plt.subplots()
(_, caps1, _) = ax.errorbar(range(9), 1-learned_ys, yerr = learned_ys_errors/se_div, label="learned", capsize=cs, elinewidth=el)
(_, caps2, _) = ax.errorbar(range(9), 1-np.ones(9)*np.mean(gen_accs), yerr=np.ones(9)*np.std(gen_accs, ddof=1)/se_div, label="gen", capsize=cs, elinewidth=el)
(_, caps3, _) = ax.errorbar(range(9), 1-np.ones(9)*np.mean(dr_accs), yerr=np.ones(9)*np.std(dr_accs, ddof=1)/se_div, label="dr", capsize=cs, elinewidth=el)
ax.set_title("average error across users")
ax.set_xlabel("number of training examples")
ax.set_ylabel("phoneme error (holdout set)")
cap_width = 2
for cap in caps1:
    # cap.set_color('red')
    cap.set_markeredgewidth(cap_width)
for cap in caps2:
    # cap.set_color('red')
    cap.set_markeredgewidth(cap_width)
for cap in caps3:
    # cap.set_color('red')
    cap.set_markeredgewidth(cap_width)

ax.set_ylim((0.275, 0.38))
ax.set_xlim((0,8.2))
ax.legend(loc=3)
# sharelatex_path = os.path.expanduser("~/Dropbox/Apps/ShareLaTeX/velox-centipede/vldb_2016/figs")
# git_path = os.path.expanduser("~/ModelServingPaper/vldb_2016/figs")
plt.savefig(os.path.join(fig_dir, "timit_cold_start_err.pdf"))
