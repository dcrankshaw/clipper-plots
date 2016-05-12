import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import os, json

fig_dir = os.getcwd()
fig_dir = "/Users/crankshaw/ModelServingPaper/osdi_2016/figs"
# fig_dir = "/Users/crankshaw/Dropbox/Apps/ShareLaTeX/velox-centipede/osdi_2016/figs"
matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['font.size'] = 20
def analyze_run(t):
    accs = []
    thrus = []
    mean_lats = []
    p99_lats = []
    max_lats = []
    # throw out first trial to let things warm up
    for i in range(1, len(t)):
        sp = t[i].split(", ")
        accs.append(float(sp[0]))
        thrus.append(float(sp[1]))
        mean_lats.append(float(sp[2]))
        p99_lats.append(float(sp[3]))
        max_lats.append(float(sp[4]))

    acc_m, acc_err = np.mean(accs), np.std(accs)/(np.sqrt(len(t) - 1))
    thrus_m, thrus_err = np.mean(thrus), np.std(thrus)/(np.sqrt(len(t) - 1))
    mean_lats_m, mean_lats_err = np.mean(mean_lats), np.std(mean_lats)/(np.sqrt(len(t) - 1))
    p99_lats_m, p99_lats_err = np.mean(p99_lats), np.std(p99_lats)/(np.sqrt(len(t) - 1))
    max_lats_m, max_lats_err = np.mean(max_lats), np.std(max_lats)/(np.sqrt(len(t) - 1))
    return acc_m, acc_err, thrus_m, thrus_err, mean_lats_m, mean_lats_err, p99_lats_m, p99_lats_err, max_lats_m, max_lats_err


# matplotlib.rcParams.update({'font.size': 14})
path = os.path.abspath("../results/end_to_end_THRUPUT")
# plt.figure(figsize=(5,8))
fig, (ax_acc, ax_lat) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
colors = ["r", "b", "g"]
markers = ["o", "v", "s"]
i = 0
for fname in os.listdir(path):
    print fname
    runs = []
    with open(os.path.join(path,fname), "rb") as rf:
        cur_exp = None
        for line in rf:
            if line.startswith("EXPERIMENT"):
                if cur_exp is not None:
                    runs.append(cur_exp)
                cur_exp = []
            elif line.startswith("0."):
                cur_exp.append(line.strip())
        runs.append(cur_exp)

    acc_m, acc_err, thrus_m, thrus_err, mean_lats_m, mean_lats_err, p99_lats_m, p99_lats_err, max_lats_m, max_lats_err = zip(*[analyze_run(r) for r in runs])
    acc_m = np.array(acc_m)
    acc_err = np.array(acc_err)
    thrus_m = np.array(thrus_m)
    thrus_err = np.array(thrus_err)
    mean_lats_m = np.array(mean_lats_m)
    mean_lats_err = np.array(mean_lats_err)
    p99_lats_m = np.array(p99_lats_m)
    p99_lats_err = np.array(p99_lats_err)
    max_lats_m = np.array(max_lats_m)
    max_lats_err = np.array(max_lats_err)

    ts = np.array(np.argsort(thrus_m))
    label = fname[:-4]
    c = colors[i % len(colors)]
    m = markers[i % len(markers)]
    s = 100
    ax_acc.errorbar(thrus_m[ts], acc_m[ts], yerr=acc_err[ts], fmt="%s%s-" % (c,m), ms=10, label="%s" % label)
    print thrus_m[ts]
    print acc_m[ts]



#     ax_lat.errorbar(thrus_m[ts], mean_lats_m[ts], yerr=mean_lats_err[ts], fmt="%s%s-" % (c,m), label="%s" % label)
#     ax_lat.errorbar(thrus_m[ts], p99_lats_m[ts], yerr=p99_lats_err[ts], fmt="%s%s--" % (c,m))
    if i is 2:
        ax_lat.scatter(thrus_m[ts], p99_lats_m[ts], s=s, c="w", edgecolors=c, marker=m, label="p99")
        ax_lat.scatter(thrus_m[ts], mean_lats_m[ts], s=s, c=c, marker=m, label="mean")


    else:
        ax_lat.scatter(thrus_m[ts], mean_lats_m[ts], s=s, c=c, marker=m)
        ax_lat.scatter(thrus_m[ts], p99_lats_m[ts], s=s, c="w", edgecolors=c, marker=m,)
        

    
    i += 1

ax_acc.set_ylabel("test accuracy")
ax_acc.set_ylim((0.48, 1.0))
ax_acc.legend(loc=1, fontsize='small', ncol=3)
ax_lat.set_ylabel("latency (ms)")
ax_lat.set_xlabel("throughput (predictions/sec)")
ax_lat.legend(loc=4,fontsize='small')
ax_lat.set_ylim((0, 32))
# fig.tight_layout
fig.set_size_inches((9,6))
ax_acc.set_xticklabels(range(0, 16001, 2000), visible=True)
ax_lat.set_xticklabels(range(0, 16001, 2000), visible=True)
# ax_lat.set_xticklabels(ax_lat.get_xticklabels(), visible=True)
plt.savefig("%s/endtoendeval.pdf" % fig_dir, bbox_inches='tight')
# plt.show()
