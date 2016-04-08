import json
import matplotlib.pyplot as plt
import numpy as np
import os


# fname = "/Users/crankshaw/model-serving/centipede-plots/results/faas_benchmarks/spark_10rf.txt"
results_path = "../results/faas_benchmarks"
# fig_dir = os.getcwd()
fig_dir = "/Users/crankshaw/Dropbox/Apps/ShareLaTeX/velox-centipede/osdi_2016/figs"
def parse_logs(fname):

    cur_batch = None
    batch_sizes = []
    p99_lat = []
    p99_err = []
    avg_lat = []
    avg_err = []
    thrus = []
    thrus_err = []
    cur_p99 = []
    cur_avg = []
    cur_thrus = []
    with open(fname, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
        # print len(lines)
        i = 0
        while i < len(lines):
            l = lines[i]
            if l.startswith("EXPERIMENT RUN BATCH SIZE"):
                # save previous batch
                if cur_batch is not None:
                    batch_sizes.append(cur_batch)
                    p99_lat.append(np.mean(cur_p99))
                    p99_err.append(np.std(cur_p99)/np.sqrt(float(len(cur_p99))))
                    avg_lat.append(np.mean(cur_avg))
                    avg_err.append(np.std(cur_avg)/np.sqrt(float(len(cur_avg))))
                    thrus.append(np.mean(cur_thrus))
                    thrus_err.append(np.std(cur_thrus)/np.sqrt(float(len(cur_thrus))))
                    
                cur_batch = int(l.split(":")[1].strip())
                cur_p99 = []
                cur_avg = []
                cur_thrus = []
                # print cur_batch
                i += 1
            elif l.strip() == "faas Metrics":
                trial_lines = lines[i:i+8]
    #             print trial_lines
                lats = trial_lines[5].split(",")
                mean = float(lats[3].split(":")[1].strip()) / 1000.0
                p99 = float(lats[6].split(":")[1].strip()) / 1000.0
                thru = float(trial_lines[7].split(",")[1].split(":")[1].strip())
                cur_avg.append(mean)
                cur_p99.append(p99)
                cur_thrus.append(thru)
                
                i += 8
            else:
                i += 1
                
        # save last batch        
        if cur_batch is not None:
            batch_sizes.append(cur_batch)
            p99_lat.append(np.mean(cur_p99))
            p99_err.append(np.std(cur_p99)/np.sqrt(float(len(cur_p99))))
            avg_lat.append(np.mean(cur_avg))
            avg_err.append(np.std(cur_avg)/np.sqrt(float(len(cur_avg))))
            thrus.append(np.mean(cur_thrus))
            thrus_err.append(np.std(cur_thrus)/np.sqrt(float(len(cur_thrus))))

    return (batch_sizes,  p99_lat,  p99_err,  avg_lat,  avg_err,  thrus,  thrus_err)




def plot_from_logs(filename, legend=False, ylim = 100):
    batch_size,  p99_lat,  p99_err,  avg_lat,  avg_err,  thrus,  thrus_err = parse_logs(results_path + "/" + filename + ".txt")
    fig, ax = plt.subplots()
    ax.errorbar(batch_size, avg_lat, yerr=avg_err, fmt='o-', label="mean latency")
    ax.errorbar(batch_size, p99_lat, yerr=p99_err, fmt='^-', label="max latency")
    # ax.plot(batch_size, res['max_latency'], '^--', label="max latency")
    ax.plot(batch_size, np.ones(len(batch_size))*20.0, "--", label='SLO')
    # ax.plot(batch_size, res['mean_latency'][0]*np.array(batch_size), label='linear scaling')
    ax2 = ax.twinx()
    ax2.errorbar(batch_size, thrus, yerr=thrus_err, fmt='ms-', label="throughput")
    ax2.set_ylabel('throughput(qps)')
    ax2.set_ylim((0, ax2.get_ylim()[1]*1.05))

    ax.set_ylim((0,ylim))
    ax.set_xlabel('batch size')
    ax.set_ylabel('latency(ms)')
    if legend:
        ax.legend(bbox_to_anchor=(1, 0.7),loc=5)
        ax2.legend(bbox_to_anchor=(1,0.54),loc=5, handlelength=3.2)
    print filename+'.pdf'
    fig.savefig(fig_dir + "/" + filename+'.pdf',bbox_inches='tight')



def plot_from_json(filename, legend=False, ylim = 100):
    res = json.load(open('../results/'+filename+'.json','r'))
    fig, ax = plt.subplots()
    batch_size = res['batch_size']
    ax.plot(batch_size, res['mean_latency'], 'o-', label="mean latency")
    ax.plot(batch_size, res['max_latency'], '^-', label="max latency")
    ax.plot(batch_size, res['slo'], '--', label='SLO')
    # ax.plot(batch_size, res['mean_latency'][0]*np.array(batch_size), label='linear scaling')
    ax2 = ax.twinx()
    ax2.plot(batch_size, res['throughput'],'s-', color='m', label="throughput")
    ax2.set_ylabel('throughput(qps)')
    ax2.set_ylim((0, ax2.get_ylim()[1]*1.05))

    ax.set_ylim((0,ylim))
    ax.set_xlabel('batch size')
    ax.set_ylabel('latency(ms)')
    if legend:
        ax.legend(bbox_to_anchor=(1, 0.7),loc=5)
        ax2.legend(bbox_to_anchor=(1,0.54),loc=5, handlelength=3.2)
    print filename+'.pdf'
    fig.savefig(fig_dir + "/" + filename+'.pdf',bbox_inches='tight')
    # plt.show()

plot_from_logs('spark_10rf', ylim=100)
plot_from_logs('spark_100rf', ylim=100)
plot_from_logs('spark_lr', ylim=30)
plot_from_logs('spark_svm', ylim=30)
plot_from_json('tf_latency', legend=False, ylim=50)
plot_from_json('sklearn_svm_local', legend=True, ylim=100)
