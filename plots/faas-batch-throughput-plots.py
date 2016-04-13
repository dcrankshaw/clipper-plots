import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil


# fname = "/Users/crankshaw/model-serving/centipede-plots/results/faas_benchmarks/spark_10rf.txt"
results_path = "../results"
# fig_dir = os.getcwd()
fig_dir = "/Users/crankshaw/Dropbox/Apps/ShareLaTeX/velox-centipede/osdi_2016/figs"
def parse_logs(fname):

    cur_batch = None
    batch_sizes = []
    p99_lat = []
    p99_err = []
    avg_lat = []
    max_lat = []
    avg_err = []
    thrus = []
    thrus_err = []
    cur_p99 = []
    cur_avg = []
    cur_thrus = []
    cur_max = []
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
                    max_lat.append(np.mean(cur_max))
                    avg_err.append(np.std(cur_avg)/np.sqrt(float(len(cur_avg))))
                    thrus.append(np.mean(cur_thrus))
                    thrus_err.append(np.std(cur_thrus)/np.sqrt(float(len(cur_thrus))))
                    
                cur_batch = int(l.split(":")[1].strip())
                cur_p99 = []
                cur_avg = []
                cur_thrus = []
                cur_max = []
                # print cur_batch
                i += 1
            elif l.strip() == "faas Metrics":
                trial_lines = lines[i:i+8]
                # print trial_lines
                lats = trial_lines[5].split(",")
                # print lats
                mean = float(lats[3].split(":")[1].strip()) / 1000.0
                max_l = float(lats[2].split(":")[1].strip()) / 1000.0
                # print max_l
                p99 = float(lats[6].split(":")[1].strip()) / 1000.0
                thru = float(trial_lines[7].split(",")[1].split(":")[1].strip())
                cur_avg.append(mean)
                cur_p99.append(p99)
                cur_thrus.append(thru)
                cur_max.append(max_l)
                
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
            max_lat.append(np.mean(cur_max))

    # print "\n\n\naaaaaaa"
    # print "MAX: ", max_lat
    # print "MEAN: ", avg_lat
    # print "bbbbbbb"

    return (batch_sizes,  p99_lat,  p99_err,  avg_lat,  avg_err,  thrus,  thrus_err, max_lat)




def plot_from_logs(filename, legend=False, ylim = 100):
    batch_size,  p99_lat,  p99_err,  avg_lat,  avg_err,  thrus,  thrus_err, max_lat = parse_logs(results_path + "/" + filename + ".txt")
    fig, ax = plt.subplots()
    # ax.set_xscale("log")
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
    # ax.set_xscale("log")
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

def parse_from_json(filename):
    res = json.load(open(filename,'r'))
    return res

def plot_batch_bar(ys, plot_fname, ylabel, ylim=None, p99=False):
    bar_color = 'steelblue'
    fig, ax = plt.subplots()
    ind = np.arange(3)
    h_width = 0.4
    rects1 = ax.bar(ind, ys, h_width*2, color=bar_color)

    labels = ['dynamic', 'mean', 'p99']
    ax.set_ylabel(ylabel)
    ax.set_xticks(ind + h_width)
    ax.set_xticklabels(labels)
    if ylim is not None:
        ax.set_ylim((0, ylim))

    def autolabel(rects):
    # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.3f' % height,
                ha='center', va='bottom')

    autolabel(rects1)
    fig.set_size_inches(5.0, 4.0)

    plt.savefig(plot_fname)
    shutil.copy(plot_fname, fig_dir)

def plot_dynamic_batch(dynamic_fname, static_fname, plot_fname):
    dyn_results = parse_logs(os.path.join(results_path,dynamic_fname))
    print dyn_results
    max_lat_threshhold_results = None
    mean_lat_threshhold_results = None
    if static_fname[-4:] == 'json':
        res = parse_from_json(os.path.join(results_path, static_fname))
        max_ind = 0
        max_lat = -1
        lat = 20
        for (i,v) in enumerate(res[u'max_latency']):
            if v < lat and v > max_lat:
                max_ind = i
                max_lat = v
        max_lat_threshhold_results = (res[u'mean_latency'][max_ind], res[u'std'][max_ind], res[u'throughput'][max_ind])
        mean_ind = 0
        mean_lat = -1
        for (i,v) in enumerate(res[u'mean_latency']):
            if v < lat and v > mean_lat:
                mean_ind = i
                mean_lat = v
        mean_lat_threshhold_results = (res[u'mean_latency'][mean_ind], res[u'std'][mean_ind], res[u'throughput'][mean_ind])
    elif static_fname[-3:] == 'txt':
        stat_results = parse_logs(os.path.join(results_path,static_fname))
        # print stat_results
        max_ind = 0
        max_lat = -1
        lat = 20
        # print stat_results[7]
        # this is really p99 not max
        for (i,v) in enumerate(stat_results[1]):
            if v < lat and v > max_lat:
                max_ind = i
                max_lat = v
        max_lat_threshhold_results = (stat_results[3][max_ind], 0.0, stat_results[5][max_ind])
        mean_ind = 0
        mean_lat = -1
        for (i,v) in enumerate(stat_results[3]):
            if v < lat and v > mean_lat:
                mean_ind = i
                mean_lat = v
        # mean_lat_threshhold_results = (res[u'mean_latency'][mean_ind], res[u'std'][mean_ind], res[u'throughput'][mean_ind])
        mean_lat_threshhold_results = (stat_results[3][mean_ind], 0.0, stat_results[5][mean_ind])
        

    # return (batch_sizes,  p99_lat,  p99_err,  avg_lat,  avg_err,  thrus,  thrus_err, max_lat)

    thru_ys = [dyn_results[5][0], mean_lat_threshhold_results[2],  max_lat_threshhold_results[2]]
    plot_batch_bar(thru_ys, "%s_thru.pdf" % plot_fname, "Throughput")
    lat_ys = [dyn_results[3][0], mean_lat_threshhold_results[0],  max_lat_threshhold_results[0]]
    plot_batch_bar(lat_ys, "%s_lat.pdf" % plot_fname, "Mean Latency", ylim = 23)

    # bar_color = 'steelblue'
    # fig, ax = plt.subplots()
    # ind = np.arange(3)
    # h_width = 0.4
    # rects1 = ax.bar(ind, [max_lat_threshhold_results[2], max_lat_threshhold_results[2], dyn_results[5][0]], h_width*2, color=bar_color)
    #
    # labels = ['dynamic', 'mean', 'max']
    # ax.set_ylabel('Throughput')
    # ax.set_xticks(ind + h_width)
    # ax.set_xticklabels(labels)
    #
    # def autolabel(rects):
    # # attach some text labels
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
    #             '%.3f' % height,
    #             ha='center', va='bottom')
    #
    # autolabel(rects1)
    # fig.set_size_inches(5.0, 4.0)
    #
    # plt.savefig('%s_thru.pdf' % plot_fname)

    # print dyn_results

# plot_from_logs('spark_10rf', ylim=100)
# plot_from_logs('spark_100rf', ylim=100)
# plot_from_logs('spark_lr', ylim=30)
# plot_from_logs('spark_svm', ylim=30)
# plot_from_json('tf_latency', legend=False, ylim=50)
# plot_from_json('sklearn_svm_local', legend=True, ylim=100)
# plot_dynamic_batch('sklearn_svm_dynamic_batch.txt', 'sklearn_svm_local.json', 'sklearn_dynamic_batch')
plot_dynamic_batch('spark_lr_dynamic_batch.txt', 'spark_lr.txt', 'spark_lr_dynamic_batch')
plot_dynamic_batch('spark_10rf_dynamic_batch.txt', 'spark_10rf.txt', 'spark_10rf_dynamic_batch')

