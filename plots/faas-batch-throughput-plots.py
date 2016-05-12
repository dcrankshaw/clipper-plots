import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil


# fname = "/Users/crankshaw/model-serving/centipede-plots/results/faas_benchmarks/spark_10rf.txt"
results_path = "../results"
# fig_dir = os.getcwd()
fig_dir = "/Users/crankshaw/ModelServingPaper/osdi_2016/figs"
matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['font.size'] = 10
nbins=5
width=2.5
height=1.3*2
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




def plot_from_logs(ax, filename, legend=False, ylim = 100, thru_label=False):
    batch_size,  p99_lat,  p99_err,  avg_lat,  avg_err,  thrus,  thrus_err, max_lat = parse_logs(results_path + "/" + filename + ".txt")
    print np.min(thrus), np.max(thrus)
    print batch_size[4], avg_lat[4]
    # fig, ax = plt.subplots()
    # ax.locator_params(nbins=nbins)
    ax.locator_params(tight=True, nbins=nbins)
    # ax.set_xscale("log")
    ax.errorbar(batch_size, avg_lat, yerr=avg_err, fmt='o-', label="mean latency")
    ax.errorbar(batch_size, p99_lat, yerr=p99_err, fmt='^-', label="p99 latency")
    # ax.plot(batch_size, res['max_latency'], '^--', label="max latency")
    ax.plot(batch_size, np.ones(len(batch_size))*20.0, "--", label='SLO')
    # ax.plot(batch_size, res['mean_latency'][0]*np.array(batch_size), label='linear scaling')
    ax2 = ax.twinx()
    ax2.locator_params(tight=True, nbins=nbins)
    ax2.errorbar(batch_size, thrus, yerr=thrus_err, fmt='ms-', label="throughput")
    ax2.set_ylim((0, ax2.get_ylim()[1]*1.1))
    ax.set_xlim((-5, ax.get_xlim()[1]*1.05))
    ax2.set_xlim((-5, ax2.get_xlim()[1]*1.05))

    ax.set_ylim((0,ylim))
    if thru_label:
        ax2.set_ylabel('Throughput(qps)')
    if legend:
        ax2.set_ylabel('throughput(qps)')
        ax.set_xlabel('batch size')
        ax.set_ylabel('latency(ms)')
        ax.legend(bbox_to_anchor=(1, 0.7),loc=5)
        ax2.legend(bbox_to_anchor=(1,0.54),loc=5, handlelength=3.2)
    # print fig_dir + "/" + filename+'.pdf'
    # fig.set_size_inches(width, height)
    # fig.savefig(fig_dir + "/" + filename+'.pdf',bbox_inches='tight')



def plot_from_json(ax, filename, legend=False, ylim = 100, thru_label=False):
    res = json.load(open('../results/'+filename+'.json','r'))
    # fig, ax = plt.subplots()

    ax.locator_params(tight=True, nbins=nbins)
    # ax.set_xscale("log")
    batch_size = res['batch_size']
    ax.plot(batch_size, res['mean_latency'], 'o-', label="mean latency")
    ax.plot(batch_size, res['max_latency'], '^-', label="p99 latency")
    ax.plot(batch_size, res['slo'], '--', label='SLO')
    # ax.plot(batch_size, res['mean_latency'][0]*np.array(batch_size), label='linear scaling')
    ax2 = ax.twinx()
    ax2.locator_params(tight=True,nbins=nbins)
    ax2.plot(batch_size, res['throughput'],'s-', color='m', label="throughput")
    ax2.set_ylim((0, ax2.get_ylim()[1]*1.05))
    ax.set_xlim((-5, ax.get_xlim()[1]*1.05))
    ax2.set_xlim((-5, ax2.get_xlim()[1]*1.05))

    ax.set_ylim((0,ylim))
    if thru_label:
        ax2.set_ylabel('Throughput(qps)')

    # ax.set_xlabel('batch size')
    if legend:
        # ax.set_ylabel('latency(ms)')
        # ax2.set_ylabel('throughput(qps)')
        ax.legend(bbox_to_anchor=(1.0, 0.60),loc=5,handlelength=3.2,fontsize=9)
        ax2.legend(bbox_to_anchor=(1.0,0.23),loc=5, handlelength=4.1,fontsize=9)
    # print filename+'.pdf'
    # fig.set_size_inches(width,height)
    # fig.savefig(fig_dir + "/" + filename+'.pdf',bbox_inches='tight')
    # plt.show()

def parse_from_json(filename):
    res = json.load(open(filename,'r'))
    return res

def plot_batch_bar(ax, ys, plot_fname, ylabel=None, ylim=None, p99=False):

    width=4
    height=1.5
    matplotlib.rcParams['font.size'] = 18
    # bar_color = 'steelblue'
    plt.set_cmap('Greys')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.locator_params(tight=True, nbins=nbins)
    ind = np.arange(3) + 0.2
    h_width = 0.35
    # rects1 = ax.bar(ind, ys[:-1], h_width*2, color=bar_color)
    rects1 = ax.bar(ind, ys, h_width*2)
    colors = ['#333333', '#666666', '#999999']
    for i in range(len(rects1)):
        rects1[i].set_color(colors[i])

    labels = ['adaptive', 'optimal', 'no batch']
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xticks(ind + h_width)
    ax.set_xticklabels(labels)
    if ylim is not None:
        ax.set_ylim((0, ylim))
    else:
        ylim = ax.get_ylim()[1]
        ax.set_ylim((0, ylim*1.3))

    def autolabel(rects):
    # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.1f' % height,
                ha='center', va='bottom')

    autolabel(rects1)
    # fig.set_size_inches(width,height)
    #
    # print plot_fname
    # plt.savefig(fig_dir + "/" + plot_fname, bbox_inches='tight')
    # shutil.copy(plot_fname, fig_dir)

def plot_dynamic_batch(dynamic_fname, static_fname, plot_fname, title):

    matplotlib.rcParams['font.size'] = 10
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
        min_lat_threshhold_results = (res[u'mean_latency'][0], res[u'std'][0], res[u'throughput'][0])
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
        min_lat_threshhold_results = (stat_results[3][0], 0.0, stat_results[5][0])
        

    # return (batch_sizes,  p99_lat,  p99_err,  avg_lat,  avg_err,  thrus,  thrus_err, max_lat)

    thru_ys = [dyn_results[5][0], mean_lat_threshhold_results[2],  min_lat_threshhold_results[2]]
    # plot_batch_bar(thru_ys, "%s_thru.pdf" % plot_fname, "Throughput")

    fig, (axt, axl) = plt.subplots(2,1,sharex=False)
    plot_batch_bar(axt, thru_ys, "%s_thru.pdf" % plot_fname, ylabel="Throughput (qps)")
    lat_ys = [dyn_results[3][0], mean_lat_threshhold_results[0],  min_lat_threshhold_results[0]]
    plot_batch_bar(axl, lat_ys, "%s_lat.pdf" % plot_fname, ylim = 26, ylabel="Latency (ms)")
    # axt.set_title(title)

    width=4.0
    height=4.0
    fig.set_size_inches(width,height)
    # plot_fname = "
    print plot_fname
    plt.savefig(fig_dir + "/" + plot_fname + ".pdf", bbox_inches='tight')

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

def plot_fig3():
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
    fig.subplots_adjust(hspace=0.4)
    fig.subplots_adjust(wspace=0.24)
    # fig, ax = plt.subplots(3,2)
    # print ax
    # plot_from_logs('spark_10rf', ylim=100)
    plot_from_json(ax1, 'sklearn_svm_local', legend=True, ylim=100)
    ax1.set_title("a) Scikit-Learn SVM")
    plot_from_logs(ax2, 'spark_svm', ylim=40)
    ax2.set_title("b) PySpark SVM")
    plot_from_logs(ax3,'spark_lr', ylim=40, thru_label=True)
    ax3.set_title("c) PySpark LR")
    plot_from_logs(ax4, 'spark_100rf', ylim=100)
    ax4.set_title("d) PySpark 100 RF")
    plot_from_logs(ax5, 'spark_10rf', ylim=100)
    ax5.set_title("e) PySpark 10 RF")
    plot_from_json(ax6, 'tf_latency', legend=False, ylim=50, thru_label=True)
    ax6.set_title("f) TF ConvNet GPU")

    ax1.set_ylabel('Latency(ms)')
    ax4.set_ylabel('Latency(ms)')

    ax4.set_xlabel('batch size')
    ax5.set_xlabel('batch size')
    ax6.set_xlabel('batch size')

    # ax2.set_ylabel('throughput(qps)')
    # plot_from_json(ax6, 'sklearn_svm_local', legend=True, ylim=100)

    
    filename = "faas-thruputs.pdf"
    # print fig_dir + "/" + filename+'.pdf'

    width=12
    height=3
    # plt.tight_layout()
    fig.set_size_inches(width, height)
    fig.savefig(fig_dir + "/" + filename,bbox_inches='tight')

# plot_dynamic_batch('sklearn_svm_dynamic_batch.txt', 'sklearn_svm_local.json', 'sklearn_dynamic_batch', "a) Scikit-Learn SVM")
# plot_dynamic_batch('spark_lr_dynamic_batch.txt', 'spark_lr.txt', 'spark_lr_dynamic_batch', "b) PySpark LR")

if __name__=='__main__':
    plot_fig3()









# plot_dynamic_batch('spark_10rf_dynamic_batch.txt', 'spark_10rf.txt', 'spark_10rf_dynamic_batch')
#
