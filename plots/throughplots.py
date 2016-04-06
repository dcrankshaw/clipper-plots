import json
import matplotlib.pyplot as plt
import numpy as np


def plot(filename, legend=False, ylim = 100):
    res = json.load(open('../results/'+filename+'.json','r'))
    fig, ax = plt.subplots()
    batch_size = res['batch_size']
    ax.plot(batch_size, res['mean_latency'], 'o-', label="mean latency")
    ax.plot(batch_size, res['max_latency'], '^--', label="max latency")
    ax.plot(batch_size, res['slo'], label='SLO')
    ax.plot(batch_size, res['mean_latency'][0]*np.array(batch_size), label='linear scaling')
    ax2 = ax.twinx()
    ax2.plot(batch_size, res['throughput'],'o-', color='m', label="throughput")
    ax2.set_ylabel('throughput(qps)')

    ax.set_ylim((0,ylim))
    ax.set_xlabel('batch size')
    ax.set_ylabel('latency(ms)')
    if legend:
        ax.legend(bbox_to_anchor=(1, 0.7),loc=5)
        ax2.legend(bbox_to_anchor=(1,0.54),loc=5, handlelength=3.2)
    fig.savefig(filename+'.pdf',bbox_inches='tight')
    plt.show()

plot('tf_latency',legend=True,ylim=100)
plot('pyspark_10rf_local',ylim=400)
plot('pyspark_100rf_local',ylim=450)
plot('pyspark_lr_local',ylim=40)
plot('pyspark_svm_local',ylim=40)
plot('sklearn_svm_local',ylim=600)