import numpy as np
import matplotlib.pyplot as plt
import time
import json

# fig_dir = '/Users/crankshaw/Dropbox/sharelatex/ModelServingPaper/figs/gen-'
fig_dir = 'gen_figs/gen-'
bar_color = 'steelblue'
formats = ['bo', 'r^', 'gv', 'k*']


def cache_lookup_latency():

  res_path = 'cache_latency_dist.json'
  with open(res_path, 'r') as fp:
      data = json.load(fp)

  res_for_hist = data['res_for_hist']
  fig, ax = plt.subplots()


  ax.hist(res_for_hist, bins=100, color=bar_color)


  ax.set_xlabel("Cache latency ($\mu$s)")
  # default_size = fig.get_size_inches()
  # size_mult = 0.7
  fig.set_size_inches(5.0, 3.0)
  # fig.set_size_inches(default_size[0]*size_mult*1.6,default_size[1]*size_mult)
  plt.gcf().subplots_adjust(bottom=0.25)
  plt.savefig('%scache_lookup_latency.pdf' % fig_dir)

def wiki_cache_hit_rate():

  article_ch_rate = 0.974668141978
  pred_ch_rate = 0.494301293664
  shared_ch_rate = 0.987334070989

  fig, ax = plt.subplots()
  ind = np.arange(3)
  h_width = 0.4
  rects1 = ax.bar(ind, [pred_ch_rate, article_ch_rate, shared_ch_rate], h_width*2, color=bar_color)

  labels = ['pred', 'features', 'shared']
  ax.set_ylabel('Cache hit rate')
  ax.set_xticks(ind + h_width)
  ax.set_xticklabels(labels)

  def autolabel(rects):
      # attach some text labels
      for rect in rects:
          height = rect.get_height()
          ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                  '%.3f' % height,
                  ha='center', va='bottom')
          
  autolabel(rects1)
  fig.set_size_inches(5.0, 4.0)

  plt.savefig('%scache_hit_rate_wiki.pdf' % fig_dir)

def retrain_latency():


  res_path = 'retrain_results.json'
  with open(res_path, 'r') as fp:
      data = json.load(fp)
  init_points = 20
  max_points = 50
  points = np.arange(init_points, max_points + 1)

  # retrain_latency_results = {'all_train_times': all_train_times, 'w_train_times': w_train_times}
  fig, ax = plt.subplots()
  ms = 6
  ax.plot(points, np.array(data['w_train_times'])*1000.0, '%s-' % formats[0], markersize=ms, label='tasks')
  ax.plot(points, np.array(data['all_train_times'])*1000.0, '%s--' % formats[1], markersize=ms, label='features + tasks')
  ax.set_yscale('log')
  ax.set_xlabel('train points per task')
  ax.set_ylabel('retrain latency (ms)')
  ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.3)
  ax.legend(loc=0)
  # fig.set_size_inches(6.0, 3.5)
  fig.set_size_inches(4.0, 3.0)
  plt.gcf().subplots_adjust(bottom=0.2, left=0.2)

  plt.savefig('%sretrain_latency.pdf' % fig_dir)


def anytime_predictions():
  # data = {'noise_params': noise_params, 'p99_lats': p99_lats, 'accuracies': accuracies }
  res_path = 'anytime-preds.json'
  with open(res_path, 'r') as fp:
      data = json.load(fp)

  fig, (ax_lat, ax_acc) = plt.subplots(2, sharex=True)
  i = 0
  line_styles = ['r:', 'b-', 'g-.', 'k--']
  for s in ['-1', '10', '50', '100']:
      lab = s
      if s == '-1':
          lab = 'No SLA' 
      # ax_lat.plot(data['noise_params'], data['p99_lats'][s], '%s-' % formats[i], label=lab)
      # ax_acc.plot(data['noise_params'], data['accuracies'][s], '%s-' % formats[i], label=lab)
      ax_lat.plot(data['noise_params'], data['p99_lats'][s], line_styles[i], label=lab)
      ax_acc.plot(data['noise_params'], data['accuracies'][s], line_styles[i], label=lab)
      i += 1

  axes = (ax_lat, ax_acc)
  for ax in axes:
      ax.set_ylim(0, ax.get_ylim()[1] + 0.1)
  ax_lat.legend(loc=0)
  ax_lat.set_ylabel("Latency")
  ax_acc.set_ylabel("Error")
  ax_acc.set_xlabel("Noise Param")
  fig.set_size_inches(5.0, 8.0)
  plt.gcf().subplots_adjust(left=0.15)
  plt.savefig('%sanytime-preds.pdf' % fig_dir)


def feature_eval_latency():

  with open("runtimes2.json", 'r') as f:
    data = json.load(f)

  ntrials = data['ntrials']
  times = data['times']
  labels = data['labels']
  predictions = data['predictions']

  def info_str(x):
    avg = np.round(np.mean(x) * 1000, 2)
    p99 = np.round(np.percentile(x, 99) * 1000, 2)
    return "(avg = " + str(avg) + ", p99 = " + str(p99) + ")"

  bins = 100
  fig = plt.figure()
  ax4 = fig.add_subplot(414)
  ax1 = fig.add_subplot(411, sharex=ax4)
  ax2 = fig.add_subplot(412, sharex=ax4)
  ax3 = fig.add_subplot(413, sharex=ax4)


  ax1.set_title("10 Trees " + info_str(times['rf10.pipeline']), y=.5)
  ax2.set_title("100 Trees " + info_str(times['rf100.pipeline']) , y=.5)
  ax3.set_title("500 Trees\n" + info_str(times['rf500.pipeline']), x=0.6, y=.5)
  ax4.set_title("DNN\n" + info_str(times['imagenet']), x=0.3, y=0.5)
  ax4.set_xlabel("Latency (ms)")

  ax1.hist(np.array(times['rf10.pipeline']) * 1000.0, bins, color=bar_color)
  ax2.hist(np.array(times['rf100.pipeline']) * 1000.0, bins, color=bar_color)
  ax3.hist(np.array(times['rf500.pipeline']) * 1000.0, bins, color=bar_color)
  ax4.hist(np.array(times['imagenet']) * 1000.0, bins, color=bar_color)

  fig.set_size_inches(9.0, 8.0)
  # fig.savefig("latency_of_mllib_tree_pipelines.pdf")
  plt.savefig('%smllib-tree-model-eval-lat.pdf' % fig_dir)


  bins = 100
  fig = plt.figure()
  ax4 = fig.add_subplot(414)
  ax1 = fig.add_subplot(411, sharex=ax4)
  ax2 = fig.add_subplot(412, sharex=ax4)
  ax3 = fig.add_subplot(413, sharex=ax4)



  ax1.set_title("NOP " + info_str(times['nop']), y=.5)
  ax2.set_title("Single LR " + info_str(times['lrModel8.pipeline']), y=.5)
  ax3.set_title("Decision Tree " + info_str(times['singleDT.pipeline']), y=.5)
  ax4.set_title("One-vs-All LR\n" + info_str(times['ovrLR.pipeline']), y=.5)

  ax4.set_xlabel("Latency (ms)")

  ax1.hist(np.array(times['nop']) * 1000.0, bins, color=bar_color)
  ax2.hist(np.array(times['lrModel8.pipeline']) * 1000.0, bins, color=bar_color)
  ax3.hist(np.array(times['singleDT.pipeline']) * 1000.0, bins, color=bar_color)
  ax4.hist(np.array(times['ovrLR.pipeline']) * 1000.0, bins, color=bar_color)

  fig.set_size_inches(9.0, 8.0)

  # fig.savefig("latency_of_mllib_various_pipelines.pdf")
  plt.savefig('%smllib-pipeline-eval-lat.pdf' % fig_dir)


def feature_eval_latency_poster():

  with open("runtimes2.json", 'r') as f:
    data = json.load(f)

  ntrials = data['ntrials']
  times = data['times']
  labels = data['labels']
  predictions = data['predictions']

  def info_str(x):
    avg = np.round(np.mean(x) * 1000, 2)
    p99 = np.round(np.percentile(x, 99) * 1000, 2)
    return "(avg = " + str(avg) + ", p99 = " + str(p99) + ")"

  bins = 100
  fig = plt.figure()
  # ax4 = fig.add_subplot(414)
  ax3 = fig.add_subplot(313)
  ax1 = fig.add_subplot(311, sharex=ax3)
  ax2 = fig.add_subplot(312, sharex=ax3)


  ax1.set_title("10 Trees " + info_str(times['rf10.pipeline']), y=.5)
  ax2.set_title("100 Trees " + info_str(times['rf100.pipeline']) , y=.5)
  ax3.set_title("500 Trees\n" + info_str(times['rf500.pipeline']), x=0.6, y=.5)
  # ax4.set_title("DNN\n" + info_str(times['imagenet']), x=0.3, y=0.5)
  # ax4.set_xlabel("Latency (ms)")

  ax1.hist(np.array(times['rf10.pipeline']) * 1000.0, bins, color=bar_color)
  ax2.hist(np.array(times['rf100.pipeline']) * 1000.0, bins, color=bar_color)
  ax3.hist(np.array(times['rf500.pipeline']) * 1000.0, bins, color=bar_color)
  # ax4.hist(np.array(times['imagenet']) * 1000.0, bins, color=bar_color)

  fig.set_size_inches(7.0, 7.0)
  # fig.savefig("latency_of_mllib_tree_pipelines.pdf")
  plt.savefig('%smllib-tree-model-eval-lat-no-dnn.pdf' % fig_dir)


  # bins = 100
  # fig = plt.figure()
  # ax4 = fig.add_subplot(414)
  # ax1 = fig.add_subplot(411, sharex=ax4)
  # ax2 = fig.add_subplot(412, sharex=ax4)
  # ax3 = fig.add_subplot(413, sharex=ax4)
  #
  #
  #
  # ax1.set_title("NOP " + info_str(times['nop']), y=.5)
  # ax2.set_title("Single LR " + info_str(times['lrModel8.pipeline']), y=.5)
  # ax3.set_title("Decision Tree " + info_str(times['singleDT.pipeline']), y=.5)
  # ax4.set_title("One-vs-All LR\n" + info_str(times['ovrLR.pipeline']), y=.5)
  #
  # ax4.set_xlabel("Latency (ms)")
  #
  # ax1.hist(np.array(times['nop']) * 1000.0, bins, color=bar_color)
  # ax2.hist(np.array(times['lrModel8.pipeline']) * 1000.0, bins, color=bar_color)
  # ax3.hist(np.array(times['singleDT.pipeline']) * 1000.0, bins, color=bar_color)
  # ax4.hist(np.array(times['ovrLR.pipeline']) * 1000.0, bins, color=bar_color)
  #
  # fig.set_size_inches(9.0, 8.0)
  #
  # # fig.savefig("latency_of_mllib_various_pipelines.pdf")
  # plt.savefig('%smllib-pipeline-eval-lat.pdf' % fig_dir)
  #


def pred_dot_lats():

  res_path = 'pred_times.json'
  with open(res_path, 'r') as fp:
      data = json.load(fp)

  times = data['times']
  vec_sizes = data['vec_sizes']
  # retrain_latency_results = {'all_train_times': all_train_times, 'w_train_times': w_train_times}
  fig, ax = plt.subplots()
  ms = 2
  ax.scatter(vec_sizes, times, s=10, c = 'b', marker='o')
  ax.set_xlabel('dimension of ensemble')
  ax.set_ylabel('prediction latency ($\mu s$)')
  ax.set_ylim(-1, 100)
  # ax.legend(loc=0)
  # fig.set_size_inches(6.0, 3.5)
  ax.set_xlim(0, np.max(np.array(vec_sizes)) + 100)
  fig.set_size_inches(5.0, 3.0)
  plt.locator_params(nbins=3)
  plt.gcf().subplots_adjust(bottom=0.20, left=0.17)

  plt.savefig('%spred_latency.pdf' % fig_dir)


def digits_train_fs():

  res_path = 'digits_train_fs.json'
  with open(res_path, 'r') as fp:
    data = json.load(fp)

  iters = range(20,71)
  fig,ax = plt.subplots()
  ms = 6
  ax.plot(iters, data['digit_no_retrain_fs'], '%s-' % formats[0], markersize=ms, label='tasks', )
  ax.plot(iters, data['digit_retrain_fs'], '%s--' % formats[1], markersize=ms, label='features + tasks')
  # ax.legend(loc=0)
  ax.set_xlabel('train points per task')
  ax.set_ylabel('Error rate')
  # ax.set_title('MNIST Data')
  # default_size = fig.get_size_inches()
  # size_mult = 1.7
  ax.set_ylim((0.0, ax.get_ylim()[1]))
  fig.set_size_inches(4.0, 3.0)
  plt.gcf().subplots_adjust(bottom=0.20, left=0.20)
  plt.savefig('%sdigits-train-fs.pdf' % fig_dir)

# def newsgroups_train_fs():


def concept_drift():

  ms = 3
  u = "0.1095625	0.1095625	0.108	0.1075625	0.10675	0.1061875	0.1071875	0.105625	0.1055	0.104125	0.1024375	0.1023125	0.1018125	0.1015625	0.102875	0.10325	0.103375	0.1021875	0.1019375	0.100375	0.0988125	0.5461	0.50355	0.4755	0.44955	0.426675	0.41245	0.398025	0.38315	0.37295	0.3612	0.351875	0.3418	0.33385	0.325825	0.320975	0.31345	0.30565	0.2984	0.291825	0.286375	0.279525	0.275675	0.271075	0.267225	0.261425	0.256775	0.253	0.24795	0.2448	0.24025	0.2369	0.2334	0.22885	0.22655	0.22315	0.22075	0.217525	0.215175	0.213175	0.2102	0.2078	0.20545	0.203625	0.202075	0.1996	0.19755	0.195975	0.193625	0.19165	0.1902	0.188775	0.186625	0.18485	0.183975	0.1825	0.181175	0.1794	0.177775	0.176425	0.174775	0.174075	0.1725	0.172025	0.170875	0.169925	0.169275	0.167875	0.166825	0.1655	0.164425	0.163625	0.16245	0.1613	0.160025	0.15925	0.158075	0.157325	0.15665	0.155625	0.1552	0.154175	0.1529	0.15205	0.151025	0.15065	0.14935	0.1488	0.1483	0.147975	0.147025	0.145725	0.14535	0.144625	0.14435	0.1439	0.143225	0.142325	0.142	0.141775	0.141375"
  digit_train_all = [float(i) for i in u.split('\t')]
  v = "0.1095625	0.1095625	0.1111875	0.1139375	0.1138125	0.1149375	0.11725	0.1160625	0.1185	0.123125	0.1200625	0.121375	0.12275	0.123125	0.1255	0.1269375	0.127375	0.1278125	0.1295	0.1290625	0.13025	0.543625	0.48275	0.446275	0.408175	0.3901	0.3719	0.353	0.331975	0.314075	0.300575	0.29595	0.281375	0.267475	0.25885	0.245725	0.233075	0.222475	0.20895	0.18925	0.1624	0.1297	0.13425	0.13165	0.1312	0.13255	0.1339	0.132375	0.131575	0.13545	0.1335	0.13305	0.133225	0.1342	0.133375	0.13215	0.130725	0.127625	0.1301	0.13	0.129175	0.12965	0.12825	0.128575	0.126425	0.1274	0.127625	0.1293	0.130825	0.1296	0.13025	0.13405	0.13315	0.13275	0.133075	0.13055	0.132075	0.13155	0.131225	0.130425	0.133475	0.131575	0.126975	0.12655	0.1264	0.127525	0.128425	0.12835	0.12575	0.1272	0.127325	0.127825	0.126525	0.1296	0.128325	0.129475	0.127525	0.1267	0.1252	0.123025	0.12175	0.122975	0.12615	0.128525	0.1256	0.127	0.130425	0.13025	0.129725	0.129975	0.128725	0.127375	0.1282	0.129625	0.128475	0.1273	0.12985	0.1302	0.12885	0.13005	0.12965"
  digit_retrain_new = [float(i) for i in v.split('\t')]
  w = "0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955"
  digit_no_retrain = [float(i) for i in w.split('\t')]

  iters = range(20, 20+len(digit_train_all))
  fig,ax = plt.subplots()
  ax.plot(iters, digit_train_all, '%s-' % formats[0], markersize=ms, label='all')
  ax.plot(iters, digit_retrain_new, '%s-' % formats[1], markersize=ms, label = 'window')
  ax.plot(iters, digit_no_retrain, '%s-' % formats[2], markersize=ms, label = 'none')
  ax.set_xlabel('Num Examples')
  ax.set_ylabel('Prediction Error')
  # ax.set_title('Total Concept Drift - MNIST Data')
  # ax.legend(loc=0)

  fig.set_size_inches(4.0, 3.0)
  ax.set_ylim((0.0, ax.get_ylim()[1]))
  # plt.gcf().subplots_adjust(bottom=0.15, left=0.12)
  plt.gcf().subplots_adjust(bottom=0.20, left=0.20)
  plt.savefig('%sconcept-drift-total.pdf' % fig_dir)


  #### digit, partial concept drift
  o = "0.1095625	0.1095625	0.1070625	0.108	0.1065625	0.106125	0.1050625	0.1045	0.1038125	0.102875	0.10425	0.102875	0.103	0.1016875	0.1006875	0.1005625	0.1025	0.1006875	0.10175	0.100625	0.0995625	0.301175	0.2898	0.281625	0.27815	0.273125	0.270275	0.264225	0.25935	0.257075	0.2542	0.250725	0.246975	0.2433	0.23905	0.2359	0.23365	0.231275	0.2286	0.226375	0.223975	0.220275	0.21685	0.21485	0.212	0.210875	0.207725	0.206	0.204	0.202175	0.1999	0.1983	0.196175	0.1955	0.19525	0.1936	0.190525	0.19025	0.188325	0.187525	0.185425	0.18435	0.182525	0.180925	0.180525	0.179025	0.178225	0.1772	0.176275	0.175875	0.175275	0.1745	0.173325	0.172825	0.172175	0.170925	0.170175	0.1695	0.16865	0.168725	0.16675	0.166	0.1652	0.1653	0.16495	0.164775	0.163975	0.16315	0.162325	0.161575	0.160625	0.1602	0.16015	0.1596	0.15895	0.15875	0.158175	0.157725	0.1576	0.1577	0.156725	0.1565	0.155975	0.1555	0.15495	0.15485	0.154675	0.154125	0.153725	0.153275	0.15285	0.152475	0.15255	0.151725	0.1517	0.151625	0.151125	0.151	0.15055	0.150475	0.1502"
  digit_train_all = [float(i) for i in o.split('\t')]
  p = "0.1095625	0.1095625	0.1093125	0.1113125	0.1124375	0.11425	0.1140625	0.1146875	0.11775	0.1190625	0.1210625	0.121	0.120875	0.123	0.122	0.12225	0.1245	0.1246875	0.12575	0.125375	0.12525	0.311675	0.2997	0.289375	0.28445	0.278775	0.2768	0.268	0.264225	0.258525	0.25405	0.2504	0.2454	0.2436	0.234975	0.2312	0.2298	0.229325	0.23085	0.228775	0.22825	0.2253	0.2227	0.22555	0.2216	0.22045	0.21785	0.2165	0.2183	0.220425	0.21995	0.220075	0.2213	0.22005	0.218375	0.218325	0.215025	0.218875	0.2187	0.217525	0.215425	0.2181	0.218775	0.22245	0.22395	0.2235	0.224825	0.221975	0.221525	0.22085	0.2255	0.2255	0.224725	0.2239	0.222025	0.22195	0.224875	0.224925	0.225525	0.2268	0.2267	0.2234	0.22225	0.22095	0.223325	0.220075	0.22195	0.223525	0.22425	0.221075	0.217975	0.218475	0.22135	0.220725	0.221825	0.221425	0.222325	0.2198	0.220575	0.221575	0.222	0.2232	0.224375	0.2241	0.2238	0.225075	0.227825	0.226125	0.2259	0.227325	0.2258	0.2275	0.225975	0.22555	0.228725	0.22935	0.2277	0.2277	0.228625	0.23025	0.229375"
  digit_retrain_new = [float(i) for i in p.split('\t')]
  q = "0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675"
  digit_no_retrain = [float(i) for i in q.split('\t')]

  iters = range(20, 20+len(digit_train_all))
  fig,ax = plt.subplots()
  ax.plot(iters, digit_train_all, '%s-' % formats[0], markersize=ms, label='all')
  ax.plot(iters, digit_retrain_new, '%s-' % formats[1], markersize=ms, label = 'window')
  ax.plot(iters, digit_no_retrain, '%s-' % formats[2], markersize=ms, label = 'none')
  ax.set_xlabel('Num Examples')
  ax.set_ylabel('Prediction Error')
  # ax.legend(loc=0)

  fig.set_size_inches(4.0, 3.0)
  ax.set_ylim((0.0, ax.get_ylim()[1]))
  plt.gcf().subplots_adjust(bottom=0.20, left=0.20)
  plt.savefig('%sconcept-drift-partial.pdf' % fig_dir)

  #### digit, no concept drift
  r = "0.105625	0.105625	0.10475	0.1038125	0.1060625	0.1074375	0.1050625	0.105875	0.1045625	0.1054375	0.10375	0.104875	0.104125	0.1036875	0.101375	0.102125	0.1025	0.1016875	0.1015	0.1025625	0.1005	0.104925	0.104075	0.10395	0.104425	0.1049	0.103825	0.10365	0.10305	0.1026	0.10215	0.102425	0.101875	0.101625	0.102125	0.102175	0.1017	0.100425	0.100725	0.099925	0.09955	0.099475	0.099175	0.09865	0.09865	0.098575	0.098025	0.0983	0.09785	0.0981	0.097825	0.09765	0.097325	0.096725	0.0963	0.095575	0.095025	0.093975	0.093175	0.09225	0.09255	0.092875	0.092875	0.0922	0.091575	0.091625	0.0913	0.090725	0.09065	0.090625	0.090225	0.08995	0.089875	0.089525	0.089375	0.08915	0.088825	0.088625	0.088175	0.087775	0.087225	0.086875	0.086975	0.087275	0.0873	0.08715	0.0871	0.086925	0.0863	0.08615	0.086575	0.0865	0.086275	0.086225	0.08625	0.086025	0.086125	0.086225	0.08605	0.086475	0.085875	0.08565	0.0855	0.085675	0.085775	0.0859	0.08575	0.0854	0.084975	0.085025	0.085225	0.085075	0.0849	0.0846	0.08445	0.084425	0.084625	0.084875	0.084575	0.084075	0.083875"
  digit_train_all = [float(i) for i in r.split('\t')]
  s = "0.105625	0.105625	0.107	0.11025	0.110375	0.112125	0.1111875	0.1131875	0.1129375	0.1165625	0.1175	0.117875	0.1174375	0.1196875	0.1213125	0.1254375	0.1254375	0.12675	0.1278125	0.128625	0.1301875	0.13185	0.132275	0.13365	0.133775	0.132825	0.134225	0.133175	0.133275	0.1344	0.1351	0.1342	0.135325	0.1353	0.13555	0.13335	0.13115	0.132525	0.1328	0.135425	0.136425	0.1352	0.137	0.13655	0.135825	0.13435	0.134725	0.1365	0.14055	0.139125	0.140775	0.138525	0.140225	0.1395	0.139625	0.142	0.1416	0.1379	0.13675	0.13565	0.136425	0.138375	0.13765	0.138325	0.138075	0.13735	0.135	0.136225	0.136325	0.1385	0.137225	0.133975	0.133575	0.131825	0.13225	0.133625	0.135125	0.136125	0.13595	0.13625	0.13565	0.13395	0.132	0.13115	0.131475	0.13485	0.136825	0.136925	0.136425	0.135675	0.136075	0.136775	0.13565	0.13535	0.134875	0.135975	0.13665	0.138825	0.13585	0.13475	0.137025	0.137875	0.138225	0.138025	0.139675	0.138575	0.138175	0.135525	0.136575	0.135925	0.135775	0.13855	0.137775	0.138775	0.1369	0.1342	0.1344	0.1331	0.13055	0.132175	0.131875"
  digit_retrain_new = [float(i) for i in s.split('\t')]
  t = "0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625	0.105625"
  digit_no_retrain = [float(i) for i in t.split('\t')]

  iters = range(20, 20+len(digit_train_all))
  fig,ax = plt.subplots()
  ax.plot(iters, digit_train_all, '%s-' % formats[0], markersize=ms, label='all')
  ax.plot(iters, digit_retrain_new, '%s-' % formats[1], markersize=ms, label = 'window')
  ax.plot(iters, digit_no_retrain, '%s-' % formats[2], markersize=ms, label = 'none')

  ax.set_xlabel('Num Examples')
  ax.set_ylabel('Prediction Error')
  ax.legend(loc=0)

  fig.set_size_inches(4.0, 3.0)
  ax.set_ylim((0.0, ax.get_ylim()[1]))
  # plt.gcf().subplots_adjust(bottom=0.15, left=0.12)
  plt.gcf().subplots_adjust(bottom=0.20, left=0.20)
  plt.savefig('%sconcept-drift-none.pdf' % fig_dir)


if __name__=='__main__':
  # cache_lookup_latency()
  # wiki_cache_hit_rate()
  # retrain_latency()
  # anytime_predictions()
  # feature_eval_latency_poster()
  # digits_train_fs()
  # pred_dot_lats()
  concept_drift()
  
