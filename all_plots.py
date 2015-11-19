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
  fig.set_size_inches(8.0, 5.0)
  plt.gcf().subplots_adjust(bottom=0.15)

  plt.savefig('%sretrain_latency.pdf' % fig_dir)


def anytime_predictions():
  # data = {'noise_params': noise_params, 'p99_lats': p99_lats, 'accuracies': accuracies }
  res_path = 'anytime-preds.json'
  with open(res_path, 'r') as fp:
      data = json.load(fp)

  fig, (ax_lat, ax_acc) = plt.subplots(2, sharex=True)
  for s in ['-1', '10', '50', '100']:
      lab = s
      if s == -1:
          lab = 'No SLA' 
      ax_lat.plot(data['noise_params'], data['p99_lats'][s], '%s-' % formats[0], label=lab)
      ax_acc.plot(data['noise_params'], data['accuracies'][s], '%s-' % formats[0], label=lab)

  axes = (ax_lat, ax_acc)
  for ax in axes:
      ax.legend(loc=0)
      ax.set_ylim(0, ax.get_ylim()[1] + 0.1)
  ax_lat.set_ylabel("Latency")
  ax_acc.set_ylabel("Error")
  ax_acc.set_xlabel("Noise Param")
  fig.set_size_inches(8.0, 10.0)
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

  fig.set_size_inches(8.0, 10.0)
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

  fig.set_size_inches(8.0, 10.0)

  # fig.savefig("latency_of_mllib_various_pipelines.pdf")
  plt.savefig('%smllib-pipeline-eval-lat.pdf' % fig_dir)


def digits_train_fs():

  res_path = 'digits_train_fs.json'
  with open(res_path, 'r') as fp:
    data = json.load(fp)

  iters = range(20,71)
  fig,ax = plt.subplots()
  ms = 6
  ax.plot(iters, data['digit_no_retrain_fs'], '%s-' % formats[0], markersize=ms, label='tasks', )
  ax.plot(iters, data['digit_retrain_fs'], '%s--' % formats[1], markersize=ms, label='features + tasks')
  ax.legend(loc=0)
  ax.set_xlabel('train points per task')
  ax.set_ylabel('Error rate')
  # ax.set_title('MNIST Data')
  # default_size = fig.get_size_inches()
  # size_mult = 1.7
  ax.set_ylim((0.0, ax.get_ylim()[1]))
  fig.set_size_inches(8.0, 5.0)
  plt.gcf().subplots_adjust(bottom=0.15, left=0.12)
  plt.savefig('%sdigits-train-fs.pdf' % fig_dir)

# def newsgroups_train_fs():


if __name__=='__main__':
  cache_lookup_latency()
  wiki_cache_hit_rate()
  retrain_latency()
  anytime_predictions()
  feature_eval_latency()
  digits_train_fs()
  
