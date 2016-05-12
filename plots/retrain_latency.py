import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import json
import os

# fig_dir = os.getcwd()
fig_dir = "/Users/crankshaw/ModelServingPaper/osdi_2016/figs"
bar_color = 'steelblue'
# formats = ['bo', 'r^', 'gv', 'k*']
formats = ['b', 'r', 'g', 'k']
matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['font.size'] = 10

nbins=3
width=2.0
height=1.5

def retrain_latency():


  res_path = '../results/retrain_results.json'
  with open(res_path, 'r') as fp:
      data = json.load(fp)
  init_points = 20
  max_points = 50
  points = np.arange(init_points, max_points + 1)


  # retrain_latency_results = {'all_train_times': all_train_times, 'w_train_times': w_train_times}
  fig, ax = plt.subplots()
  plt.locator_params(nbins=nbins)
  task_train = np.mean(np.array(data['w_train_times'])*1000.0)
  all_train = np.mean(np.array(data['all_train_times'])*1000.0)
  ind = np.arange(1) + 0.05
  h_width = 0.6
  task_rects = ax.bar(np.array(0.05), task_train, h_width, color='steelblue')
  all_rects = ax.bar(np.array(1.05), all_train, h_width, color='steelblue')

  ax.set_yscale('log')
  ax.set_ylabel("Training Latency (ms)")
  labels = ['Update', 'Retrain']
  ax.set_xticks(np.array((0, 1)) + h_width / 2.0)
  ax.set_xticklabels(labels)

  ax.xaxis.set_ticks_position('bottom')
  ax.yaxis.set_ticks_position('left')

  def autolabel(rects):
  # attach some text labels
      for rect in rects:
          height = rect.get_height()
          ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
              '%.1f' % height,
              ha='center', va='bottom')

  autolabel(task_rects)
  autolabel(all_rects)
  fig.set_size_inches(width,height)

  plt.savefig(fig_dir + "/retrain_latency.pdf", bbox_inches='tight')

  # ms = 6
  # ax.plot(points, np.array(data['w_train_times'])*1000.0, '%s-' % formats[0], markersize=ms, label='tasks')
  # ax.plot(points, np.array(data['all_train_times'])*1000.0, '%s--' % formats[1], markersize=ms, label='features + tasks')
  # ax.set_yscale('log')
  # ax.set_xlabel('train points per task')
  # ax.set_ylabel('retrain latency (ms)')
  ax.set_ylim(ax.get_ylim()[0], 99000)
  # ax.legend(loc=0)
  # # fig.set_size_inches(6.0, 3.5)
  # fig.set_size_inches(4.0, 3.0)

  # plt.gcf().subplots_adjust(bottom=0.2, left=0.2)
  plt.savefig('%s/retrain_latency.pdf' % fig_dir, bbox_inches = 'tight')

if __name__=='__main__':
  retrain_latency()
