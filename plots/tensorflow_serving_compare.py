import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil


# fname = "/Users/crankshaw/model-serving/centipede-plots/results/faas_benchmarks/spark_10rf.txt"
results_path = "../results"
# fig_dir = os.getcwd()
# fig_dir = "/Users/crankshaw/Dropbox/Apps/ShareLaTeX/velox-centipede/osdi_2016/figs"
fig_dir = "/Users/crankshaw/ModelServingPaper/osdi_2016/figs"
matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['font.size'] = 18
nbins=5
def parse_clipper_logs(fname):

    cur_batch = 0
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
    print "Clipper: avg: %f, p99: %f, thru: %f" % (avg_lat[0], p99_lat[0], thrus[0])
    return (avg_lat[0], p99_lat[0], thrus[0])

    # return (batch_sizes,  p99_lat,  p99_err,  avg_lat,  avg_err,  thrus,  thrus_err, max_lat)

def parse_tfserve_log(fname):
    mean_lats = []
    p99_lats = []
    thrus = []
    with open(fname, 'r') as f:
        for l in f:
            sp = l.split(" ")
            mean_lats.append(float(sp[2].rstrip(",")))
            p99_lats.append(float(sp[4].rstrip(",")))
            thrus.append(float(sp[6].rstrip(",")))


    # print np.mean(mean_lats)
    # print np.percentile(p99_lats, 99)
    # print np.sum(thrus)
    # print

    print "Tensorflow-Serving: avg: %f, p99: %f, thru: %f" % (np.mean(mean_lats), np.percentile(p99_lats, 99), np.sum(thrus))
    return (np.mean(mean_lats), np.percentile(p99_lats, 99), np.sum(thrus))


def plot():
  # print parse_clipper_logs("../results/tensorflow_compare/convnet_b32_slo10.txt")
  # print parse_tfserve_log("../results/tensorflow_compare/tf_serve_convnet_b32_results_35.txt") 
  # matplotlib.rcParams.update({'font.size': 12})
  print "ConvNet"
  conv_clipper = parse_clipper_logs("../results/tensorflow_compare/convnet_b32_slo10.txt")
  conv_tf = parse_tfserve_log("../results/tensorflow_compare/tf_serve_convnet_b32_results_35.txt")

  print "SoftMax"
  softmax_clipper = parse_clipper_logs("../results/tensorflow_compare/softmax_clipper.txt")
  softmax_tf = parse_tfserve_log("../results/tensorflow_compare/softmax_tfserve_60.txt")

  """
    PLOT THROUGHPUT
  """
  clipper_color = 'steelblue'
  # tf_color = 'darkred'
  tf_color = 'maroon'
  fig, ax = plt.subplots()
  plt.locator_params(nbins=nbins)
  ind = np.arange(2) + 0.05

  h_width = 0.30
  gap = 0.1
  clipper_thrus = ax.bar(ind, (conv_clipper[2], softmax_clipper[2]),
                          h_width, color=clipper_color, label="Clipper")

  tf_thrus = ax.bar(ind + h_width + gap, (conv_tf[2], softmax_tf[2]),
                          h_width, color=tf_color, label="TF-Serving")

  labels = ['ConvNet', 'SoftMax']
  ax.set_ylabel('Throughput')
  ax.set_xticks(ind + h_width + (gap/2.0))
  ax.set_xticklabels(labels)
  # ax.legend(loc=0)
  ax.legend(loc=2, fontsize=14,bbox_to_anchor=(-.03, 1.05))
  ax.xaxis.set_ticks_position('bottom')
  ax.yaxis.set_ticks_position('left')
  ylim = 15500
  if ylim is not None:
      ax.set_ylim((0, ylim))

  def autolabel(rects):
  # attach some text labels
      for rect in rects:
          height = rect.get_height()
          ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
              '%.0f' % height,
              ha='center', va='bottom')

  autolabel(clipper_thrus)
  autolabel(tf_thrus)
  width = 7.0*0.6
  height = 4.0*0.6
  fig.set_size_inches(width, height)

  plot_fname = "tf_compare_thrus.pdf"
  plt.savefig(fig_dir + "/" + plot_fname, bbox_inches='tight')
  # shutil.copy(plot_fname, fig_dir)

  """
    PLOT LATENCY
  """
  # clipper_color = 'steelblue'
  # tf_color = 'darkred'
  fig, ax = plt.subplots()

  plt.locator_params(nbins=nbins)
  clipper_lats = np.array((conv_clipper[0], softmax_clipper[0]))
  clipper_p99s = np.array((conv_clipper[1], softmax_clipper[1]))
  clipper_lat_errs = ((0,0), clipper_p99s - clipper_lats)
  tf_lats = np.array((conv_tf[0], softmax_tf[0]))
  tf_p99s = np.array((conv_tf[1], softmax_tf[1]))
  tf_lat_errs = ((0,0), tf_p99s - tf_lats)

  clipper_mean_lats = ax.bar(ind, clipper_lats,
                          h_width, color=clipper_color,
                          yerr=clipper_lat_errs,
                          ecolor='k',
                          error_kw={'elinewidth': 1.5,
                                    'capsize': 7,
                                    'linewidth': 4
                                    },
                          # linewidth=1,
                          label="Clipper")
  # clipper_p99_lats = ax.bar(ind, clipper_p99s,
  #                         h_width, fill=False,
  #                         edgecolor=clipper_color,
  #                         ecolor='k',
  #                         hatch="x",
  #                         label="Clipper p99")

  tf_mean_lats = ax.bar(ind + h_width + gap, tf_lats,
                          h_width,
                          yerr=tf_lat_errs,
                          ecolor='k',
                          error_kw={'elinewidth': 1.5,
                                    'capsize': 7,
                                    'linewidth': 4
                                   },
                          # linewidth=1,
                          color=tf_color, label="TF-Serving")
  # tf_p99_lats = ax.bar(ind + h_width + gap, tf_p99s,
  #                         h_width, fill=False,
  #                         edgecolor=tf_color,
  #                         ecolor='k',
  #                         hatch="x", label="TF-Serving p99")

  labels = ['ConvNet', 'SoftMax']
  ax.set_ylabel('Latency (ms)')
  ax.set_xticks(ind + h_width + (gap/2.0))
  ax.set_xticklabels(labels)
  # ax.legend(loc=2, fontsize=13)
  ylim = 63
  if ylim is not None:
      ax.set_ylim((0, ylim))

  def autolabel(rects):
  # attach some text labels
      for rect in rects:
          # print type(rect)
          height = rect.get_height()
          ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
              '%.2f' % height,
              ha='center', va='bottom')

  def autolabel_with_heights(rects, heights):
  # attach some text labels
      i = 0
      for rect in rects:
          # print type(rect)
          height = heights[i]
          ax.text(rect.get_x() + rect.get_width()/2., 1.03*height,
              '%.2f' % height,
              ha='center', va='bottom')
          i += 1

  # autolabel(clipper_p99_lats)
  autolabel_with_heights(clipper_mean_lats, clipper_p99s)
  # autolabel(tf_p99_lats)
  autolabel_with_heights(tf_mean_lats, tf_p99s)
  # width = 6.0*0.8
  # height = 4.0*0.8
  fig.set_size_inches(width, height)
  plot_fname = "tf_compare_lats.pdf"
  plt.savefig(fig_dir + "/" + plot_fname, bbox_inches='tight')
  # shutil.copy(plot_fname, fig_dir)



if __name__=='__main__':
  plot()

