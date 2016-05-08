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
matplotlib.rcParams['font.size'] = 11

width=2
height=2

def plot():

  #### Digit; Training fs vs non-training fs
  b = "0.10025	0.53775	0.50025	0.47	0.43175	0.40675	0.374	0.349	0.339	0.31675	0.30725	0.29625	0.27775	0.2665	0.244	0.2285	0.221	0.20225	0.1925	0.166	0.13925	0.10575	0.106	0.10625	0.10375	0.11075	0.53575	0.47375	0.43625	0.407	0.38075	0.35825	0.339	0.316	0.2935	0.2845	0.27175	0.2525	0.24175	0.23	0.2105	0.19975	0.19075	0.17925	0.1565	0.133	0.10125	0.1005	0.10275	0.104	0.10225"
  digit_retrain_fs = [float(i) for i in b.split('\t')]
  # res_path = '/Users/crankshaw/Desktop/retrain_latency/retrain_results.json'
  # with open(res_path, 'w') as fp:
  #     json.dump(retrain_latency_results, fp)
  c = "0.10025	0.53725	0.477	0.43425	0.4155	0.39175	0.37325	0.34675	0.3345	0.32125	0.3085	0.2895	0.28225	0.27725	0.2595	0.25325	0.23775	0.2315	0.20325	0.1945	0.17175	0.1485	0.152	0.14975	0.14125	0.13625	0.5335	0.4815	0.43975	0.421	0.39325	0.36175	0.33975	0.31475	0.299	0.291	0.27225	0.25825	0.2615	0.248	0.239	0.228	0.215	0.208	0.187	0.159	0.1425	0.14275	0.14325	0.13975	0.13925"
  digit_no_retrain_fs = [float(i) for i in c.split('\t')]

  iters = range(20,71)
  fig,ax = plt.subplots()
  ax.plot(iters, digit_no_retrain_fs, label='Adapt')
  ax.plot(iters, digit_retrain_fs, label='Retrain')
  # ax.legend()
  ax.set_xlabel('Updates')
  ax.set_ylabel('Error')
  # ax.set_title('MNIST Data')
  ax.set_ylim((0.0, ax.get_ylim()[1]))
  fig.set_size_inches(width, height)
  plt.savefig('%s/digits_retrain_fs.pdf' % fig_dir, bbox_inches='tight')


  #### Newsgroup 10k users
  d = "0.0771	0.524	0.46725	0.4235	0.36325	0.3325	0.277	0.24775	0.22275	0.209	0.19875	0.1965	0.19175	0.18625	0.18475	0.17525	0.175	0.1655	0.1665	0.1605	0.1555	0.14525	0.14525	0.14275	0.143	0.13925	0.13875	0.1415	0.1445	0.148	0.14625"
  newsgroup_no_retrain_fs = [float(i) for i in d.split('\t')]
  e = "0.0771	0.52425	0.4665	0.41425	0.354	0.319	0.264	0.22825	0.204	0.1925	0.1765	0.17	0.16125	0.15375	0.15475	0.1505	0.14875	0.14	0.139	0.1385	0.137	0.13375	0.131	0.1275	0.1245	0.124	0.12225	0.119	0.11975	0.118	0.116"""
  newsgroup_retrain_fs = [float(i) for i in e.split('\t')]
  iters = range(20,51)
  fig,ax = plt.subplots()
  ax.plot(iters, newsgroup_no_retrain_fs, label='Adapt')
  ax.plot(iters, newsgroup_retrain_fs, label='Retrain')
  ax.legend(loc=0,fontsize='small')
  ax.set_xlabel('Updates')
  ax.set_ylabel('Error')
  # ax.set_title('Newsgroup Data -- 10k Users')
  ax.set_ylim((0.0, ax.get_ylim()[1]))
  fig.set_size_inches(width, height)
  plt.savefig('%s/newsgroup_retrain_fs.pdf' % fig_dir, bbox_inches='tight')


  #### Newsgroup 200 users
  m = "0.0885	0.52625	0.48625	0.469	0.455	0.44675	0.44225	0.417	0.41125	0.407	0.399	0.38675	0.381	0.37975	0.36775	0.36175	0.352	0.32475	0.322	0.3115	0.29925	0.292	0.28175	0.2705	0.26675	0.258	0.26225	0.24875	0.26775	0.28575	0.289"
  newsgroup_no_retrain_fs_200 = [float(i) for i in m.split('\t')]
  n = "0.0885	0.50975	0.4375	0.36625	0.336	0.30675	0.29975	0.264	0.24025	0.235	0.22525	0.21825	0.21075	0.1965	0.19275	0.183	0.17425	0.1655	0.15525	0.15275	0.149	0.14025	0.13475	0.12725	0.129	0.12475	0.1195	0.11825	0.11525	0.11275	0.11025"
  newsgroup_retrain_fs_200 = [float(i) for i in n.split('\t')]
  iters = range(20,51)
  fig,ax = plt.subplots()
  ax.plot(iters, newsgroup_no_retrain_fs_200, label='without retraing fs')
  ax.plot(iters, newsgroup_retrain_fs_200, label='retrain fs')
  ax.set_xlabel('Updates')
  ax.set_ylabel('Error')
  # ax.set_xlabel('Total number of data points')
  # ax.set_ylabel('Error rate')
  # ax.set_title('Newsgroup Data -- 200 Users')
  ax.set_ylim((0.0, ax.get_ylim()[1]))
  fig.set_size_inches(width, height)
  plt.savefig('%s/newsgroup_retrain_fs_200.pdf' % fig_dir, bbox_inches='tight')

if __name__=='__main__':
  plot()
