import pandas as pd
import numpy as np
import sklearn.linear_model
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import time

import requests
import json
import os



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
ax3.set_title("500 Trees " + info_str(times['rf500.pipeline']), y=.5)
ax4.set_title("DNN " + info_str(times['imagenet']), y=0.5)
ax4.set_xlabel("Latency (ms)")

ax1.hist(np.array(times['rf10.pipeline']) * 1000.0, bins)
ax2.hist(np.array(times['rf100.pipeline']) * 1000.0, bins)
ax3.hist(np.array(times['rf500.pipeline']) * 1000.0, bins)
ax4.hist(np.array(times['imagenet']) * 1000.0, bins)

fig.savefig("latency_of_mllib_tree_pipelines.pdf")


bins = 100
fig = plt.figure()
ax4 = fig.add_subplot(414)
ax1 = fig.add_subplot(411, sharex=ax4)
ax2 = fig.add_subplot(412, sharex=ax4)
ax3 = fig.add_subplot(413, sharex=ax4)



ax1.set_title("NOP " + info_str(times['nop']), y=.5)
ax2.set_title("Single LR " + info_str(times['lrModel8.pipeline']), y=.5)
ax3.set_title("Decision Tree " + info_str(times['singleDT.pipeline']), y=.5)
ax4.set_title("One-vs-All LR " + info_str(times['ovrLR.pipeline']), y=.5)

ax4.set_xlabel("Latency (ms)")

ax1.hist(np.array(times['nop']) * 1000.0, bins)
ax2.hist(np.array(times['lrModel8.pipeline']) * 1000.0, bins)
ax3.hist(np.array(times['singleDT.pipeline']) * 1000.0, bins)
ax4.hist(np.array(times['ovrLR.pipeline']) * 1000.0, bins)

fig.savefig("latency_of_mllib_various_pipelines.pdf")
