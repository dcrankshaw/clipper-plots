#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil


# fname = "/Users/crankshaw/model-serving/centipede-plots/results/faas_benchmarks/spark_10rf.txt"
results_path = "../results"
# fig_dir = os.getcwd()
fig_dir = "/Users/crankshaw/ModelServingPaper/osdi_2016/figs"

import matplotlib.font_manager as font_manager

#view available fonts
for font in font_manager.findSystemFonts():
  print font
