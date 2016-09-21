from __future__ import print_function
import json
import numpy as np
import os
import sys
import matplotlib as mpl

# NSDI_FIG_DIR = "/Users/crankshaw/model-serving/clipper_paper/ModelServingPaper/nsdi_2017/fake-figs"
# PAPER_ROOT_DIR = "/Users/crankshaw/model-serving/clipper_paper/ModelServingPaper"
PAPER_ROOT_DIR = "/Users/giuliozhou/Research/RISE/ModelServingPaper"
# NSDI_FIG_DIR = os.path.join(PAPER_ROOT_DIR, "nsdi_2017/figs2")
NSDI_FIG_DIR = os.path.join(PAPER_ROOT_DIR, "nsdi_2017/figs2")

name_map = {
        "noop": "No-Op",
        "logistic_reg": "Logistic Regression (SKLearn)",
        "linear_svm": "Linear SVM (SKLearn)",
        "spark_svm": "Linear SVM (PySpark)",
        "kernel_svm": "Kernel SVM (SKLearn)",
        "rf_d16": "Random Forest (SKlearn)",
        }

def get_results_files(results_dir):
    results_files = []
    for name in os.listdir(results_dir):
        if "results" in name:
            results_files.append(name)
    return results_files

def barchart_label(ax, rects, size=None, hmult=1.05, rot=0):
    for rect in rects:
        height = rect.get_height()
        if size is not None:
            ax.text(rect.get_x() + rect.get_width()/2., hmult*height,
                    '%d' % int(height), size=size,
                    ha='left', va='bottom',rotation=rot)
        else:
            ax.text(rect.get_x() + rect.get_width()/2., hmult*height,
                    '%d' % int(height),
                    ha='left', va='bottom',rotation=rot)

