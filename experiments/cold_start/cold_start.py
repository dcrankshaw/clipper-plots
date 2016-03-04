import pandas as pd
import numpy as np
from sklearn import linear_model as lm
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import classification as lg
import digit_class as digits
#import digits_function as digitsfunc
from datetime import datetime
import time
from sklearn import svm, ensemble
import json


###### Load Data #################

train_x, train_y = digits.load_digits("/Users/xinw/Documents/projects/velox-centipede/data", digits_filename = "mnist_train.csv")
#train_x, train_y = digits.load_digits("", digits_filename = "mnist_train.csv")
Z = digits.normalize_digits(train_x)
test_x, test_y = digits.load_digits("/Users/xinw/Documents/projects/velox-centipede/data", digits_filename = "mnist_test.csv")
#test_x, test_y = digits.load_digits("", digits_filename = "mnist_test.csv")
test_Z = digits.normalize_digits(test_x)
#Z = digits.fourier_project(train_x)

####### Initialization ##########
tasks = digits.create_mtl_datasets(Z, train_y, nTasks=100, taskSize=30, testSize=40)
xs,ys,dy,ss,test_xs,test_ys,test_dy,test_ts = digits.generate_data(tasks)
k = 20
k_svm = 10

oracle_mtl_errors = []
oracle_mtl = lg.LgSegmentModel(xs,ys,ss,'train-all',k)

for j in range(6):
    oracle_mtl.train_all_fs()
    oracle_mtl.train_ws()
oracle_err = lg.seg_model_error_01(oracle_mtl, test_xs, test_ys, test_ts, num=20)
#oracle_mtl_errors.append(oracle_err)
print "Initial Error OracleMTL: %f" % oracle_err

svm_mtl = lg.UserDefineModel(xs,ys,dy,ss,'train-all','l2') # using the default value 
svm_mtl.train_all_fs()
svm_mtl.train_ws()
svm_err = lg.seg_model_error_01(svm_mtl,test_xs,test_ys,test_ts)
print 'Initial Error SVM_mtl-l2 ', svm_err
svm_feature_errors = []

svm_mtl1 = lg.UserDefineModel(xs,ys,dy,ss,'train-all','l1') # using the default value 
svm_mtl1.train_all_fs()
svm_mtl1.train_ws()
svm_err = lg.seg_model_error_01(svm_mtl1,test_xs,test_ys,test_ts)
print 'Initial Error SVM_mtl-l1 ', svm_err
svm_feature_errors1 = []

d_mtl = lg.NonSharingModel(xs,ys,ss)
d_mtl.train_ws()
d_err = lg.seg_model_error_01(d_mtl,test_xs,test_ys,test_ts)
print 'Initial Error non-sharing model ', d_err
d_feature_errors = []
print 

####### Cold-Start Experiment #################
# number of points range from 2-30, step = 1
task = digits.create_mtl_datasets(Z,train_y,nTasks=100,taskSize=30,testSize=40)
test_xs,test_ys,test_ts = digits.generate_additional_data(task,oracle_mtl, svm_mtl, svm_mtl1, d_mtl,0)


iters = range(2,30)
for i in iters:
    print 'cold-start: # of points trained: ', i
    
    oracle_mtl.train_ws(i)
    oracle_err = lg.seg_model_error_01(oracle_mtl, test_xs, test_ys, test_ts, num=20)
    oracle_mtl_errors.append(oracle_err)
    print 'Testing Error of oracle model is ', oracle_err
        
    svm_mtl.train_ws(i)
    svm_err = lg.seg_model_error_01(svm_mtl,test_xs,test_ys,test_ts)
    svm_feature_errors.append(svm_err)
    print 'Testing Error of svm model --l2 is ', svm_err

    svm_mtl1.train_ws(i)
    svm_err = lg.seg_model_error_01(svm_mtl1,test_xs,test_ys,test_ts)
    svm_feature_errors1.append(svm_err)
    print 'Testing Error of svm model-- l1 is ', svm_err
    
    d_mtl.train_ws(i)
    d_err = lg.seg_model_error_01(d_mtl,test_xs,test_ys,test_ts)
    d_feature_errors.append(d_err)
    print 'Testing Error of non-sharing model is ', d_err
    print    
   
f = open('~/result/cold_start.json','w')
res = {}
res['num_points'] = iters
res['independent'] = d_feature_errors
res['oracle'] = oracle_mtl_errors
res['l1'] = svm_feature_errors1
res['l2'] = svm_feature_errors
json.dump(res,f)
f.close()





