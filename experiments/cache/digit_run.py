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
# number of points range from 2-20, step = 1
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
   
f = open('cold-start-experiment.txt','w')
f.write('Oracle Model:\n')
f.write('# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n')    

for t in oracle_mtl_errors:
    f.write('\t'+str(t))
f.write('\n\nSVM Model--L2 Norm:\n# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n')  

for t in svm_feature_errors:
    f.write('\t'+str(t))
f.write('\n')

f.write('\n\nSVM Model--L1 Norm:\n# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n')  

for t in svm_feature_errors1:
    f.write('\t'+str(t))
f.write('\n')

f.write('\n\n Non-sharing Model:\n# of points:')
for i in iters:
    f.write('\t'+str(i))
f.write('\n')  

for t in d_feature_errors:
    f.write('\t'+str(t))
f.write('\n')

f.close()

## plot ###
fig,ax = plt.subplots()
ax.plot(iters, oracle_mtl_errors, label='oracle')
ax.plot(iters, svm_feature_errors, label = 'svm-l2')
ax.plot(iters, svm_feature_errors1, label = 'svm-l1')
ax.plot(iters, d_feature_errors, label = 'non-sharing')
ax.set_xlabel('Size of the new task')
ax.set_ylabel('Error rate')
ax.set_title('Cold Start')
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()




"""

####### Add Tasks Experiment #####
k = 10
iters = range(100,1000,100)
oracle_mtl_errors = []
predefine_mtl_errors = []
sep_model_errors = []
for i in iters:
    print 'number of tasks is ', i
    tasks = digits.create_mtl_datasets(Z, train_y, nTasks=i, taskSize=10, testSize=20)
    xs,ys,dy,ss,test_xs,test_ys,test_dy,test_ts = digits.generate_data(tasks)
    
    ### oracle_mtl ###
    oracle_mtl = lg.LgSegmentModel(xs,ys,ss,k)      
    for j in range(6):
        oracle_mtl.train_all_fs()
        oracle_mtl.train_ws()
    oracle_err = lg.seg_model_error_01(oracle_mtl, test_xs, test_ys, test_ts)
    oracle_mtl_errors.append(oracle_err)
    print "Done training OracleMTL: %f" % oracle_err
    
    ### predefine_mtl ###
    predefine_mtl = lg.UserDefineModel(xs,ys,dy,ss,k)
    predefine_mtl.train_all_fs()
    predefine_mtl.train_ws()
    predefine_err = lg.seg_model_error_01(predefine_mtl,test_xs,test_ys,test_ts)
    predefine_mtl_errors.append(predefine_err)
    print "Done training predefineMTL: %f" % predefine_err
    
    ### Sep models ###
    sep_models = lg.LgSegmentModel(xs,ys,ss,k).segments
    for sid, s in sep_models.iteritems():
        m = lm.Ridge()
        m.fit(s.xs, s.ys)
        s.model = m
    sep_model_test_error = lg.separate_model_error_01(sep_models, test_xs, test_ys, test_ts)
    sep_model_errors.append(sep_model_test_error)
    print "Done training Separate Models: %f" % sep_model_test_error
    
### plot ####
fig, ax = plt.subplots()
#ax.plot(iters, oracle_mtl_errors, label="oracle")
#ax.plot(iters, sep_model_errors, label="sep")
ax.plot(iters, predefine_mtl_errors,label="predefine")
ax.set_title("Error vs Number of Tasks")
ax.set_xlabel("Total Number of Tasks")
ax.set_ylabel("01 Errors")
#ax.set_yscale('log')
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.7
#ax.set_ylim((0.0, 0.45))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()
"""

"""
######## Varying number of features ###
k_oracle = 20
k_svm = 10
nTasks=100
taskSize=100
tasks = digits.create_mtl_datasets(Z, train_y, nTasks, taskSize, testSize=20)
xs,ys,dy,ss,test_xs,test_ys,test_dy,test_ts = digits.generate_data(tasks)
oracle_feature_errors = []

iter_oracle = range(20,k_oracle+1)
iter_svm = range(0,k_svm*3+1,5)

for i in iter_oracle:
    print 'number of feature computed is ', i
    
    ### oracle_mtl ###
    oracle_mtl = lg.LgSegmentModel(xs,ys,ss,k_oracle)      
    for j in range(8):
        oracle_mtl.train_all_fs()
        oracle_mtl.train_ws()
    oracle_err = lg.seg_model_error_01(oracle_mtl, test_xs, test_ys, test_ts,i)
    oracle_feature_errors.append(oracle_err)
    print "Done training OracleMTL: %f" % oracle_err

points = nTasks*taskSize

svm_mtl = lg.UserDefineModel(xs,ys,dy,ss,'svm','logistic',points,k_svm)
svm_mtl.train_all_fs()
svm_mtl.train_ws()
svm_feature_errors = []


nb_mtl = lg.UserDefineModel(xs,ys,dy,ss,'naive_bayes',k_svm)
nb_mtl.train_all_fs()
nb_mtl.train_ws()
nb_feature_errors = []

log_mtl = lg.UserDefineModel(xs,ys,dy,ss,'logistic',k_svm)
log_mtl.train_all_fs()
log_mtl.train_ws()
log_feature_errors = []

dt_mtl = lg.UserDefineModel(xs,ys,dy,ss,'decision_tree',k_svm)
dt_mtl.train_all_fs()
dt_mtl.train_ws()
dt_feature_errors = []


gb_mtl = lg.UserDefineModel(xs,ys,dy,ss,'gradient_boosting',k_svm)
gb_mtl.train_all_fs()
gb_mtl.train_ws()
gb_feature_errors = []

for i in iter_svm:
    ### predefine_mtl ###
    print 'number of features is ', i
    svm_err = lg.seg_model_error_01(svm_mtl,test_xs,test_ys,test_ts,i)
    svm_feature_errors.append(svm_err)
    print "Done training svm_MTL: %f" % svm_err
    
    nb_err = lg.seg_model_error_01(nb_mtl,test_xs,test_ys,test_ts,i)
    nb_feature_errors.append(nb_err)
    print "Done training nb_MTL: %f" % nb_err
    
    log_err = lg.seg_model_error_01(log_mtl,test_xs,test_ys,test_ts,i)
    log_feature_errors.append(log_err)
    print "Done training log_MTL: %f" % log_err
    
    dt_err = lg.seg_model_error_01(dt_mtl,test_xs,test_ys,test_ts,i)
    dt_feature_errors.append(dt_err)
    print "Done training dt_MTL: %f" % dt_err
    

    gb_err = lg.seg_model_error_01(gb_mtl,test_xs,test_ys,test_ts,i)
    gb_feature_errors.append(gb_err)
    print "Done training svm_MTL: %f" % gb_err

### plot #####

fig, ax = plt.subplots()
ax.plot(iter_oracle, oracle_feature_errors, label="oracle")
#ax.plot(iters, sep_model_errors, label="sep")
#ax.plot(feature_iter, predefine_feature_errors,label="predefine")
ax.set_title("Error vs Number of Features Used in Online Updates")
ax.set_xlabel("Number of Features")
ax.set_ylabel("01 Errors")
#ax.set_yscale('log')
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()


fig, ax = plt.subplots()
#ax.plot(iter_oracle, oracle_feature_errors, label="oracle")
#ax.plot(iters, sep_model_errors, label="sep")
ax.plot(iter_svm, svm_feature_errors,label=svm_mtl.f_type)
#ax.plot(iter_svm, log_feature_errors,label=log_mtl.f_type)
#ax.plot(iter_svm, dt_feature_errors,label=dt_mtl.f_type)
#ax.plot(iter_svm, gb_feature_errors,label=gb_mtl.f_type)
#ax.plot(iter_svm, nb_feature_errors,label=nb_mtl.f_type)
ax.set_title("Error vs Number of Features Used in Online Updates")
ax.set_xlabel("Number of Features")
ax.set_ylabel("01 Errors")
#ax.set_yscale('log')
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()
"""
