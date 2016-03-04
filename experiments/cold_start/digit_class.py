import pandas as pd
import numpy as np
import classification as lg
import sklearn.linear_model
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import random

class Task:
    def __init__(self, pref,X, y,digity,test_X, test_y, test_digity):
        self.pref = pref
        self.X = X
        self.y = y
        self.digity = digity
        self.test_X = test_X
        self.test_y = test_y
        self.test_digity = test_digity
        self.cluster = None
        
    def __str__(self):
        return "pref: %d, y: %s, X: %s, test_y: %s, test_X: %s" % (self.pref,
                                                                   str(self.y),
                                                                   str(self.X),
                                                                   str(self.test_y),
                                                                   str(self.test_X))
    
    def __repr__(self):
        return str(self)

    

def load_digits(digits_location, digits_filename = "train-mnist-dense-with-labels.data"):
    digits_path = digits_location + "/" + digits_filename
    print "Source file:", digits_path
    df = pd.read_csv(digits_path, sep=",", header=None)
    data = df.values
    print "Number of image files:", len(data)
    y = data[:,0]   # digit label
    X = data[:,1:]
    return (X, y)

def to_image(x):
    return np.reshape(x,[28,28])

def display_digit(x):
    plt.imshow(to_image(x), interpolation='none')

def display_random_digits(X, y):
    ind = np.random.permutation(len(X))
    plt.figure()
    for i in range(0, 16):
        plt.subplot(4,4,i+1)
        display_digit(X[ind[i],:])
        plt.draw()
        # Display the plot
  

def normalize_digits(X):
    mu = np.mean(X,0)
    sigma = np.var(X,0)
    Z = (X - mu) / np.array([np.sqrt(z) if z > 0 else 1. for z in sigma])
    return Z 

def fourier_project(X, nfeatures = 4096, scale = 1.e-4):
    (n,d) = X.shape
    W = np.random.normal(scale = scale, size = [d, nfeatures])
    phase = np.random.uniform( size = [1, nfeatures]) * 2.0 * np.pi
    randomFeatures = np.cos(X.dot(W) + phase)
    return randomFeatures

def filter_two_class(X, y, digitA = 3, digitB = 9):
    yInd = (y == (digitA + 1)) | (y == (digitB + 1))
    yBinary = (y == (digitA + 1)) * 1.
    return (yInd, yBinary[yInd])


def train_test_split(y, propTrain = 0.75):
    ind = np.random.permutation(len(y))
    split_ind = ind[0.75 * len(y)]
    train_ind = ind[:split_ind]
    test_ind = ind[split_ind:]
    print "Train size: ", len(train_ind)
    print "Train true: ", np.mean(y[train_ind] == 1.0)
    print "Test size:  ", len(test_ind)
    print "Test true:  ", np.mean(y[test_ind] == 1.0)
    return (train_ind, test_ind)


class OracleModel:

    def __init__(self, train_xs, train_ys, test_xs, test_ys, pref_digit):
        self.train_xs = train_xs
        self.train_ys = train_ys
        self.test_xs = test_xs
        self.test_ys = test_ys
        self.pref_digit = pref_digit
        self.model = None

def gen_oracle_xs(pref_digit, num_examples):
    split = int(num_examples / 2.)
    true_xs = [np.zeros(10) for j in range(0, split)]
    for x in true_xs:
        x[pref_digit] = 1
    false_xs = [np.zeros(10) for j in range(0, split)]
    for x in false_xs:
        non_pref = np.random.randint(0,10)
        while non_pref == pref_digit:
            non_pref = np.random.randint(0,10)
        x[non_pref] = 1
    all_x = np.concatenate((true_xs, false_xs), axis = 0)
    all_y = np.concatenate((np.ones(split), np.zeros(split)), axis=0)
    shuffle_perm = np.random.permutation(len(all_x))
    xs = all_x[shuffle_perm, :]
    ys = all_y[shuffle_perm]
    return (xs, ys)


def create_oracle_datasets(nTasks=100, taskSize=100, testSize=100):
    tasks = []
    split = taskSize / 2
    for i in range(0, nTasks):
        if i % 50 == 0:
            print "making task", i
        prefDigit = np.random.randint(0,10)
        (train_xs, train_ys) = gen_oracle_xs(prefDigit, taskSize)
        (test_xs, test_ys) = gen_oracle_xs(prefDigit, testSize)
        tasks.append(OracleModel(train_xs, train_ys, test_xs, test_ys, prefDigit))
    return tasks

def create_mtl_datasets(X, y,nTasks=1000, taskSize=100, testSize=100): 
    tasks = []
    for i in range(0, nTasks):
        #if i % 50 == 0:
         #   print "making task", i
        #print 'making task', i    
        #split = int(taskSize / 2.)
        #test_split = int(testSize / 2.)
        
        ### For training data positive vs negative = 7:3, for testing data postive vs negative = 4:6 
        
        ## randome split datasets
        #split1 = int(taskSize *0.7)  #positive
        
        
        split1 = 0
        split2 = 0
        for m in range(taskSize):
            rand = np.random.rand(1)
            if rand >=0.7:
                split1 +=1
            else:
                split2 +=1
                
        while(taskSize>1 and (split1==0 or split2==0)):
            split1 = np.random.randint(taskSize)
            split2 = taskSize - split1 # negative
        
        
        test_split1 = 0
        test_split2 = 0
        for m in range(testSize):
            rand = np.random.rand(1)
            if rand <=0.4:
                test_split1 +=1
            else:
                test_split2 +=1
                
        while(testSize>1 and (test_split1==0 or test_split2==0)):
            test_split1 = np.random.randint(testSize)
            test_split2 = testSize - test_split1 # negative
        
        
        j = 0
        while True:
            j += 1 
            prefDigit = np.random.randint(0,10)
            perm = np.random.permutation(len(y)) # shuffle dataset
            inClass = y == prefDigit # bitmask of examples in preferred class
            t_ind = np.flatnonzero(inClass[perm]) # positive examples
            f_ind = np.flatnonzero(~inClass[perm]) # negative examples
            if t_ind.shape[0]>=split1+test_split1 and f_ind.shape[0]>=split2+test_split2:
                break
        
        tX = []
        fX = []
        tY = []
        fY = []
        for s in range(split1):
            tX.append(X[perm[t_ind[s]]])
            tY.append(y[perm[t_ind[s]]])
        for s in range(split2):
            fX.append(X[perm[f_ind[s]]])
            fY.append(y[perm[f_ind[s]]])
            
        newX = tX+fX
        newY = [1]*split1 + [0]*split2
        digity = tY+fY
        z = zip(newX,newY,digity)
        random.shuffle(z)
        newX,newY,digity = zip(*z)
        
        test_tX = []
        test_fX = []
        test_tY = []
        test_fY = []
        
        for s in range(split1,split1+test_split1):
            test_tX.append(X[perm[t_ind[s]]])
            test_tY.append(y[perm[t_ind[s]]])
         
        for s in range(split2,split2+test_split2):
            test_fX.append(X[perm[f_ind[s]]])
            test_fY.append(y[perm[f_ind[s]]])
            
        test_newX = test_tX + test_fX
        test_newY = [1]*test_split1 + [0]*test_split2
        test_digity = test_tY + test_fY
        z = zip(test_newX, test_newY, test_digity)
        random.shuffle(z)
        test_newX, test_newY, test_digity = zip(*z)        
        task = Task(prefDigit, newX, newY, digity, test_newX, test_newY, test_digity)
        tasks += [task]
    return tasks


def generate_data(tasks):
    pref_dist = np.zeros(10)
    for t in tasks:
        pref_dist[t.pref] += 1
    #print pref_dist

    xs = []
    ys = []
    dy = []
    ss = []

    test_xs = []
    test_ys = []
    test_dy = []
    test_ss = []

    for sid, t in enumerate(tasks):
        for i in range(len(t.y)):
            xs.append(t.X[i])
            ys.append(t.y[i])
            dy.append(t.digity[i])
            ss.append(sid)
        for i in range(len(t.test_y)):
            test_xs.append(t.test_X[i])
            test_ys.append(t.test_y[i])
            test_dy.append(t.test_digity[i])
            test_ss.append(sid)
    print 'finish generate data for user specific model!'
    return (xs,ys,dy,ss,test_xs,test_ys,test_dy,test_ss)

def generate_additional_data (tasks,model_1, model_2, model_3, model_4, overlap):
    '''
    Generate new data for addtional tasks; overlap == 0 -> totally new task(only one); overlap == 1 -> totally add to existing tasks. 
    
    model_1: oracle; model_2: user define model
    '''
    xs = model_2.xs
    ys = model_2.ys
    ss = model_2.ss
    dy = model_2.digitys
    
    if ss:
        max_ss = max(ss)
        min_ss = min(ss)
    
    test_xs = []
    test_ys = []
    test_ss = []
    
    for sid, t in enumerate(tasks):
        if overlap == 0:
            sid = sid + max_ss + 1
            #print 'sid is ', sid
        elif overlap == 1:
            if len(tasks) <= (max_ss-min_ss+1):
                #perm = np.random.permutation(max_ss-min_ss+1)[0:len(tasks)]
                #sid = perm[sid]
                #print 'sid is ', sid
                perm = range(min_ss,max_ss+1)
                sid = perm[sid]
            else:
                perm = np.random.permutation(max_ss-min_ss+1)
                perm = np.concatenate(perm,np.asarray([i+max_ss+1 for i in range(len(len(task)-(max_ss-min_ss+1)))]))       
                sid = perm[sid]
        elif overlap == 2:
            sid = max_ss
            #print 'sid is ', sid
        for i in range(len(t.y)):
            xs.append(t.X[i])
            ys.append(t.y[i])
            ss.append(sid)
            dy.append(t.digity[i])
            if sid not in model_1.segments:
                model_1.segments[sid] = lg.Segment(sid, model_1.k,t.pref)
            model_1.segments[sid].add_example(t.X[i], t.y[i])
                        
            if sid not in model_2.segments:
                model_2.segments[sid] = lg.Segment(sid, model_2.k,t.pref)
            model_2.segments[sid].add_example(t.X[i], t.y[i])

            if sid not in model_3.segments:
                model_3.segments[sid] = lg.Segment(sid, model_3.k,t.pref)
            model_3.segments[sid].add_example(t.X[i], t.y[i])
            
            if sid not in model_4.segments:
                model_4.segments[sid] = lg.Segment(sid, model_4.k,t.pref)
            model_4.segments[sid].add_example(t.X[i], t.y[i])            

            
        for i in range(len(t.test_y)):
            test_xs.append(t.test_X[i])
            test_ys.append(t.test_y[i])
            test_ss.append(sid)
    
    model_1.xs = xs
    model_1.ys = ys
    model_1.ss = ss

    model_3.xs = xs
    model_3.ys = ys
    model_3.ss = ss
    model_3.digitys = dy
    
    model_4.xs = xs
    model_4.ys = ys
    model_4.ss = ss
    return (test_xs,test_ys,test_ss)




    
def generate_data_concept_drift(X,y,model_1,model_2, g_type ='partial'):
    xs = model_2.xs
    ys = model_2.ys
    ss = model_2.ys
    dy = model_2.digitys

    max_ss = max(ss)
    min_ss = min(ss)
    
    test_xs = []
    test_ys = []
    test_ss = []
    
    for s in model_2.segments:
        if g_type == 'partial':
            if type(model_2.segments[s].pref) == list:
                pref = model_2.segments[s].pref[0]
                pref2 = np.random.randint(10)
                while pref==pref2:
                    pref2 = np.random.randint(10)
                model_2.segments[s].pref[1] = pref2
            else:
                pref = model_2.segments[s].pref
                pref2 = np.random.randint(10)
                while pref==pref2:
                    pref2 = np.random.randint(10)
                model_2.segments[s].pref = [pref,pref2]
        elif g_type == 'total':
            pref = model_2.segments[s].pref
            if type(mode_2.segments[s]) == list:
                pref = pref[0]
            pref2 = np.random.randint(10)
            while pref2==pref:
                pref2 = np.random.randint(10)
            model_2.segments[s].pref = pref2    

        model_1.segments[s].pref = model_2.segments[s].pref
        tasks = create_mtl_datasets_pref(X, y, model_2.segments[s].pref, nTasks=1, taskSize=1, testSize=100)
        for t in tasks:
            for i in range(len(t.y)):
                xs.append(t.X[i])
                ys.append(t.y[i])
                ss.append(s)
                dy.append(t.digity[i])
                model_2.segments[s].add_example(t.X[i],t.y[i])
                model_1.segments[s].add_example(t.X[i],t.y[i])
                
            for i in range(len(t.test_y)):
                test_xs.append(t.test_X[i])
                test_ys.append(t.test_y[i])
                test_ss.append(s)   

    model_1.xs = xs
    model_1.ys = ys
    model_1.ss = ss
    
    return (test_xs,test_ys,test_ss)
        
        

    
def create_mtl_datasets_pref(X, y, pref,nTasks=1000, taskSize=100, testSize=100): 
    '''
    pref is a list of preference of 2
    '''
    tasks = []
    for i in range(0, nTasks):
        #if i % 50 == 0:
         #   print "making task", i
        #print 'making task', i    
        #split = int(taskSize / 2.)
        #test_split = int(testSize / 2.)
        
        ### For training data positive vs negative = 7:3, for testing data postive vs negative = 4:6 
        #split1 = int(taskSize *0.7)  #positive
        
        
        split1 = 0
        split2 = 0
        for m in range(taskSize):
            rand = np.random.rand(1)
            if rand >=0.7:
                split1 +=1
            else:
                split2 +=1
        
        while(taskSize>1 and (split1==0 or split2==0)):
            split1 = np.random.randint(taskSize)
            split2 = taskSize - split1 # negative
        
                
        test_split1 = 0
        test_split2 = 0
        for m in range(testSize):
            rand = np.random.rand(1)
            if rand <=0.4:
                test_split1 +=1
            else:
                test_split2 +=1
                
        while(testSize>1 and (test_split1==0 or test_split2==0)):
            test_split1 = np.random.randint(testSize)
            test_split2 = testSize - test_split1 # negative
        
        j = 0
        while True:
            j += 1 
            #prefDigit = np.random.randint(0,10)
            if type(pref)==list:
                prefDigit = pref[0]
                perm = np.random.permutation(len(y)) # shuffle dataset
                inClass = y == prefDigit # bitmask of examples in preferred class
                prefDigit = pref[1]
                inClass1 = y == prefDigit
                inClass = inClass + inClass1
                prefDigit = pref
            else:
                prefDigit = pref
                perm = np.random.permutation(len(y)) # shuffle dataset
                inClass = y == prefDigit # bitmask of examples in preferred class
                
            t_ind = np.flatnonzero(inClass[perm]) # positive examples
            f_ind = np.flatnonzero(~inClass[perm]) # negative examples
            if t_ind.shape[0]>=split1+test_split1 and f_ind.shape[0]>=split2+test_split2:
                break
        
        tX = []
        fX = []
        tY = []
        fY = []
        for s in range(split1):
            tX.append(X[perm[t_ind[s]]])
            tY.append(y[perm[t_ind[s]]])
        for s in range(split2):
            fX.append(X[perm[f_ind[s]]])
            fY.append(y[perm[f_ind[s]]])
            
        newX = tX+fX
        newY = [1]*split1 + [0]*split2
        digity = tY+fY
        z = zip(newX,newY,digity)
        random.shuffle(z)
        newX,newY,digity = zip(*z)
        
        test_tX = []
        test_fX = []
        test_tY = []
        test_fY = []
        
        for s in range(split1,split1+test_split1):
            test_tX.append(X[perm[t_ind[s]]])
            test_tY.append(y[perm[t_ind[s]]])
         
        for s in range(split2,split2+test_split2):
            test_fX.append(X[perm[f_ind[s]]])
            test_fY.append(y[perm[f_ind[s]]])
            
        test_newX = test_tX + test_fX
        test_newY = [1]*test_split1 + [0]*test_split2
        test_digity = test_tY + test_fY
        z = zip(test_newX, test_newY, test_digity)
        random.shuffle(z)
        test_newX, test_newY, test_digity = zip(*z)        
        task = Task(prefDigit, newX, newY, digity, test_newX, test_newY, test_digity)
        tasks += [task]
    return tasks
    
    
    
