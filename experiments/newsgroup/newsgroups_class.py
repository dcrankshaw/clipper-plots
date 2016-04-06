import numpy as np
from sklearn.datasets import fetch_20newsgroups
from datetime import datetime
from sklearn import linear_model as lm
from sklearn import svm, tree, ensemble
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import random
from sklearn.linear_model import SGDClassifier

class Task:
    def __init__(self, pref,X, y, target_y, test_X, test_y, test_target_y):
        self.pref = pref
        self.X = X
        self.y = y
        self.target_y = target_y
        self.test_X = test_X
        self.test_y = test_y
        self.test_target_y = test_target_y
        self.cluster = None


def load_data():
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    twenty_test = fetch_20newsgroups(subset='test',shuffle=True, random_state=42)
     
    x_train = twenty_train.data
    y_train = twenty_train.target
    x_test = twenty_test.data
    y_test = twenty_test.target
    print 'data loaded!'
    return (x_train, y_train, x_test, y_test)


def create_mtl_datasets(X, y,nTasks=1000, taskSize=100, testSize=100): 
    tasks = []
    for i in range(0, nTasks):
        #if i % 50 == 0:
         #   print "making task", i
        #print 'making task', i    
        #split = int(taskSize / 2.)
        #test_split = int(testSize / 2.)
        
        ### For training data positive vs negative = 7:3, for testing data postive vs negative = 4:6 
        split1 = int(taskSize *0.5)  #positive
        split2 = taskSize - split1 # negative
        
        test_split1 = int(taskSize *0.5) # positive
        test_split2 = taskSize - test_split1
        
        
        j = 0
        while True:
            j += 1 
            prefDigit = np.random.randint(0,20)
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
    pref_dist = np.zeros(20)
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
            dy.append(t.target_y[i])
            ss.append(sid)
        for i in range(len(t.test_y)):
            test_xs.append(t.test_X[i])
            test_ys.append(t.test_y[i])
            test_dy.append(t.test_target_y[i])
            test_ss.append(sid)
    print 'finish generate data!'
    return (xs,ys,dy,ss,test_xs,test_ys,test_dy,test_ss)

def generate_additional_data (tasks, model_1, model_2, overlap):
    '''
    Generate new data for addtional tasks; overlap == 0 -> totally new task(only one); overlap == 1 -> totally add to existing tasks. 
    
    model_1: oracle; model_2: user define model
    '''
    xs = model_2.xs
    ys = model_2.ys
    ss = model_2.ss
    dy = model_2.target_y
    
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
            dy.append(t.target_y[i])
            if sid not in model_1.segments:
                model_1.segments[sid] = Segment(sid, model_1.k,t.pref)
            model_1.segments[sid].add_example(t.X[i], t.y[i])
                        
            if sid not in model_2.segments:
                model_2.segments[sid] = Segment(sid, model_2.k,t.pref)
            model_2.segments[sid].add_example(t.X[i], t.y[i])
            
            """
            if sid not in model_3.segments:
                model_3.segments[sid] = Segment(sid, model_3.k,t.pref)
            model_3.segments[sid].add_example(t.X[i], t.y[i])
            """
            
        for i in range(len(t.test_y)):
            test_xs.append(t.test_X[i])
            test_ys.append(t.test_y[i])
            test_ss.append(sid)
    
    model_1.xs = xs
    model_1.ys = ys
    model_1.ss = ss
    model_1.target_y = dy

    return (test_xs,test_ys,test_ss)




    
def generate_data_concept_drift(X,y,model, num=100,g_type ='partial', perm = None):
    xs = model.xs
    ys = model.ys
    ss = model.ys
    dy = model.target_y

    max_ss = max(ss)
    min_ss = min(ss)
    
    test_xs = []
    test_ys = []
    test_ss = []
    
    if perm == None:
        perm = np.random.permutation(len(model.segments))[0:num]
        
    size = np.random.randint(5,12)
    for s in perm:
        if g_type == 'partial':
            if type(model.segments[s].pref) == list:
                a = np.random.randint(2)
                pref = model.segments[s].pref[a]
                pref2 = np.random.randint(10)
                while pref==pref2:
                    pref2 = np.random.randint(10)
                model.segments[s].pref[1-a] = pref2
            else:
                pref = model.segments[s].pref
                pref2 = np.random.randint(10)
                while pref==pref2:
                    pref2 = np.random.randint(10)
                model.segments[s].pref = [pref,pref2]
        elif g_type == 'total':
            pref = model.segments[s].pref
            if type(model.segments[s]) == list:
                pref = pref[0]
            pref2 = np.random.randint(10)
            while pref2==pref:
                pref2 = np.random.randint(10)
            model.segments[s].pref = pref2
        
        tasks = create_mtl_datasets_pref(X, y, model.segments[s].pref, nTasks=1, taskSize=size, testSize=40)
        for t in tasks:
            for i in range(len(t.y)):
                xs.append(t.X[i])
                ys.append(t.y[i])
                ss.append(s)
                dy.append(t.target_y[i])
                model.segments[s].add_example(t.X[i],t.y[i])
                
            for i in range(len(t.test_y)):
                test_xs.append(t.test_X[i])
                test_ys.append(t.test_y[i])
                test_ss.append(s)   

    return (test_xs,test_ys,test_ss,perm,size)
        
        

    
def create_mtl_datasets_pref(X, y,pref,nTasks=1000, taskSize=100, testSize=100): 
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
        split1 = int(taskSize/2.)
        split2 = taskSize - split1
        test_split1 = int(testSize/2.)
        test_split2 = testSize - test_split1
       
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


def seg_model_error_01(model, xs, ys, ss, num = 20):
    wrong = 0.0
    #print len(xs)
    segment_tests = {}
    for i in range(len(ss)):
        if ss[i] not in segment_tests:
                segment_tests[ss[i]] = Segment(ss[i], model.k)
        segment_tests[ss[i]].add_example(xs[i],ys[i])
    (y_pred,yss) = model.predict(segment_tests, num)
    for i in range(len(y_pred)):
        if y_pred[i] != yss[i]:
            wrong += 1
    return wrong/len(xs)    



class UserDefineModel:
    def __init__(self, xs, ys, target_y, ss, strategy = 'train-all', penalty= 'l2',k=20):

        assert len(xs) == len(ys) and len(xs) == len(ss)

        self.xs = xs
        self.ys = ys
        self.target_y = target_y
        self.ss = ss
        self.k = k
        self.perm = np.random.permutation(self.k)
        
        self.q = int(len(xs)*0.5)
        self.avg_f = None
        self.strategy = strategy
        self.penalty = penalty
        self.points = 20  # keep 20 points in cache
        self.wind = 20
        
        self.trained_fs = False
        self.trained_ws = False
        
        #self.transformed_ys = {}
        # randomly initialize segment models
        self.segments = {}
        # self.fs = [DummyModel(len(xs[0])) for i in range(k)]
        # self.fs = [None]*k
        self.fs = None

        for i in range(len(self.ss)):
            if self.ss[i] not in self.segments:
                self.segments[self.ss[i]] = Segment(self.ss[i], self.k)
            self.segments[self.ss[i]].add_example(self.xs[i], self.ys[i])
            
     
    
    def compute_features(self, x, s, i=20):
        # here using the confidence of the samples as the features to compute x.
        # result is of the shape (10,len(x))
        result = self.fs.predict_proba(x)
        #result = self.fs.decision_function(x)
        
        ws = self.segments[s].ws        
        if i == 20:
            return result.transpose()
        else:
            best_features = np.fliplr([np.argsort(ws)])[0][:i]
            #best_features = self.perm[0:i]
            res = np.zeros(result.shape)
            #avg = np.mean(result[:,0:i],axis=1)
            #for j in best_features:
            for j in range(self.k):
                if j in best_features:
                    res[:,j] = result[:,j]
                else:
                    #print 'the relative difference between true and avg: ', np.mean(result[:,j])
                    res[:,j] = self.segments[s].avg_f[j]
                    #res[:,j] = 0
            res= res.transpose()
            return res
            
    def train_ws(self, index=20,perm=None):
        feature_time = 0.0
        train_time = 0.0
        if perm == None:
            perm = self.segments
        for s in perm:
            segment = self.segments[s]
            start = datetime.now()
            ft = datetime.now() # featurization time
            
            """
            perm = np.random.permutation(len(segment.ys))[0:index]
            ys = [segment.ys[perm[m]] for m in range(index)]
            while not ((0 in ys) and (1 in ys)):
                perm = np.random.permutation(len(segment.ys))[0:index]
                ys = [segment.ys[perm[m]] for m in range(index)]
            xs = [segment.xs[perm[m]] for m in range(index)]
            """
            xs = segment.xs[0:index]
            ys = segment.ys[0:index]
            
            if self.strategy == 'train-all':
                #print len(xs)
                transformed_xs = self.compute_features(xs,s,self.k).transpose()           
                segment.avg_f = np.mean(transformed_xs, axis = 0)
                new_ws = lm.LogisticRegression(fit_intercept=False,penalty = self.penalty)
                new_ws.fit(transformed_xs,ys)
                segment.model = new_ws
                segment.ws = new_ws.coef_[0]
            elif self.strategy == 'retrain-new':
                xs = xs[index-self.wind:]
                ys = ys[index-self.wind:]
                #print len(xs)
                transformed_xs = self.compute_features(xs,s,self.k).transpose()           
                segment.avg_f = np.mean(transformed_xs, axis = 0)
                new_ws = lm.LogisticRegression(fit_intercept=False,penalty = self.penalty)
                new_ws.fit(transformed_xs,ys)
                segment.model = new_ws
                segment.ws = new_ws.coef_[0]
            elif self.strategy == 'robust-retrain':     
                xs = xs[self.points:]
                ys = ys[self.points:]
                if len(ys)<=2 and (not (0 in ys and 1 in ys)):
                    if len(ys)==0:
                        jj = 2
                        while segment.ys[self.points-1] == segment.ys[self.points-jj]:
                            jj +=1
                        xs = [segment.xs[self.points-jj],segment.xs[self.points-1]]
                        ys = [segment.ys[self.points-jj],segment.ys[self.points-1]]
                    elif len(ys)==1:
                        jj = 1
                        while segment.ys[self.points-jj] == ys[0]:
                            jj +=1
                        xs = [segment.xs[self.points-jj]]+xs
                        ys = [segment.ys[self.points-jj]]+ys
                    elif len(ys)==2:
                        jj = 1
                        while segment.ys[self.points-jj] == ys[1]:
                            jj +=1
                        xs[0] = segment.xs[self.points-jj]
                        ys[0] = segment.ys[self.points-jj]
                    
                #print len(xs)
                transformed_xs = self.compute_features(xs,s,self.k).transpose()           
                segment.avg_f = np.mean(transformed_xs, axis = 0)
                new_ws = lm.LogisticRegression(fit_intercept=False,penalty = self.penalty)
                new_ws.fit(transformed_xs,ys)
                segment.model = new_ws
                segment.ws = new_ws.coef_[0]
            elif self.strategy == 'average-weight':
                xs = xs[index-self.points:]
                ys = ys[index-self.points:]
                transformed_xs = self.compute_features(xs,s,self.k).transpose()           
                segment.avg_f = np.mean(transformed_xs, axis = 0)
                
                ws = segment.ws
                new_ws = lm.LogisticRegression(fit_intercept=False,penalty = self.penalty)
                new_ws.fit(transformed_xs,ys)
                new_ws.coef_ = (new_ws.coef_+ws)/2
                segment.model = new_ws
                segment.ws = new_ws.coef_[0]
            elif self.strategy == 'last-point':
                xs = xs[-1]
                ys = ys[-1]
                ws = segment.ws
                transformed_xs = self.compute_features(xs,s,self.k).transpose()[0,:]
                segment.avg_f = np.mean(transformed_xs, axis = 0)
                
                step_size = 1
                delta = (ys-np.dot(ws.reshape((1,self.k)),transformed_xs.reshape(self.k,1)))* transformed_xs
                #ws = ws + step_size*delta
                model = segment.model
                model.coef_ += step_size*delta[0]
                segment.model = model
                segment.ws = model.coef_[0]
            elif self.strategy == 'gradient-step':
                xs = xs[index-self.points:]
                ys = np.array(ys[index-self.points:]).reshape((self.points,1))
                transformed_xs = self.compute_features(xs,s,self.k).transpose()           
                segment.avg_f = np.mean(transformed_xs, axis = 0)
                               
                ws = segment.ws
                step_size = 10
                delta = 1.0/self.points*np.dot(transformed_xs.transpose(),(ys- np.dot(transformed_xs,ws.reshape((self.k,1)))))
                model = segment.model
                model.coef_ += step_size*delta.reshape((1,self.k))[0]
                segment.ws = model.coef_[0]
            
            
            
            
            #new_ws = lm.LogisticRegression()
            #new_ws = svm.SVC(kernel='linear')
            #new_ws = SGDClassifier()
            tt = datetime.now() # training time
            feature_time += (ft - start).total_seconds()
            train_time += (tt - ft).total_seconds()
        self.trained_ws = True
        print "ws: feature_time (s): %f, train_time (s): %f" % (feature_time, train_time)        
     
        
    ###### Train Shared Models #######
    def train_all_fs(self):
        fit_time = 0.0
        start = datetime.now()
        xs = self.xs
        dy = self.target_y
        #self.fs = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf',SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
        self.fs = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
        self.fs = self.fs.fit(xs,dy)
        end = datetime.now()
        print "TOTALS: fit time: %f" % (end-start).total_seconds()
        y_pred = self.fs.predict(xs)
        accuracy = np.mean(y_pred==dy)
        print 'Training accuracy of fs is ', accuracy
        self.trained_fs = True
    
    """
    def predict(self, x, s, i=20):
        if self.trained_fs and self.trained_ws:
            #print x
            features = self.compute_features(x,s,i).transpose()
            print features
            y = self.segments[s].model.predict(features)
            return y
        else:
            print "Please train model first"
            return 0.0
    """
    def predict(self,segs,i=20):
        y_preds = []
        ys = []
        if self.trained_fs and self.trained_ws:
            for s in segs:
                features = self.compute_features(segs[s].xs,s,i).transpose()
                y_pred = self.segments[s].model.predict(features)
                y_preds += y_pred.tolist()
                ys += segs[s].ys
            return (y_preds,ys)
        else:
            print 'Please train model first'
            return ([],[])
    
    
    
    # used for online updates
    def add_new_data_plusOne(self, xs, ys,ts):
        # if we have new users, create new segments
        for i in range(len(ts)):
            cls = ts[i]+1
            ts[i] = cls
            if cls not in self.segments:
                self.segments[cls] = Segment(cls, self.k)
            self.segments[cls].add_example(xs[i], ys[i])
    
    def add_new_data_unchanged(self, xs, ys,ts):
        # if we have new users, create new segments
        for i in range(len(ts)):
            cls = ts[i] 
            if cls not in self.segments:
                self.segments[cls] = Segment(ts[i], self.k)
            self.segments[cls].add_example(xs[i], ys[i])
        
    

class Segment:

    def __init__(self, sid, k, pref=None):
        # randomly init model
        self.ws = np.random.randn(k)
        self.sid = sid
        self.xs = []
        self.ys = []
        self.model = None
        self.avg_f = None
        self.pref = pref

    def add_example(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        
    def __str__(self):
        return str(self.ws)
    
    def __repr__(self):
        return str(self.ws)

    