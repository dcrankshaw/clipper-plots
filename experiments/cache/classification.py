import numpy as np
import os
import csv
from sklearn import linear_model as lm
from scipy.special import expit
from scipy.optimize import minimize, newton
from sklearn import svm, tree, ensemble
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import numpy.matlib as ml

def base_model():
    #return lm.LogisticRegression()
    #return svm.LinearSVC(dual=False)
    return svm.SVC()

class DummyModel:
    def __init__(self, d):
        """Random d-dimensional linear regression model.
        
        Used to randomly initialize feature models to something
        with a `predict()` method before they can be trained.
        
        Args:
            d (int): dimensions of model
        
        """
        
        
        self.d = d
        self.model = np.random.randn(d)
        
    
    def predict(self, x):
        assert len(x) == self.d
        return [np.dot(self.model, x)]

    
def ll(fi, wi, y, c):
    return (-1*np.log(1 + np.exp(c + wi*fi)) + y*(c + wi*fi))

def ll_der(fi, wi, y, c):
    return wi*y - wi*np.exp(c + wi*fi)/(np.exp(c + wi*fi) + 1)

class ClusterMTLModel:

    def __init__(self, tasks, k):

        """

        Args:
            xs (ndarray): n-by-d array containing feature vectors for all training data.
            ys (ndarray): n-by-1 array containing labels for each feature vector in `xs`.
                Label at index `i` in `ys` corresponds to feature vector at index `i` in `xs`.
                Labels must be either 0 or 1.
            ss (ndarray): n-by-1 array containing segment labels for each feature vector in `xs`.
            k (int): The dimension of the resulting segmented model.

        """

        self.k = k
        
        self.trained_fs = False
        self.trained_ws = False

        self.tasks = tasks
        self.clusters = None


    def set_clusters(self, clusters):
        self.clusters = clusters
        self.trained_fs = True

    def randomly_assign_tasks(self):
        for t in self.tasks:
            rand_f = np.random.randint(0, self.k)
            t.cluster = self.clusters[rand_f]
        self.trained_ws = True


    def assign_tasks(self):
        for t in self.tasks:
            best_accuracy = -1.0
            best_f = None
            for f in self.clusters:
                cur_accuracy = f.eval_task(t)
                if cur_accuracy > best_accuracy:
                    best_accuracy = cur_accuracy
                    best_f = f
            t.cluster = best_f
        self.trained_ws = True

    def train_clusters(self):
        # reset training data
        for f in self.clusters:
            f.xs = []
            f.ys = []

        # set training data
        for t in self.tasks:
            t.cluster.xs.append(t.X)
            t.cluster.ys.append(t.y)
            
        for f in self.clusters:
            f_model = base_model()
            f.xs = np.concatenate(f.xs, axis=0)
            f.ys = np.concatenate(f.ys, axis=0)
            if len(f.xs) > 0:
                f_model.fit(f.xs, f.ys)
                f.model = f_model
            else:
                print "cluster %d has no members" %f.idx
        self.trained_fs = True

    def predict(self, x, t):
        if self.trained_fs and self.trained_ws:
            return self.tasks[t].cluste.model.predict(x)[0]
        else:
            print "Please train model first"
            return 0.0

    def cluster_model_error_01(self):
        wrong = 0.0
        total = 0.0
        for t in self.tasks:
            for i in range(len(t.test_y)):
                y_pred = t.cluster.model.predict(t.test_X[i])[0]
                y_true = t.test_y[i]
                if y_pred != y_true:
                    wrong += 1.0
                total += 1.0
        return wrong/total


class Cluster:
    
    def __init__(self, idx, model=None):
        self.model = model
        self.idx = idx
        self.xs = []
        self.ys = []

        # (self.xs, self.ys) = collect_tasks_examples(tasks)
        
#    def collect_tasks_examples(self, tasks):
#        xs = []
#        ys = []
#        for t in tasks:
#            xs.extend(t.xs)
#            ys.extend(t.ys)
#        return (xs, ys)

    def eval_task(self, task):
        correct = 0.0
        for i in range(len(task.test_y)):
            y_true = task.test_y[i]
            y_pred = self.model.predict(task.test_X[i])[0]
            if y_true == y_pred:
                correct += 1
        return correct / float(len(task.test_y))

    
        
    

class LgSegmentModel:

    def __init__(self, xs, ys, ss, strategy,k):

        """

        Args:
            xs (ndarray): n-by-d array containing feature vectors for all training data.
            ys (ndarray): n-by-1 array containing labels for each feature vector in `xs`.
                Label at index `i` in `ys` corresponds to feature vector at index `i` in `xs`.
                Labels must be either 0 or 1.
            ss (ndarray): n-by-1 array containing segment labels for each feature vector in `xs`.
            k (int): The dimension of the resulting segmented model.

        """

        assert len(xs) == len(ys) and len(xs) == len(ss)

        self.xs = xs
        self.ys = ys
        self.ss = ss
        self.k = k
        self.strategy = strategy
        
        self.points = 20

        self.trained_fs = False
        self.trained_ws = False
        
        self.transformed_ys = {}
        
        self.avg_f = None
        self.type = 0
        
        


        # randomly initialize segment models
        self.segments = {}
        # self.fs = [DummyModel(len(xs[0])) for i in range(k)]
        self.fs = [None]*k

        for i in range(len(self.ss)):
            if self.ss[i] not in self.segments:
                self.segments[self.ss[i]] = Segment(self.ss[i], self.k)
            self.segments[self.ss[i]].add_example(self.xs[i], self.ys[i])
            
    def compute_features(self, x, s, num=20):
        result = np.array([self.fs[i].predict(x)[0] for i in range(self.k)])
        return result
        """
        if num == 20:
            return result
        else:
            ws = self.segments[s].ws
            best_features = np.fliplr([np.argsort(ws)])[0][:num]
            res = np.zeros(result.shape)
            for j in range(self.k):
                if j in best_features:
                    res[j] = result[j]
        """ 
            
            #result[:num] = result[best_features]
            #avg = np.mean(result[0:num])
            #result[num:] = avg
            #return res

    def train_ws(self,i=20):
        feature_time = 0.0
        train_time = 0.0
        for s in self.segments:
            segment = self.segments[s]
            transformed_xs = []
            start = datetime.now()
            
            perm = np.random.permutation(len(segment.ys))[0:i]
            ys = [segment.ys[perm[m]] for m in range(i)]
            while not ((0 in ys) and (1 in ys)):
                perm = np.random.permutation(len(segment.ys))[0:i]
                ys = [segment.ys[perm[m]] for m in range(i)]
            xs = [segment.xs[perm[m]] for m in range(i)]
            
            for x in xs:
                transformed_xs.append(self.compute_features(x,s,self.k))
            ft = datetime.now() # featurization time
            if self.strategy == 'train-all':
                #print 'strategy is ', self.strategy
                new_ws = lm.LogisticRegression(fit_intercept=False)
                new_ws.fit(transformed_xs,ys)
                segment.model = new_ws
                segment.ws = new_ws.coef_[0]
            elif self.strategy == 'retrain-new':
                #print 'strategy is ', self.strategy
                new_ws = lm.LogisticRegression(fit_intercept=False)
                xs = [transformed_xs[i] for i in range(len(transformed_xs)-self.points, len(transformed_xs))]
                ys = [segment.ys[i] for i in range(len(transformed_xs)-self.points, len(transformed_xs))]
                new_ws.fit(xs,ys)
                segment.model = new_ws
                segment.ws = new_ws.coef_[0]
            elif self.strategy == 'average-weight':
                #print 'strategy is ', self.strategy
                ws = segment.ws
                new_ws = lm.LogisticRegression(fit_intercept=False)
                xs = [transformed_xs[i] for i in range(len(transformed_xs)-self.points, len(transformed_xs))]
                ys = [segment.ys[i] for i in range(len(transformed_xs)-self.points, len(transformed_xs))]
                new_ws.fit(xs,ys)
                segment.model = new_ws
                segment.ws =(new_ws.coef_[0]+ws)/2
            elif self.strategy == 'last-point':
                #print 'strategy is ', self.strategy
                ws = segment.ws
                xs = transformed_xs[-1]
                ys = segment.ys[-1]
                step_size = 0.001
                delta = (ys-np.dot(ws.reshape((1,self.k)),xs.reshape(self.k,1)))*xs
                ws = ws + step_size*delta
                segment.ws = ws
            elif self.strategy == 'Gradient-step':
                #print 'strategy is ', self.strategy
                ws = segment.ws
                xs = np.zeros((self.points,self.k))
                ys = np.zeros((self.points,1))
                for i in range(len(transformed_xs)-self.points, len(transformed_xs)):
                    for j in range(len(transformed_xs[i])):
                        xs[i-(len(transformed_xs)-self.points)][j] = transformed_xs[i][j]
                    ys[i-(len(transformed_xs)-self.points)] = segment.ys[i]
                step_size = 0.001
                delta = np.dot(xs.transpose(),(ys- np.dot(xs,ws.reshape((self.k,1)))))
                ws = ws + step_size*delta.reshape((1,self.k))
                segment.ws = ws
                
            tt = datetime.now() # training time
            feature_time += (ft - start).total_seconds()
            train_time += (tt - ft).total_seconds()
        self.trained_ws = True
        print "ws: feature_time (s): %f, train_time (s): %f" % (feature_time, train_time)


    #####################################
    # Train shared models
    #####################################
    
    
    def transform_ys(self, i):
        # TODO make sure that y \in {0, 1}, not {-1, 1}
        X = []
        transformed_ys = []
        for d in range(len(self.xs)):
            y = self.ys[d]
            wi = self.segments[self.ss[d]].ws[i]
            yhat = 0.0
            # box constraints
            if y == 1.0 and wi > 0.0:
                yhat = 1.0
            elif y == 1.0 and wi <= 0.0:
                yhat = 0.0
            elif y == 0.0 and wi > 0.0:
                yhat = 0.0
            elif y == 0.0 and wi <= 0.0:
                yhat = 1.0
            else:
                raise ValueError("y == %f and wi == %f illegal" % (y, wi))
            X.append(self.xs[d])
            transformed_ys.append(yhat)
        #print transformed_ys
        self.transformed_ys[i] = transformed_ys
        return (X, transformed_ys)

    def train_single_f(self, i):
        start = datetime.now()
        X, transformed_ys = self.transform_ys(i)
        t_transform = datetime.now()
        #print "f_%d: transform ys time (s): %f" % (i, (t_transform - start).total_seconds())
        # TODO: is X being overwritten?
        f_i = base_model() # logistic regression
        #f_i = svm.NuSVC()
        #f_i = tree.DecisionTreeClassifier()
        #f_i = ensemble.RandomForestClassifier()
        #f_i = ensemble.GradientBoostingClassifier()
        f_i.fit(X, transformed_ys)
        t_fit = datetime.now()
        #print "fit single f time (s): %f\n" % (t_fit - t_transform).total_seconds()

        return (f_i, (t_transform - start).total_seconds(), (t_fit - t_transform).total_seconds())

    def train_all_fs(self):
        trans_time = 0.0
        fit_time = 0.0
        for i in range(self.k):
            (self.fs[i], t, f) = self.train_single_f(i)
            trans_time += t
            fit_time += f
        print "TOTALS: transform time: %f, fit time: %f" % (trans_time, fit_time)
        self.trained_fs = True
        
    def set_fs(self, fs):
        assert(len(fs) == self.k)
        self.fs = fs
        self.trained_fs = True
        

    def predict(self, x, s, num=20):
        if self.trained_fs and self.trained_ws:
            features = self.compute_features(x,s,num)
            y = self.segments[s].model.predict(features)
            return y
        else:
            print "Please train model first"
            return 0.0
        
    # used for online updates
    def add_new_data(self, xs, ys, ts):
        # if we have new users, create new segments
        for i in range(len(ts)):
            if ts[i] not in self.segments:
                self.segments[ts[i]] = Segment(ts[i], self.k)
            self.segments[ts[i]].add_example(xs[i], ys[i])


            
##############################################            
##### UserSpecificModel ######################
##### Add digit label as another instance ####
##############################################


class UserDefineModel:
    def __init__(self, xs, ys, digitys, ss, strategy,reg='l2', sep='share', q=20, k=10):

        """

        Args:
            xs (ndarray): n-by-d array containing feature vectors for all training data.
            ys (ndarray): n-by-1 array containing labels for each feature vector in `xs`.
                Label at index `i` in `ys` corresponds to feature vector at index `i` in `xs`.
                Labels must be either 0 or 1.
            ss (ndarray): n-by-1 array containing segment labels for each feature vector in `xs`.
            k (int): The dimension of the resulting segmented model.

        """

        assert len(xs) == len(ys) and len(xs) == len(ss)

        self.xs = xs
        self.ys = ys
        self.digitys = digitys
        self.ss = ss
        self.k = k
        self.perm = np.random.permutation(k)
        self.strategy = strategy
        self.reg = reg
        self.sep = sep
        
        self.q = q  # number of points used for online updates
        self.points = 20
        
        
        self.type = 1
        
        self.trained_fs = False
        self.trained_ws = False
        #self.f_type = f_type
        #self.w_type = w_type
        
        
        #self.transformed_ys = {}
        # randomly initialize segment models
        self.segments = {}
        # self.fs = [DummyModel(len(xs[0])) for i in range(k)]
        # self.fs = [None]*k
        self.fs = None
        self.fs2 = None
        self.fs3 = None
        
        self.hit = 0
 
        for i in range(len(self.ss)):
            if self.ss[i] not in self.segments:
                self.segments[self.ss[i]] = Segment(self.ss[i], self.k)
            self.segments[self.ss[i]].add_example(self.xs[i], self.ys[i])

    def compute_features(self, x, s, i=10):
        # here using the confidence of the samples as the features to compute x.
        # result is of the shape (10,len(x))
        #result = self.fs.predict_proba(x)
        result = self.fs.decision_function(x)
        #result2 = self.fs2.decision_function(x)
        #result3 = self.fs3.decision_function(x)
        
        #result = np.concatenate((result,result2,result3),axis = 1)
        return result.transpose()
        """
        if i == 10:
            return result.transpose()
        else:
            ws = self.segments[s].ws
            best_features = np.fliplr([np.argsort(abs(ws))])[0][:i]
            #best_features = self.perm[0:i]
            res = np.zeros(result.shape)
            #avg = np.mean(result[:,0:i],axis=1)
            #for j in best_features:
            for j in range(self.k):
                if j in best_features:
                    res[:,j] = result[:,j]
                else:
                    res[:,j] = self.segments[s].avg_f[j]
                    #res[:,j] = 0
            res= res.transpose()
            return res
        """    
    def train_ws(self,i=20):
        feature_time = 0.0
        train_time = 0.0
        print 'start training ws!'
        for s in self.segments:
            segment = self.segments[s]
            start = datetime.now()
            
            """
            perm = np.random.permutation(len(segment.ys))[0:i]
            ys = [segment.ys[perm[m]] for m in range(i)]
            while not ((0 in ys) and (1 in ys)):
                perm = np.random.permutation(len(segment.ys))[0:i]
                ys = [segment.ys[perm[m]] for m in range(i)]
            xs = [segment.xs[perm[m]] for m in range(i)]
            """
            xs = segment.xs
            ys = segment.ys
            transformed_xs = self.compute_features(xs,s,self.k).transpose()
            
            n = transformed_xs.shape[0]
            segment.avg_f = np.mean(transformed_xs, axis = 0)
            #print 'transformed_xs.shape is ', transformed_xs.shape
            ft = datetime.now() # featurization time
            if self.strategy == 'train-all':
                #print 'strategy is ', self.strategy
                new_ws = lm.LogisticRegression(fit_intercept=False, penalty=self.reg)
                new_ws.fit(transformed_xs, ys)
                segment.model = new_ws
                segment.ws = new_ws.coef_[0]
            elif self.strategy == 'retrain-new':
                #print 'strategy is ', self.strategy
                new_ws = lm.LogisticRegression(fit_intercept=False)
                xs = transformed_xs[n-self.points:n,:]
                ys = [segment.ys[i] for i in range(n-self.points,n)]
                new_ws.fit(xs,ys)
                segment.model = new_ws
                segment.ws = new_ws.coef_[0]
            elif self.strategy == 'average-weight':
                #print 'strategy is ', self.strategy
                ws = segment.ws
                new_ws = lm.LogisticRegression(fit_intercept=False)
                xs = transformed_xs[n-self.points:n,:]
                ys = [segment.ys[i] for i in range(n-self.points,n)]
                new_ws.fit(xs,ys)
                segment.model = new_ws
                segment.ws =(new_ws.coef_[0]+ws)/2
            elif self.strategy == 'last-point':
                #print 'strategy is ', self.strategy
                ws = segment.ws
                xs = transformed_xs[-1,:]
                ys = segment.ys[-1]
                step_size = 0.001
                delta = (ys-np.dot(ws.reshape((1,self.k)),xs.reshape(self.k,1)))*xs
                ws = ws + step_size*delta
                segment.ws = ws
            elif self.strategy == 'Gradient-step':
                #print 'strategy is ', self.strategy
                ws = segment.ws
                xs = transformed_xs[n-self.points:n,:]
                
                ys = np.array([segment.ys[i] for i in range(n-self.points,n)]).reshape((self.points,1))
                step_size = 0.001
                delta = np.dot(xs.transpose(),(ys- np.dot(xs,ws.reshape((self.k,1)))))
                ws = ws + step_size*delta.reshape((1,self.k))
                segment.ws = ws
            
            #new_ws = lm.LogisticRegression(fit_intercept=False,penalty='l2')
            #new_ws = lm.LogisticRegression()
            #new_ws = svm.SVC(kernel='linear')
            #new_ws.fit(transformed_xs, ys)
            tt = datetime.now() # training time
            feature_time += (ft - start).total_seconds()
            train_time += (tt - ft).total_seconds()
            #segment.model = new_ws
            #segment.ws = new_ws.coef_[0]
            #print 'ws of %d is ' % s, segment.ws
            #y_pred=segment.model.predict(transformed_xs)
            #ac = np.mean(y_pred==segment.ys)
            #print 'training accuracy of ws is ', ac
        self.trained_ws = True
        print "ws: feature_time (s): %f, train_time (s): %f" % (feature_time, train_time)        
     
        
    ###### Train Shared Models #######
    def train_all_fs(self):
        fit_time = 0.0
        start = datetime.now()
        xs = self.xs
        dy = self.digitys
        
        #### fs1: svm; fs2: logistic; fs3 = naiveBayes
        #self.fs = svm.LinearSVC()
        self.fs = lm.LogisticRegression()
        self.fs.fit(xs,dy)
        #self.fs2 = lm.LogisticRegression()
        #self.fs2.fit(xs,dy)
        #self.fs3 = SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
        #self.fs3.fit(xs,dy)
        
        end = datetime.now()
        print "TOTALS: fit time: %f" % (end-start).total_seconds()
        self.trained_fs = True
    
    def predict(self, x, s, key, dic,k=10):
        if self.trained_fs and self.trained_ws:
            #features = self.compute_features(x,s,i).transpose()
            feature = [0]*self.k
            for i in range(len(feature)):
                if key[i] in dic[i]:
                    feature[i] = dic[i][key[i]]
                    #print 'hit!'
                else:
                    self.hit += 1
                    feature[i] = self.segments[s].avg_f[i]
                    dic[i][key[i]] = self.compute_features(x,s,k).transpose()[0][i]
                    #print 'miss!'
           
            feature = np.asarray(feature)
            #print 'len of feature ', feature.shape
            y = self.segments[s].model.predict(feature)
            return y
        else:
            print "Please train model first"
            return 0.0
    
    def predict_non_cache(self, x, s,i=10):
        if self.trained_fs and self.trained_ws:
            features = self.compute_features(x,s,i).transpose()
            y = self.segments[s].model.predict(features)
            return y
        else:
            print "Please train model first"
            return 0.0
    
    # used for online updates
    def add_new_data(self, xs, ys, ts):
        # if we have new users, create new segments
        for i in range(len(ts)):
            if ts[i] not in self.segments:
                self.segments[ts[i]] = Segment(ts[i], self.k)
            self.segments[ts[i]].add_example(xs[i], ys[i])
    

        
        
class NonSharingModel:
    def __init__(self,xs,ys,ss,k=10):
        self.xs = xs
        self.ys = ys
        self.ss = ss
        self.k = k
        
        self.segments = {}
        for i in range(len(self.ss)):
            if self.ss[i] not in self.segments:
                self.segments[self.ss[i]] = Segment(self.ss[i], self.k)
            self.segments[self.ss[i]].add_example(self.xs[i], self.ys[i])

           
    def train_ws(self, i=None):
        
        for s in self.segments:
            segment = self.segments[s]
            start = datetime.now()
            if i==None:
                i = len(segment.ys)
            perm = np.random.permutation(len(segment.ys))[0:i]
            ys = [segment.ys[perm[m]] for m in range(i)]
            while not ((0 in ys) and (1 in ys)):
                perm = np.random.permutation(len(segment.ys))[0:i]
                ys = [segment.ys[perm[m]] for m in range(i)]
            xs = [segment.xs[perm[m]] for m in range(i)]
            
            new_ws = lm.LogisticRegression(fit_intercept=False)
            new_ws.fit(xs, ys)
            segment.model = new_ws
            segment.ws = new_ws.coef_[0]
        end = datetime.now()
        
        print 'Total Training time: %f.' % (end-start).total_seconds()
        
        
    def predict(self, x, s, i=10):
        y = self.segments[s].model.predict(x)
        return y
                  
        
        
class Segment:

    def __init__(self, sid, k,pref=None):
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

    
def likelihood(model, xs, ys, ss):
    ll = 0.0
    for i in range(len(xs)):
        wi = model.segments[ss[i]].ws
        y = ys[i]
        x = xs[i]
        features = model.compute_features(x)
        w_f = np.dot(wi, features)
        ll += y*w_f - np.log(1 + np.exp(w_f))
    return ll/len(ys)

def seg_model_error(model, xs, ys, ss):
    loss = 0.0
    for i in range(len(xs)):
        y_pred = model.predict(xs[i], ss[i])
        #w_f = np.dot(model.compute_features(xs), model.segments[ss[i]].ws)
        w_f = np.dot(model.segments[ss[i]].ws, model.compute_features(xs))        
        loss += -1.0*np.log(1 + np.exp(w_f)) + ys[i]*w_f
    return loss/len(xs)

def seg_model_error_01(model, xs, ys, ss, keys, dics,  num = 10):
    wrong = 0.0
    for i in range(len(xs)):        
        kk = [] 
        for j in range(10):
            kk.append(str(keys[j][i].tolist()))        
        y_pred = model.predict(xs[i], ss[i], kk, dics, num)
        y_true = ys[i]
        if y_pred != y_true:
            wrong += 1
    return wrong/len(xs)

def seg_model_error_init(model, xs, ys, ss, num = 10):
    wrong = 0.0
    for i in range(len(xs)):             
        y_pred = model.predict_non_cache(xs[i], ss[i],num)
        y_true = ys[i]
        if y_pred != y_true:
            wrong += 1
    return wrong/len(xs)

def base_model_error_01(model, xs, ys):
    wrong = 0.0
    for i in range(len(xs)):
        y_pred = model.predict(xs[i])[0]
        y_true = ys[i]
        if y_pred != y_true:
            wrong += 1
    return wrong/len(xs)

def separate_model_error_01(models, xs, ys, ss):
    wrong = 0.0
    for i in range(len(xs)):
        y_pred = models[ss[i]].model.predict(xs[i])[0]
        y_true = ys[i]
        if y_pred != y_true:
            wrong += 1
    return wrong/len(xs)


"""
def generate_training_data(d, k, n, num_s, test_n):

    '''Randomly generates MTL models and runs them forward to create training data.

    Args:
        d (int): Dimension of input feature vectors x.
        k (int): Number of shared feature models (dimension of intermediate embedding).
        n (int): Number of training data points.
        num_s (int): Number of segments to distribute training data among (number of ws).
        test_n (int): Number of test data points to generate.

    '''
    
    # Generate training data
    xs = []
    ss = []
    for i in range(n):
        xs.append(np.random.randn(d))
        ss.append(np.random.randint(num_s))

    ws = [np.random.randn(k) for i in range(num_s)]
    fs = [np.random.randn(d) for i in range(k)]

    ys = []

    for i in range(len(xs)):
        x = xs[i]
        w = ws[ss[i]]
        features_raw = []
        for j in fs:
            features_raw.append(np.dot(j, x))
        features = np.array(features_raw)
        y = max(0, np.sign(np.dot(w, features))) # max() is to convert -1 to 0
        ys.append(y)
        
     
    # Generate test data
    test_x = []
    test_s = []
    test_y = []
    
    for i in range(test_n):
        test_x.append(np.random.randn(d))
        test_s.append(np.random.randint(num_s))
        
    for i in range(len(test_x)):
        x = test_x[i]
        w = ws[test_s[i]]
        features_raw = []
        for j in fs:
            features_raw.append(np.dot(j, x))
        features = np.array(features_raw)
        y = max(0, np.sign(np.dot(w, features))) # max() is to convert -1 to 0
        test_y.append(y)
    
    return (xs, ys, ss, test_x, test_y, test_s)
"""
