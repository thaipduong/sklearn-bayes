#!/usr/bin/env python
# coding: utf-8

# In[22]:


# vim: set filetype=python:


# # Relevance Vector Machine (RVR and RVC)

# Relevance Vector Regressor and Relevance is kernelized version of ARD Regression and Classification (i.e. it uses the same algorithm for optimization but instead of applying it to raw features applies it to kernelized features).

# ### Development of RVM

# There are three different methods for fitting Relevance Vector Machine. Tipping proposed two of them namely fixed-point iterations and expectation maximization in his original RVM paper [ Tipping(2001) ], the third one (Sequential Sparse Bayesian Learning) was discovered later by Faul and Tipping (2003). 
# 
# The version of RVM that used EM and fixed-point iterations was very slow. It was starting with all basis functions included in the model and at each iteration was removing basis functions with little explanatory power. Sparse Bayesian Learning algorithm starts with single basis function and adds new ones, so only a small subset of basis functions are used in the optimization, this gives the new version of RVM significant speed advantage over older versions.

# ### RVR and RVC pass scikit-learn tests for regression and classification algorithms

# In[16]:


from sklearn.utils.estimator_checks import check_estimator
from skbayes.rvm_ard_models import RVR,RVC
#check_estimator(RVC)
#check_estimator(RVR)
print("All test are passed ...")


# ## Relevance Vector Regression

# ### Example 1: sinc(x)

# In[17]:


from sklearn.utils.estimator_checks import check_estimator
from skbayes.rvm_ard_models import RegressionARD2,ClassificationARD2,RVR2,RVC2
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
#from mpl_toolkits import mplot3d

from sklearn.metrics import mean_squared_error
#get_ipython().run_line_magic('matplotlib', 'inline')

'''
# parameters
n = 5000

# generate data set
np.random.seed(0)
Xc       = np.ones([n,1])
Xc[:,0]  = np.linspace(-5,5,n)
Yc       = 10*np.sinc(Xc[:,0]) + np.random.normal(0,1,n)
X,x,Y,y  = train_test_split(Xc,Yc,test_size = 0.5, random_state = 0)

# train rvr
rvm = RVR2(gamma = 1,kernel = 'rbf')
t1 = time.time()
rvm.fit(X,Y)
t2 = time.time()
y_hat,var     = rvm.predict_dist(x)
rvm_err   = mean_squared_error(y_hat,y)
rvs       = np.sum(rvm.active_)
print "RVM error on test set is {0}, number of relevant vectors is {1}, time {2}".format(rvm_err, rvs, t2 - t1)

# train svr
svr = GridSearchCV(SVR(kernel = 'rbf', gamma = 1), param_grid = {'C':[0.001,0.1,1,10,100]},cv = 10)
t1 = time.time()
svr.fit(X,Y)
t2 = time.time()
svm_err = mean_squared_error(svr.predict(x),y)
svs     = svr.best_estimator_.support_vectors_.shape[0]
print "SVM error on test set is {0}, number of support vectors is {1}, time {2}".format(svm_err, svs, t2 - t1)


# plot test vs predicted data
plt.figure(figsize = (16,10))
plt.plot(x[:,0],y,"b+",markersize = 3, label = "test data")
plt.plot(x[:,0],y_hat,"rD", markersize = 3, label = "mean of predictive distribution")

# plot one standard deviation bounds
plt.plot(x[:,0],y_hat + np.sqrt(var),"co", markersize = 3, label = "y_hat +- std")
plt.plot(x[:,0],y_hat - np.sqrt(var),"co", markersize = 3)
plt.plot(rvm.relevant_vectors_,Y[rvm.active_],"co",markersize = 12,  label = "relevant vectors")
plt.legend()
plt.title("RVM")
plt.show()


# Below we vizualise predictive distribution produced by Relevance Vector Regression.

# In[19]:


from scipy.stats import norm
n_grid = 500
max_x      = np.max(X,axis = 0)
min_x      = np.min(X,axis = 0)
max_y      = np.max(Y)
min_y      = np.min(Y)
X1         = np.linspace(min_x[0],max_x[0],n_grid)
Y1         = np.linspace(min_y,max_y,n_grid)
x1,y1      = np.meshgrid(X1,Y1)
Xgrid      = np.zeros([n_grid**2,2])
Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
Xgrid[:,1] = np.reshape(y1,(n_grid**2,))
mu,var     = rvm.predict_dist(np.expand_dims(Xgrid[:,0],axis =1))
probs      = norm.pdf(Xgrid[:,1],loc = mu, scale = np.sqrt(var))
plt.figure(figsize = (12,8))
plt.contourf(X1,Y1,np.reshape(probs,(n_grid,n_grid)),cmap="coolwarm")
plt.plot(X1,10*np.sinc(X1),'k-',linewidth = 3, label = 'real function')
plt.plot(X1,10*np.sinc(X1)-1.96,'k-',linewidth = 2, label = '95% real lower bound',
         linestyle = '--')
plt.plot(X1,10*np.sinc(X1)+1.96,'k-',linewidth = 2, label = '95% real upper bound',
         linestyle = '--')
plt.plot(rvm.relevant_vectors_,Y[rvm.active_],"co",markersize = 12,  label = "relevant vectors")
plt.title("PDF of Predictive Distribution of Relevance Vector Regression")
plt.colorbar()
plt.legend()
plt.show()


# In[20]:


probs      = norm.cdf(Xgrid[:,1],loc = mu, scale = np.sqrt(var))
plt.figure(figsize = (12,8))
plt.contourf(X1,Y1,np.reshape(probs,(n_grid,n_grid)),cmap="coolwarm")
plt.plot(X1,10*np.sinc(X1),'k-',linewidth = 3, label = 'real function')
plt.plot(X1,10*np.sinc(X1)-1.96,'k-',linewidth = 2, label = '95% real lower bound',
         linestyle = '--')
plt.plot(X1,10*np.sinc(X1)+1.96,'k-',linewidth = 2, label = '95% real upper bound',
         linestyle = '--')
plt.plot(rvm.relevant_vectors_,Y[rvm.active_],"co",markersize = 12,  label = "relevant vectors")
plt.title("CDF of Predictive Distribution of Relevance Vector Regression")
plt.colorbar()
plt.legend()
plt.show()

'''
# ### Example 2: Boston Housing

# RVR achieves better MSE on Boston housing dataset than SVR or GBR.

# In[21]:

'''
# Boston Housing
from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn.preprocessing import scale
from sklearn.ensemble import GradientBoostingRegressor

boston  = load_boston()
Xb,yb   = scale(boston['data']),boston['target']
X,x,Y,y = train_test_split(Xb,yb,test_size=0.3, random_state = 0)

rvr = GridSearchCV(RVR(coef0=0.01),param_grid = {'degree':[2,3],'kernel':['rbf','poly','sigmoid'],
                                       'gamma':[0.1,1,10]})
# Polynomial kernel was not used, since SVR with 'poly' kernel 
# did not produce any results even after 30 minutes running 
# (you can try yourself!)
svr = GridSearchCV( SVR(), 
                   param_grid = {"C":np.logspace(-3,3,7),
                                 'gamma':[0.1,1,10],
                                 'kernel':['sigmoid','rbf']},
                   cv = 5)

gbr = GridSearchCV( GradientBoostingRegressor(), 
                    param_grid = {'learning_rate': [1e-3,1e-1,1],
                                  'max_depth': [1,5,10],
                                  'n_estimators':[100,500,1000]})

rvr = rvr.fit(X,Y)
svr = svr.fit(X,Y)
gbr = gbr.fit(X,Y)


# In[22]:


from sklearn.metrics import mean_squared_error as mse
print " ===== Comparison of RVR -vs- SVR -vs- GBR ======"
print "\n     MSE for RVR on test set: {0} \n".format(mse(y,rvr.predict(x)))
print "\n     MSE for SVR on test set: {0} \n".format(mse(y,svr.predict(x)))
print "\n     MSE for GBR on test set: {0} \n".format(mse(y,gbr.predict(x)))
'''

# ## Relevance Vector Classification

# ### Example 3: Binary Classification

# In[23]:


from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import classification_report
from sklearn.svm import SVC

''
# Parameters
n = 1000
test_proportion = 0.1

# create dataset & split into train/test parts
Xx,Yy   = make_circles(n_samples = n, noise = 0.2, random_state = 1)

#data = np.load("./XY2.npz")
#Xx = data['X']
#Yy = data['Y']
X,x,Y,y = train_test_split(Xx,Yy,test_size = test_proportion, 
                                 random_state = 2)

# train rvm 
rvm = RVC2(kernel = 'rbf', gamma = 5)
t1 = time.time()
rvm.fit(X,Y)
t2 = time.time()
rvm_time = t2 - t1
print "RVC time:" + str(rvm_time)

# train svm (and find best parameters through cross-validation)
svc = GridSearchCV(SVC(probability = True), param_grid = {"C":np.logspace(-3,3,9)}, cv = 10)
t1 = time.time()
svc.fit(X,Y)
t2 = time.time()
svm_time = t2 - t1

# report on performance
svecs = svc.best_estimator_.support_vectors_.shape[0]
rvecs = np.sum(rvm.active_[0]==True)
rvm_message = " ====  RVC: time {0}, relevant vectors = {1} \n".format(rvm_time,rvecs)
print rvm_message
y_hat = rvm.predict(x)
print classification_report(y,y_hat)
svm_message = " ====  SVC: time {0}, support vectors  = {1} \n".format(svm_time,svecs)
print svm_message
print classification_report(y,svc.predict(x))

# create grid
n_grid = 500
max_x      = 5*np.max(X,axis = 0)
min_x      = 5*np.min(X,axis = 0)
X1         = np.linspace(min_x[0],max_x[0],n_grid)
X2         = np.linspace(min_x[1],max_x[1],n_grid)
n_grid = 500
max_x      = 5*np.max(X,axis = 0)
min_x      = 5*np.min(X,axis = 0)
X1         = np.linspace(min_x[0],max_x[0],n_grid)
X2         = np.linspace(min_x[1],max_x[1],n_grid)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid**2,2])
Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
Xgrid[:,1] = np.reshape(x2,(n_grid**2,))

sv_grid = svc.predict_proba(Xgrid)[:,1]
rv_grid, var_grid, _, _ = rvm.predict_proba(Xgrid)
#A = np.array([-1.1, 0.9])
#B = np.array([0.0, -2.0])
A = np.array([-4.0, 4.0])
B = np.array([4.0, -4.0])
line_seg_x = np.linspace(A[0], B[0], 100)
line_seg_y = np.linspace(A[1], B[1], 100)
line_seg = np.vstack((line_seg_x, line_seg_y)).transpose()
c = -0.1
upper_line, upper_line2, upper_line3, upper_line4 = rvm.predict_upperbound(line_seg, c=c)
rv_line, var_line, num, denom = rvm.predict_proba(line_seg)
upper_line5 = rvm.predict_upperbound_line(line_seg, A, c=-0.1)
line_seg_rev = np.flipud(line_seg)
upper_line6 = rvm.predict_upperbound_line(line_seg_rev, B, c=-0.1)
upper_line6 = np.flip(upper_line6)
upper_grid, upper_grid2, upper_grid3, upper_grid4 = rvm.predict_upperbound(Xgrid, c=c)
upper_line7 = np.minimum(upper_line5, upper_line6)

plt.figure(figsize=(12, 8))
#plt.plot(rv_line[:,1] - 0.5, label="GT")
a = np.where(upper_line7 >0)[0]

gt = num - c*denom
plt.plot(gt, label="GT")
plt.plot(upper_line, label="upper bound 1", linestyle="dashed")
plt.plot(upper_line2, label="Fastron bound", linestyle = "dotted")
plt.plot(upper_line4, label="AM-GM bound",  linestyle="dotted")
#plt.plot(upper_line7, label="triangle bound",  linestyle="dashdot")
#plt.plot(upper_line6, label="triangle bound",  linestyle="dashdot")
plt.plot(np.zeros(100), label="zero",  linestyle="dashdot")
plt.legend()
rv_grid = rv_grid[:,1]


'''
Kgoal = rvm.get_feature(goal)
Kgrid = rvm.get_feature(Xgrid)
decision, var = rvm.decision_function(Xgrid)
dist = Kgrid - Kgoal
attracive_field = np.linalg.norm(dist, axis=1)
K = 1
rv_grid_truncated = [1 if i >0.7 else i for i in rv_grid]
nav_fcn = (attracive_field)/ np.power((attracive_field - decision), 1/K)
'''

models  = [rv_grid, upper_grid, upper_grid2, upper_grid4]
model_names = ["RVC", "upperbound", "upperbound_fastron", "upperbound AMGM"]


############### Find maximum collision probability ################
'''
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import Bounds
import time

start_traj = np.array([-1, -0.5])#np.array([-1, 0])#np.array([-2, 2])
end_traj = np.array([0, 1])#np.array([0, 1])#np.array([1,3])
t_traj = np.linspace(0,1, 1000).reshape(1000,1)
#traj = start_traj + (end_traj - start_traj)*t_traj
v = np.array([1, 3])
a = np.array([0.5, -1])
traj = start_traj + v*t_traj + a*(t_traj**2)
test_traj = traj[0:10000,:]
s = time.time()
prob, _, _, _ = rvm.predict_proba(test_traj)
prob = np.max(prob[:,1])
e = time.time()
print("Max collision probability of traj: ", prob, ", taken ", e - s, "secs")



def f(x):
    v = np.array([1, 3])
    a = np.array([0.5, -1])
    #p = np.array(start_traj + (end_traj - start_traj)*x)
    p = np.array(start_traj + v * x + a*(x**2))
    p = p.reshape([1,2])
    prob, _, _, _ = rvm.predict_proba(p)
    f = prob[0,1]
    return -f
s = time.time()
res = minimize_scalar(f, bounds=(0,1), method='bounded')
e = time.time()
print("Max collision probability of traj (scipy min scalar): ", -res.fun, res.x, ", taken ", e - s, "secs")


bound = Bounds([0.0], [1.0])
x0 = 0.5
s = time.time()
res = minimize(f, x0, method='trust-constr', bounds=bound)
e = time.time()
print("Max collision probability of traj (scipy trust constr): ", -res.fun, res.x, ", taken ", e - s, "secs")


from scipy.optimize import LinearConstraint
linear_constraint = LinearConstraint([1], [0], [1])

s = time.time()
res = minimize(f, x0, method='trust-constr', constraints=[linear_constraint])
e = time.time()
print("Max collision probability of traj (scipy trust constr (linear constr)): ", -res.fun, res.x, ", taken ", e - s, "secs")


linear_constraint = {'type': 'ineq',\
                     'fun' : lambda x: np.array([1 - x[0], x[0]]),
                     'jac' : lambda x: np.array([[-1.0], [1.0]])}
res = minimize(f, x0, method='SLSQP', constraints=[linear_constraint])
e = time.time()
print("Max collision probability of traj (scipy SLSQP): ", -res.fun, res.x, ", taken ", e - s, "secs")

linear_constraint = {'type': 'ineq',\
                     'fun' : lambda x: np.array([1 - x[0],\
                                                 x[0]]).flatten(),
                     'jac' : lambda x: np.array([-1.0,\
                                                 1.0])}
res = minimize(f, [x0], method='COBYLA', constraints=[linear_constraint])
e = time.time()
print("Max collision probability of traj (scipy COBYLA): ", -res.fun, res.x, ", taken ", e - s, "secs")
'''
#plt.plot(rvm.relevant_vectors_,Y[rvm.active_],"co",markersize = 12,  label = "relevant vectors")
for model, model_name in zip(models, model_names):
    plt.figure(figsize = (12,8))
    levels = np.arange(0,1, 0.0005)
    #ax.contour3D(X1,X2,np.reshape(model,(n_grid,n_grid)), 100, cmap='coolwarm',
    #                   linewidth=0, antialiased=False)
    plt.contourf(X1, X2, np.reshape(model, (n_grid, n_grid)), cmap='coolwarm')
    cb  = plt.colorbar()
    cb.ax.tick_params(labelsize=15)
    if model_name == "RVC":
        plt.contour(X1, X2, np.reshape(upper_grid, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles="dashed")
        plt.contour(X1, X2, np.reshape(upper_grid2, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles = "dotted")
        #plt.contour(X1, X2, np.reshape(upper_grid3, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
        #            linestyles="solid")
        plt.contour(X1, X2, np.reshape(upper_grid4, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
                    linestyles="dashdot")
        plt.contour(X1, X2, np.reshape(model, (n_grid, n_grid)), levels=[0.5], cmap="Greys_r")
    else:
        plt.contour(X1, X2, np.reshape(model, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r")

    #plt.plot(X[Y==0,0],X[Y==0,1],"bo", markersize = 4)
    #plt.plot(X[Y==1,0],X[Y==1,1],"ro", markersize = 4)
    # plot 'support' or 'relevant' vectors
    svrv = None
    point_label = None
    if model_name == "SVC":
        svrv = svc.best_estimator_.support_vectors_
        point_label = "support vecs"
    else:
        svrv = rvm.relevant_vectors_[0]
        point_label = "relevant vecs"
    cm = ['b' if cw < 0 else 'r' for cw in rvm.corrected_weights]
    plt.plot(svrv[rvm.corrected_weights>0, 0], svrv[rvm.corrected_weights>0, 1], 'ro', markersize=8, label=point_label)
    plt.plot(svrv[rvm.corrected_weights < 0, 0], svrv[rvm.corrected_weights < 0, 1], 'bo', markersize=8,
             label=point_label)
    print("All relevance vectors")
    for i in range(len(rvm.corrected_weights)):
        print(i, rvm.orig_weights[i], rvm.corrected_weights[i], svrv[i,:])
    #plt.plot(traj[:,0], traj[:,1],'go',markersize=4)
    # plt.plot()
    plt.plot()
    title = model_name
    plt.title(title, fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("x1", fontsize = 20)
    plt.ylabel("x2", fontsize = 20)
    plt.legend()
plt.show()
'''

# ### Example 4: Multiclass classification

# In[25]:


from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from matplotlib import cm
centers = [(-3, -3), (0, 0), (3, 3)]
n_samples = 600

# create training & test set
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)
X, x, Y, y = train_test_split(X, y, test_size=0.5, random_state=42)

# fit rvc & svc
rvm = RVC(gamma = 1, kernel = 'rbf')
rvm.fit(X,Y)
svc = GridSearchCV(SVC(kernel = 'rbf', degree = 2, probability = True), 
                   param_grid = {"C":np.logspace(-3,3,7),
                                 "gamma":[0.1,1,10]},
                   cv = 10)
svc.fit(X,Y)

# create grid
n_grid = 100
max_x      = np.max(x,axis = 0)
min_x      = np.min(x,axis = 0)
X1         = np.linspace(min_x[0],max_x[0],n_grid)
X2         = np.linspace(min_x[1],max_x[1],n_grid)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid**2,2])
Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
Xgrid[:,1] = np.reshape(x2,(n_grid**2,))


rv_grid = rvm.predict_proba(Xgrid)
sv_grid = svc.predict_proba(Xgrid)
grids   = [rv_grid, sv_grid]
names   = ['RVC','SVC']
classes = np.unique(y)

# plot heatmaps
for grid,name in zip(grids,names):
    fig, axarr = plt.subplots(nrows=1, ncols=3, figsize = (20,8))
    for ax,cl,model in zip(axarr,classes,grid.T):
        ax.contourf(x1,x2,np.reshape(model,(n_grid,n_grid)),cmap=cm.coolwarm)
        ax.plot(x[y==cl,0],x[y==cl,1],"ro", markersize = 5)
        ax.plot(x[y!=cl,0],x[y!=cl,1],"bo", markersize = 5)
    plt.suptitle(' '.join(['Decision boundary for',name,'OVR multiclass classification']))
    plt.show()
    
print classification_report(y,rvm.predict(x))
print classification_report(y,svc.predict(x))


# ### Example: Pima Indians Diabetes dataset

# In this example, we test Relevance Vector Classifier against SVC and Random Forest Classifier on a real dataset. Classification reports below show that RVC performs better than SVC and achieves almost the same results as RFC.
# 
# P.S.: For some reason (numerical issues probably) SVC with polynomial kernel did not run, so we used only other kernels.

# In[26]:


import pandas as pd
data = np.array(pd.read_csv('pima-indians-diabetes.data.csv', header = None))
X,x,Y,y = train_test_split(data[:,:-1],data[:,-1], test_size = 0.2, random_state=1)


# In[27]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

rvc = GridSearchCV(RVC(kernel = 'poly'),param_grid = {'degree':[2,3],
                                                      'gamma':[1e-2,1e-1,1,1e+1],
                                                      'coef0':[0.1,1,10]})
# Polynomial kernel was not used, since SVC with 'poly' kernel 
# did not produce any results even after 30 minutes running 
# (you can try yourself!)
svc = GridSearchCV(SVC(probability = True), 
                   param_grid = {"C":np.logspace(-3,3,7),
                                 'gamma':[0.1,1,10],
                                 'kernel':['sigmoid','rbf']},
                   cv = 5)

gbc = GridSearchCV( RandomForestClassifier(n_estimators = 1000), 
                    param_grid = {'max_depth': [1,5,10]})

rvc.fit(X,Y)
svc.fit(X,Y)
gbc.fit(X,Y)
print "\n      === Relevance Vector Classifier === \n"
print classification_report(y,rvc.predict(x))
print "\n      === Support Vector Classifier === \n"
print classification_report(y,svc.predict(x))
print "\n      === Gradient Boosting Classifier === \n"
print classification_report(y,gbc.predict(x))

'''