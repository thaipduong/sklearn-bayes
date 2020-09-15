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

from sklearn.utils.estimator_checks import check_estimator
from skbayes.rvm_ard_models import RegressionARD3,ClassificationARD3,RVR3,RVC4, RVSet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
#from mpl_toolkits import mplot3d
from sklearn.metrics import mean_squared_error


from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import classification_report
import pickle
''
# Parameters
n = 1000
test_proportion = 0.1
RUN = 22
filename = "laser_samples_seq_25pos50negres"
# create dataset & split into train/test parts
#Xx,Yy   = make_circles(n_samples = n, noise = 0.2, random_state = 1)

#uninflated_laser_data = np.load("/home/erl/repos/sklearn-bayes/data/laser_samples_uninflated.npz")
#uninflated_laser_Xx = uninflated_laser_data['points']*0.5
#uninflated_laser_Yy = uninflated_laser_data['labels']
#uninflated_laser_Yy[uninflated_laser_Yy < 0] = 0

laser_data = np.load("/home/erl/repos/sklearn-bayes/data/"+ filename + ".npz", allow_pickle=True,  encoding='latin1')
res = 0.25
label_seq = laser_data['label_seg']
point_seq = laser_data['point_seq']
rvm = RVC4(n_iter = 100, kernel = 'rbf', gamma =0.5)
#fig, ax = plt.subplots(figsize=(12,6))
#fig2, ax2 = plt.subplots(figsize=(12,6))
fig = plt.figure(figsize=(12,12))
fig2 = plt.figure(figsize=(12,12))
ax = fig2.add_subplot(111)
ax2 = fig.add_subplot(111)
d = 0
while (d  < len(label_seq)):#[0, 10]: #len(label_seq)):
    if d % 40 != 0 and d < len(label_seq) - 1 :
        d = d + 1
        continue
    ax.clear()
    ax2.clear()
    #ax2.set_xlim(-10, 27)
    #ax.set_xlim(-10, 27)
    #ax2.set_ylim(-5, 12)
    #ax.set_ylim(-5, 12)
    print("d = ", d)
    laser_Yy = np.array(label_seq[d])
    laser_Xx = np.array(point_seq[d])
    laser_Yy[laser_Yy < 0] = 0


    laser_Yy_pos = laser_Yy[laser_Yy > 0]
    laser_Xx_pos = laser_Xx[laser_Yy > 0, :]
    pos_portion = sum(laser_Yy[laser_Yy > 0])
    neg_portion = len(laser_Yy) - pos_portion
    r = max(int(neg_portion/pos_portion), 0)

    #for i in range(r):
    #    laser_Xx = np.vstack((laser_Xx, laser_Xx_pos))
    #    laser_Yy = np.hstack((laser_Yy, laser_Yy_pos))


    Xx = np.array([ laser_Xx[i,:]*0.25 if laser_Yy[i] > 0 else laser_Xx[i,:]*0.5 for i in range(len(laser_Yy))])
    Yy = laser_Yy




    #data = np.load("./XY2.npz")
    #Xx = data['X']
    #Yy = data['Y']
    X,x,Y,y = train_test_split(Xx,Yy,test_size = test_proportion,
                                     random_state = 2)

    from sklearn.utils import shuffle
    ind_list = [i for i in range(len(Yy))]
    shuffle(ind_list)
    X = Xx[ind_list,:]
    Y = Yy[ind_list]


    # train rvm

    t1 = time.time()
    rvm.fit(X,Y)
    #rvm.fit(X,Y)
    t2 = time.time()
    rvm_time = t2 - t1
    print("RVC time:" + str(rvm_time))
    rvecs = np.sum(rvm.active_[0] == True)
    rvm_message = " ====  RVC: time {0}, relevant vectors = {1} \n".format(rvm_time, rvecs)
    print(rvm_message)
    y_hat = rvm.predict(x)
    print(classification_report(y, y_hat))
    #print(rvm.sigma_[0].shape)
    print(len(rvm.relevant_vectors_[0]))


    ax.plot(X[Y == 0, 0], X[Y == 0, 1], "bo", markersize=3)
    ax.plot(X[Y == 1, 0], X[Y == 1, 1], "ro", markersize=3)
    #plt.pause(1)
    #ax.clear()
    svrv = np.array(rvm.all_rv_X)
    w = rvm.Mn
    pos_rv = ax2.scatter(svrv[w>0, 0], svrv[w>0, 1], color = 'r', s = 60)
    neg_rv = ax2.scatter(svrv[w < 0, 0], svrv[w < 0, 1], color = 'b', s = 60)
    plt.pause(0.1)
    #fig.savefig("/home/erl/repos/sklearn-bayes/figs/data_rvs"+str(d)+".png", bbox_inches='tight', pad_inches=0)
    #fig2.savefig("/home/erl/repos/sklearn-bayes/figs/rvs"+str(d)+".png", bbox_inches='tight', pad_inches=0)
    print("######################################")
    print("d = ", d)
    d = d + 1
plt.show()

# Saved trained model
#RVMMap = RVSet(rvm.relevant_vectors_, rvm.coef_[:,rvm.active_[0]],  rvm.sigma_, rvm.classes_, fixed_intercept=rvm.intercept_, kernel = 'rbf', gamma = 2)


#y_hat2 = rvm.predict(x)


# create grid
#n_grid = 500
#max_x      = 1.2*np.max(X,axis = 0)
#min_x      = 1.2*np.min(X,axis = 0)
#X1         = np.linspace(min_x[0],max_x[0],n_grid)
#X2         = np.linspace(min_x[1],max_x[1],n_grid)

max_x      = np.array([65,26])#10*np.max(X,axis = 0)
min_x      = np.array([-20,-5])#10*np.min(X,axis = 0)

n_grid_x = int((65+20)/0.25) + 1
n_grid_y = int((26+5)/0.25) + 1
X1         = np.linspace(min_x[0],max_x[0],n_grid_x)
X2         = np.linspace(min_x[1],max_x[1],n_grid_y)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid_x*n_grid_y,2])
Xgrid[:,0] = np.reshape(x1,(n_grid_x*n_grid_y,))
Xgrid[:,1] = np.reshape(x2,(n_grid_x*n_grid_y,))

rv_grid, var_grid, _, _ = rvm.predict_proba(Xgrid)
rv_grid = rv_grid[:,1]

# Illustrate collision checking
plt.figure()
levels = np.arange(0,1, 0.0005)
plt.contourf(X1, X2, np.reshape(rv_grid, (n_grid_y, n_grid_x)), cmap='coolwarm')


cb  = plt.colorbar()

#cb.set_ticks(0.075 + np.array([0.0, 0.15, 0.3, 0.45, 0.60, 0.75, 0.90]))
cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
cb.ax.tick_params(labelsize=20)
#plt.savefig("/home/erl/repos/sklearn-bayes/figs/rosbag_rvmmap.pdf", bbox_inches='tight', pad_inches=0)

plt.show()


f = open("/home/erl/repos/sklearn-bayes/data/results/rosbag_rvm"+str(RUN)+ filename + ".pkl","wb")
pickle.dump(rvm,f)
f.close()
print("Saved file")