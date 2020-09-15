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
from skbayes.rvm_ard_models import RegressionARD3,ClassificationARD3,RVR3,RVC5, RVSet
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

# !/usr/bin/env python
# coding: utf-8

# # Bayesian Hilbert Maps
#
# ##### Email the authors to get the latest (much faster) code.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import pickle
import time


# get_ipython().run_line_magic('matplotlib', 'inline')


# Training data (pre-processed) file format:
#
#     Column 0: Time step (0,1,2,3,...)
#     Column 1: Logitide x1
#     Column 2: Latitude x2
#     Column 3: Class labels (0 or 1)
#
# $N$ data points (rows).

# In[2]:


def load_parameters(case):
    parameters = {'gilad2': ('../../Datasets/gilad2.csv',
                             (5, 5),  # x1 and x2 resolutions for positioning hinge points
                             (-300, 300, 0, 300),  # area to be mapped [x1_min, x1_max, x2_min, x2_max]
                             1,  # N/A
                             0.3,  # threshold for filtering data
                             0.075  # gamma: kernel parameter
                             ),

                  'intel': \
                      ('../../Datasets/intel.csv',
                       (0.5, 0.5),  # x1 and x2 resolutions for positioning hinge points
                       (-20, 20, -25, 10),  # area to be mapped [x1_min, x1_max, x2_min, x2_max]
                       1,  # N/A
                       0.01,  # threshold for filtering data
                       6.71  # gamma: kernel parameter
                       ),

                  }

    return parameters[case]


# In[3]:
# run = [8,9]
# gamma_arr = [6.71, 6.71]
# r_mul = [0.5, 0.25]
# pos_neg_rescale = [True, True, True]
# k_nearest = [50, 50]

# run = [11]
# gamma_arr = [6.71]
# r_mul = [0.1 ]
# pos_neg_rescale = [False]
# use_diff = [True]
# k_nearest = [25]
# max_iter = [25]
# pos_res = [0.2]
# neg_res = [0.4]

# run = [12]
# gamma_arr = [6.71]
# r_mul = [0.1]
# pos_neg_rescale = [False]
# use_diff = [True]
# k_nearest = [25]
# max_iter = [10]
# pos_res = [0.15]
# neg_res = [0.3]

# run = [13]
# gamma_arr = [6.71]
# r_mul = [1.0]
# pos_neg_rescale = [True]
# use_diff = [True]
# k_nearest = [25]
# max_iter = [10]
# pos_res = [0.15]
# neg_res = [0.3]

# run = [14]
# gamma_arr = [3.0]
# r_mul = [1.0]
# pos_neg_rescale = [True]
# use_diff = [True]
# k_nearest = [25]
# max_iter = [10]
# pos_res = [0.15]
# neg_res = [0.3]

# run = [15]
# gamma_arr = [6.71]
# r_mul = [1.0]
# pos_neg_rescale = [True]
# use_diff = [True]
# k_nearest = [25]
# max_iter = [10]
# pos_res = [0.15]
# neg_res = [0.3]

# run = [16]
# gamma_arr = [6.71]
# r_mul = [0.5]
# pos_neg_rescale = [True]
# use_diff = [True]
# k_nearest = [50]
# max_iter = [10]
# pos_res = [0.25]
# neg_res = [0.25]

# run = [17]
# gamma_arr = [6.71]
# r_mul = [1.0]
# pos_neg_rescale = [True]
# use_diff = [True]
# k_nearest = [25]
# max_iter = [10]
# pos_res = [0.15]
# neg_res = [0.3]

# run = [18]
# gamma_arr = [6.71]
# r_mul = [0.5]
# pos_neg_rescale = [True]
# use_diff = [True]
# k_nearest = [25]
# max_iter = [10]
# pos_res = [0.2]
# neg_res = [0.4]

# run = [19]
# gamma_arr = [6.71]
# r_mul = [0.5]
# pos_neg_rescale = [True]
# use_diff = [True]
# k_nearest = [25]
# max_iter = [10]
# pos_res = [0.2]
# neg_res = [0.2]

# run = [20]
# gamma_arr = [3.0]
# r_mul = [0.5]
# pos_neg_rescale = [True]
# use_diff = [True]
# k_nearest = [25]
# max_iter = [10]
# pos_res = [0.2]
# neg_res = [0.2]

# run = [21]
# gamma_arr = [6.71]
# r_mul = [0.5]
# pos_neg_rescale = [False]
# use_diff = [True]
# k_nearest = [25]
# max_iter = [200]
# pos_res = [0.2]
# neg_res = [0.2]

# run = [22]
# gamma_arr = [6.71]
# r_mul = [0.5]
# pos_neg_rescale = [True]
# use_diff = [True]
# k_nearest = [25]
# max_iter = [10]
# pos_res = [0.2]
# neg_res = [0.4]

# run = [23]
# gamma_arr = [6.71]
# r_mul = [0.5]
# pos_neg_rescale = [True]
# use_diff = [True]
# k_nearest = [25]
# max_iter = [10]
# pos_res = [0.2]
# neg_res = [0.4]

# run = [24]
# gamma_arr = [6.71]
# r_mul = [0.5]
# pos_neg_rescale = [True]
# use_diff = [True]
# k_nearest = [25]
# max_iter = [10]
# pos_res = [0.2]
# neg_res = [0.4]
#
# run = [25]
# gamma_arr = [6.71]
# r_mul = [0.5]
# pos_neg_rescale = [True]
# use_diff = [True]
# k_nearest = [25]
# max_iter = [10]
# pos_res = [0.2]
# neg_res = [0.2]

run = [26]
gamma_arr = [6.71]
r_mul = [0.5]
pos_neg_rescale = [True]
use_diff = [True]
k_nearest = [25]
max_iter = [10]
pos_res = [0.2]
neg_res = [0.2]

fn_train, cell_resolution, cell_max_min, skip, thresh, gamma = load_parameters('intel')

for iter in range(0,1):
    print("TEST RUN ", run[iter] )
    # Set up RVM model
    rvm = RVC5(n_iter = max_iter[iter], kernel = 'rbf', gamma =gamma_arr[iter], A_thres=10.0)


    # read data
    g = pd.read_csv(fn_train, delimiter=',').values
    print('shapes:', g.shape)
    X_train = np.float_(g[np.mod(g[:,0]+1, 10) != 0, 0:3])
    Y_train = np.float_(g[np.mod(g[:,0]+1, 10) != 0, 3][:, np.newaxis]).ravel()  # * 2 - 1
    X_test = np.float_(g[np.mod(g[:,0]+1, 10) == 0, 0:3])
    Y_test = np.float_(g[np.mod(g[:,0]+1, 10) == 0, 3][:, np.newaxis]).ravel()  # * 2 - 1
    print(len(g), len(Y_test), len(Y_train))
    print(sum(Y_train), sum(Y_test))
    # X_train = np.float_(g[np.mod(np.arange(len(g)), 10) != 0, 0:3])
    # Y_train = np.float_(g[np.mod(np.arange(len(g)), 10) != 0, 3][:, np.newaxis]).ravel()  # * 2 - 1
    #
    # X_test = np.float_(g[::10, 0:3])
    # Y_test = np.float_(g[::10, 3][:, np.newaxis]).ravel()  # * 2 - 1

    '''
    pl.figure()
    pl.scatter(X_train[:, 1], X_train[:, 2], c=Y_train, s=2)
    pl.title('Training data')
    pl.colorbar()
    pl.show()
    '''
    # query locations
    q_resolution = 0.5
    xx, yy = np.meshgrid(np.arange(cell_max_min[0], cell_max_min[1] - 1, q_resolution),
                         np.arange(cell_max_min[2], cell_max_min[3] - 1, q_resolution))
    X_query = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))

    # If there are N data points (rows),
    #
    #     X_train.shape = (N,3) #the three columns are (time_step, logitude, latitude)
    #     Y_train.shape = (N,) #labels are either 0 or 1
    #
    # If we want to query the occupancy probability of N_q positions (rows),
    #
    #     X_query.shape = (N_q,2) #the two columns are (logitude, latitude)

    # In[4]:
    pres = pos_res[iter]
    nres = neg_res[iter]
    t1 = time.time()
    max_t = len(np.unique(g[:, 0]))
    print("scans = ", max_t, "skip = ", skip)
    for ith_scan in range(0, max_t, skip):
        if np.mod(ith_scan+1,10) == 0:
            continue
        # extract data points of the ith scan
        ith_scan_indx = X_train[:, 0] == ith_scan
        print('{}th scan:\n  N={}'.format(ith_scan, np.sum(ith_scan_indx)))
        y_new = Y_train[ith_scan_indx]
        X_new = X_train[ith_scan_indx, 1:]
        if pres is not None and nres is not None:
            X_new[y_new > 0.5] = pres*np.around(X_new[y_new > 0.5]/pres)
            X_new_pos = np.unique(X_new[y_new > 0.5], axis=0)
            X_new_pos_int_dict = {}
            for k in range(len(X_new_pos)):
                X_new_pos_int_dict[(int(np.rint(X_new_pos[k,0]/pres)), int(np.rint(X_new_pos[k,1]/pres)))] = 1
            X_new[y_new < 0.5] = nres*np.around(X_new[y_new < 0.5] / nres)
            X_new_neg = np.unique(X_new[y_new < 0.5], axis = 0)
            X_new_neg_reduced = []
            for k in range(len(X_new_neg)):
                if (int(np.rint(X_new_neg[k,0]/pres)), int(np.rint(X_new_neg[k,1]/pres))) in X_new_pos_int_dict:
                    continue
                X_new_neg_reduced.append([X_new_neg[k,0], X_new_neg[k,1]])
            X_new_neg_reduced = np.array(X_new_neg_reduced)
            X_new_reduced = np.vstack((X_new_pos, X_new_neg_reduced))
            y_new_reduced = np.hstack((np.ones(len(X_new_pos), dtype=float), np.zeros(len(X_new_neg_reduced), dtype=float)))
        print("Reduced dataset size: ", len(y_new_reduced))
        if ith_scan == 0:
            # get all data for the first scan and initialize the model
            X, y = X_new_reduced, y_new_reduced
            #bhm_mdl = sbhm.SBHM(gamma=gamma, grid=None, cell_resolution=cell_resolution, cell_max_min=cell_max_min, X=X,
            #                    calc_loss=False)
        else:
            # information filtering
            #q_new, _, _, _ = rvm.predict_proba(X_new)
            #q_new = q_new[:, 1]
            #info_val_indx = np.absolute(q_new - y_new) > thresh
            #X, y = X_new[info_val_indx, :], y_new[info_val_indx]
            #print('  {:.2f}% points were used.'.format(X.shape[0] / X_new.shape[0] * 100))
            X, y = X_new_reduced, y_new_reduced

        # training

        rvm.fit(X, y, pos_neg_rescale = pos_neg_rescale[iter], r_mul = r_mul[iter], k_nearest = k_nearest[iter], use_diff=use_diff[iter])

        # query the model
        rv_grid, var_grid, _, _ = rvm.predict_proba(X_query)
        Y_query = rv_grid[:,1]
        svrv = np.array(rvm.all_rv_X)
        rvy = np.array(rvm.all_rv_y)
        w = rvm.Mn
        # plot
        pl.figure(figsize=(25, 5))
        pl.subplot(141)
        ones_ = np.where(y > 0.5)
        zeros_ = np.where(y < 0.5)
        pl.scatter(X[ones_, 0], X[ones_, 1], c='r', cmap='jet', s=5, edgecolors='')
        pl.scatter(X[zeros_, 0], X[zeros_, 1], c='b', cmap='jet', s=5, edgecolors='')
        pl.title('Data points at t={}'.format(ith_scan))
        pl.xlim([cell_max_min[0], cell_max_min[1]]);
        pl.ylim([cell_max_min[2], cell_max_min[3]])
        pl.subplot(142)
        ones_ = np.where(y == 1)
        pos_rv = pl.scatter(svrv[w > 0, 0], svrv[w > 0, 1], c='r', cmap='jet', s=5, edgecolors='')
        neg_rv = pl.scatter(svrv[w < 0, 0], svrv[w < 0, 1], c='b', cmap='jet', s=5, edgecolors='')
        pl.title('rvs weights at t={}'.format(ith_scan))
        pl.xlim([cell_max_min[0], cell_max_min[1]]);
        pl.ylim([cell_max_min[2], cell_max_min[3]])
        pl.subplot(143)
        ones_ = np.where(y == 1)
        pos_rv = pl.scatter(svrv[rvy > 0.5, 0], svrv[rvy > 0.5, 1], c='r', cmap='jet', s=5, edgecolors='')
        neg_rv = pl.scatter(svrv[rvy < 0.5, 0], svrv[rvy < 0.5, 1], c='b', cmap='jet', s=5, edgecolors='')
        pl.title('rvs label at t={}'.format(ith_scan))
        pl.xlim([cell_max_min[0], cell_max_min[1]]);
        pl.ylim([cell_max_min[2], cell_max_min[3]])
        pl.subplot(144)
        pl.title('Our map at t={}'.format(ith_scan))
        pl.scatter(X_query[:, 0], X_query[:, 1], c=Y_query, cmap='jet', s=10, marker='8', edgecolors='')
        # pl.imshow(Y_query.reshape(xx.shape))
        pl.clim(0, 1)
        pl.colorbar()
        pl.xlim([cell_max_min[0], cell_max_min[1]]);
        pl.ylim([cell_max_min[2], cell_max_min[3]])
        pl.savefig('../../Outputs/test' + str(run[iter]) + '/imgs/step' + str(ith_scan) + '.png', bbox_inches='tight')
        # pl.show()
        ################################################################# query the model using lambda max
        rv_grid, _, _, _ = rvm.predict_proba_approx(X_query)
        Y_query = rv_grid[:, 1]
        svrv = np.array(rvm.all_rv_X)
        rvy = np.array(rvm.all_rv_y)
        w = rvm.Mn
        # plot
        pl.figure(figsize=(25, 5))
        pl.subplot(141)
        ones_ = np.where(y > 0.5)
        zeros_ = np.where(y < 0.5)
        pl.scatter(X[ones_, 0], X[ones_, 1], c='r', cmap='jet', s=5, edgecolors='')
        pl.scatter(X[zeros_, 0], X[zeros_, 1], c='b', cmap='jet', s=5, edgecolors='')
        pl.title('Data points at t={}'.format(ith_scan))
        pl.xlim([cell_max_min[0], cell_max_min[1]]);
        pl.ylim([cell_max_min[2], cell_max_min[3]])
        pl.subplot(142)
        ones_ = np.where(y == 1)
        pos_rv = pl.scatter(svrv[w > 0, 0], svrv[w > 0, 1], c='r', cmap='jet', s=5, edgecolors='')
        neg_rv = pl.scatter(svrv[w < 0, 0], svrv[w < 0, 1], c='b', cmap='jet', s=5, edgecolors='')
        pl.title('rvs weights at t={}'.format(ith_scan))
        pl.xlim([cell_max_min[0], cell_max_min[1]]);
        pl.ylim([cell_max_min[2], cell_max_min[3]])
        pl.subplot(143)
        ones_ = np.where(y == 1)
        pos_rv = pl.scatter(svrv[rvy > 0.5, 0], svrv[rvy > 0.5, 1], c='r', cmap='jet', s=5, edgecolors='')
        neg_rv = pl.scatter(svrv[rvy < 0.5, 0], svrv[rvy < 0.5, 1], c='b', cmap='jet', s=5, edgecolors='')
        pl.title('rvs label at t={}'.format(ith_scan))
        pl.xlim([cell_max_min[0], cell_max_min[1]]);
        pl.ylim([cell_max_min[2], cell_max_min[3]])
        pl.subplot(144)
        pl.title('Our map at t={}'.format(ith_scan))
        pl.scatter(X_query[:, 0], X_query[:, 1], c=Y_query, cmap='jet', s=10, marker='8', edgecolors='')
        # pl.imshow(Y_query.reshape(xx.shape))
        pl.clim(0, 1)
        pl.colorbar()
        pl.xlim([cell_max_min[0], cell_max_min[1]]);
        pl.ylim([cell_max_min[2], cell_max_min[3]])
        pl.savefig('../../Outputs/test' + str(run[iter]) + '/imgs/step' + str(ith_scan) + '_approx.png', bbox_inches='tight')
        pl.close("all")

    t2 = time.time()
    traintime = t2 - t1
    print("RVC time:" + str(traintime))
    f = open("../../Outputs/test" + str(run[iter]) + "/rvm_model.pkl", "wb")
    pickle.dump(rvm, f)
    f.close()

    print("Saved file")
    # In[ ]:




