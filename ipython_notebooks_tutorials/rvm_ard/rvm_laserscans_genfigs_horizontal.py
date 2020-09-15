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
from skbayes.rvm_ard_models import RegressionARD2,ClassificationARD2,RVR2,RVC2
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
#from mpl_toolkits import mplot3d
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
plt.rcParams['text.usetex'] = True
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['ps.fonttype'] = 42
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import classification_report

''
# Parameters
n = 1000
test_proportion = 0.1

# create dataset & split into train/test parts
#Xx,Yy   = make_circles(n_samples = n, noise = 0.2, random_state = 1)

uninflated_laser_data = np.load("/home/erl/repos/sklearn-bayes/data/laser_samples_uninflated.npz")
uninflated_laser_Xx = uninflated_laser_data['points']*0.5
uninflated_laser_Yy = uninflated_laser_data['labels']
uninflated_laser_Yy[uninflated_laser_Yy < 0] = 0

laser_data = np.load("/home/erl/repos/sklearn-bayes/data/laser_samples_good.npz")
laser_Xx = laser_data['points']
laser_Yy = laser_data['labels']
laser_Yy[laser_Yy < 0] = 0



laser_Yy_pos = laser_Yy[laser_Yy > 0]
laser_Xx_pos = laser_Xx[laser_Yy > 0, :]
pos_portion = sum(laser_Yy[laser_Yy > 0])
neg_portion = len(laser_Yy) - pos_portion
#r = max(int(neg_portion/pos_portion), 0)

#for i in range(r):
#    laser_Xx = np.vstack((laser_Xx, laser_Xx_pos))
#    laser_Yy = np.hstack((laser_Yy, laser_Yy_pos))


Xx = laser_Xx*0.5
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
rvm = RVC2(n_iter = 100, kernel = 'rbf', gamma = 2.0)
t1 = time.time()
rvm.fit(X,Y)
t2 = time.time()
rvm_time = t2 - t1
print("RVC time:" + str(rvm_time))

rvecs = np.sum(rvm.active_[0]==True)
rvm_message = " ====  RVC: time {0}, relevant vectors = {1} \n".format(rvm_time,rvecs)
print(rvm_message)
y_hat = rvm.predict(x)
print(classification_report(y,y_hat))

# create grid
#n_grid = 500
#max_x      = 1.2*np.max(X,axis = 0)
#min_x      = 1.2*np.min(X,axis = 0)
#X1         = np.linspace(min_x[0],max_x[0],n_grid)
#X2         = np.linspace(min_x[1],max_x[1],n_grid)
n_grid = 200
max_x      = 1.5*np.max(X,axis = 0) - np.array([0.5, 1])
min_x      = 1.5*np.min(X,axis = 0) - np.array([2, 4])


min_x[1] = min_x[1]+1
max_x[1] = max_x[1]-2

X1         = np.linspace(min_x[0],max_x[0],n_grid)
X2         = np.linspace(min_x[1],max_x[1],n_grid)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid**2,2])
Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
Xgrid[:,1] = np.reshape(x2,(n_grid**2,))

rv_grid, var_grid, _, _ = rvm.predict_proba(Xgrid)
THRESHOLD = -0.01#0.6#-0.01#-0.01
threshold_prob = norm.cdf(THRESHOLD)

########################################################################################################################
#A = np.array([-1.1, 0.9])
#B = np.array([0.0, -2.0])
A = np.array([0.0, 0.0])
B = np.array([3.5, -0.8])
#A = np.array([0.0, 0.0])
#B = np.array([-7.5, 2.5])
#A = np.array([0.0, 0.0])
#B = np.array([3.5, 6.5])

line_seg_x = np.linspace(A[0], B[0], 100)
line_seg_y = np.linspace(A[1], B[1], 100)
line_seg = np.vstack((line_seg_x, line_seg_y)).transpose()

upper_line, upper_line2, upper_line3, upper_line4, _ = rvm.predict_upperbound(line_seg)
rv_line, var_line, num, denom = rvm.predict_proba(line_seg)
upper_line5 = rvm.predict_upperbound_line(line_seg, A)
line_seg_rev = np.flipud(line_seg)
upper_line6 = rvm.predict_upperbound_line(line_seg_rev, B)
upper_line6 = np.flip(upper_line6)
upper_line7 = np.minimum(upper_line5, upper_line6)



plt.figure(figsize=(12, 4))
#plt.plot(rv_line[:,1] - 0.5, label="GT")
a = np.where(upper_line7 >0)[0]


gt = num - THRESHOLD*denom
t = np.linspace(0,1, 100)
plt.plot(t, gt, label="$G_1(x)$", linewidth=4)
plt.plot(t, upper_line, label="$G_2(x)$", linestyle="dashed", linewidth=4)
#plt.plot(upper_line2, label="Fastron bound", linestyle = "dotted")
plt.plot(t, upper_line3, label="$G_3(x)$",  linestyle="dotted", linewidth=4)
#plt.plot(upper_line7, label="triangle bound",  linestyle="dashdot")
#plt.plot(upper_line6, label="triangle bound",  linestyle="dashdot")
plt.plot(t, np.zeros(100), label="zero",  linestyle="dashdot", linewidth=4)
ax = plt.axes()
# Setting the background color
ax.set_facecolor("lightgrey")
plt.legend(fontsize = 20)
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
plt.savefig("/home/erl/repos/sklearn-bayes/figs/bounds_free..eps", bbox_inches='tight', pad_inches=0)


##################################################################

#A = np.array([0.0, 0.0])
#B = np.array([3.5, -0.8])
#A = np.array([0.0, 0.0])
#B = np.array([-7.5, 2.5])
A = np.array([0.0, 0.0])
B = np.array([3.5, 6.5])

line_seg_x = np.linspace(A[0], B[0], 100)
line_seg_y = np.linspace(A[1], B[1], 100)
line_seg = np.vstack((line_seg_x, line_seg_y)).transpose()

upper_line, upper_line2, upper_line3, upper_line4, _ = rvm.predict_upperbound(line_seg)
rv_line, var_line, num, denom = rvm.predict_proba(line_seg)
upper_line5 = rvm.predict_upperbound_line(line_seg, A)
line_seg_rev = np.flipud(line_seg)
upper_line6 = rvm.predict_upperbound_line(line_seg_rev, B)
upper_line6 = np.flip(upper_line6)
upper_line7 = np.minimum(upper_line5, upper_line6)



plt.figure(figsize=(12, 4))
#plt.plot(rv_line[:,1] - 0.5, label="GT")
a = np.where(upper_line7 >0)[0]


gt = num - THRESHOLD*denom
t = np.linspace(0,1, 100)
plt.plot(t, gt, label="$G_1(x)$", linewidth=4)
plt.plot(t, upper_line, label="$G_2(x)$", linestyle="dashed", linewidth=4)
#plt.plot(upper_line2, label="Fastron bound", linestyle = "dotted")
plt.plot(t, upper_line3, label="$G_3(x)$",  linestyle="dotted", linewidth=4)
#plt.plot(upper_line7, label="triangle bound",  linestyle="dashdot")
#plt.plot(upper_line6, label="triangle bound",  linestyle="dashdot")
plt.plot(t, np.zeros(100), label="zero",  linestyle="dashdot", linewidth=4)
ax = plt.axes()
# Setting the background color
ax.set_facecolor("lightgrey")
plt.legend(fontsize = 20)
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
plt.savefig("/home/erl/repos/sklearn-bayes/figs/bounds_colliding.eps", bbox_inches='tight', pad_inches=0)


########################################################################################################################

rv_grid = rv_grid[:,1]
upper_grid, upper_grid2, upper_grid3, upper_grid4, upper_grid5 = rvm.predict_upperbound(Xgrid)
#plt.show()



plt.figure(figsize = (10,6.67))

plt.plot(X[Y==1,0],X[Y==1,1],"rs", markersize = 4, label = 'occupied')
plt.plot(X[Y==0,0],X[Y==0,1],"bo", markersize = 4, label = 'free')
ax = plt.axes()
# Setting the background color
ax.set_facecolor("lightgrey")
plt.legend(fontsize = 25)
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
plt.xlim(min_x[0],max_x[0])#(-10.0,max_x[0])#(min_x[0],max_x[0])
#plt.ylim(min_x[1]+4,max_x[1]-2)
plt.ylim(-6.0, max_x[1])#(min_x[1],max_x[1])
print("ratio =", (max_x[1] + 6.0)/(max_x[0] - min_x[0]), (max_x[1] + 6.0))
plt.savefig("/home/erl/repos/sklearn-bayes/figs/inflated_samples.eps", bbox_inches='tight', pad_inches=0)
#plt.show()
''''''
plt.figure(figsize = (10,6.67))
plt.plot(uninflated_laser_Xx[uninflated_laser_Yy==0,0],uninflated_laser_Xx[uninflated_laser_Yy==0,1],"bo", markersize = 4, label = 'free')
plt.plot(uninflated_laser_Xx[uninflated_laser_Yy==1,0],uninflated_laser_Xx[uninflated_laser_Yy==1,1],"rs", markersize = 4, label = 'occupied')
plt.legend(fontsize = 25)
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
ax = plt.axes()
# Setting the background color
ax.set_facecolor("lightgrey")
plt.xlim(min_x[0],max_x[0])
#plt.ylim(min_x[1]+4,max_x[1]-2)
plt.ylim(-6.0, max_x[1])#(min_x[1],max_x[1])(min_x[1],max_x[1])
plt.savefig("/home/erl/repos/sklearn-bayes/figs/uninflated_samples.eps", bbox_inches='tight', pad_inches=0)



''''''
plt.figure(figsize = (10,10.0))
levels = np.arange(0,1, 0.0005)
rv_grid[rv_grid > 0.95] = 1.0
rv_grid[rv_grid < 0.05] = 0.0
plt.contourf(X1, X2, np.reshape(rv_grid, (n_grid, n_grid)),20, cmap='coolwarm')

#bounds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#norm = plt.colors.Normalize(vmin=0, vmax=1)

cb  = plt.colorbar(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], orientation="horizontal", pad=0.05)
cb.ax.tick_params(labelsize=20)
#cb.set_ticks(0.075 + np.array([0.0, 0.15, 0.3, 0.45, 0.60, 0.75, 0.90]))
#cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
#cb.ax.tick_params(labelsize=20)
plt.contour(X1, X2, np.reshape(rv_grid, (n_grid, n_grid)), levels=[threshold_prob], cmap="Greys_r")
svrv = rvm.relevant_vectors_[0]
point_label = "relevant vec."



cm = ['b' if cw < 0 else 'r' for cw in rvm.corrected_weights]
plt.plot(svrv[rvm.corrected_weights>0, 0], svrv[rvm.corrected_weights>0, 1], 'ro', markersize=8, label="pos. " + point_label)
plt.plot(svrv[rvm.corrected_weights < 0, 0], svrv[rvm.corrected_weights < 0, 1], 'bo', markersize=8,
             label="neg. " + point_label)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.xlabel("x1", fontsize=20)
#plt.ylabel("x2", fontsize=20)
plt.xlim(min_x[0],max_x[0])
#plt.ylim(min_x[1]+4,max_x[1]-2)
plt.ylim(min_x[1],max_x[1])#(-6.0, max_x[1])#(min_x[1],max_x[1])(min_x[1],max_x[1])
plt.legend(fontsize=25, loc=1)
plt.savefig("/home/erl/repos/sklearn-bayes/figs/rvm_models_horizontal.eps", bbox_inches='tight', pad_inches=0)
plt.show()


models  = [rv_grid, upper_grid, upper_grid2, upper_grid4, upper_grid5]
model_names = ["RVC", "upperbound", "upperbound_fastron", "upperbound AMGM", "compare bounds"]
line_width = 3
#plt.plot(rvm.relevant_vectors_,Y[rvm.active_],"co",markersize = 12,  label = "relevant vectors")
for model, model_name in zip(models, model_names):
    plt.figure(figsize = (10,10))
    levels = np.arange(0,1, 0.0005)
    #ax.contour3D(X1,X2,np.reshape(model,(n_grid,n_grid)), 100, cmap='coolwarm',
    #                   linewidth=0, antialiased=False)
    plt.contourf(X1, X2, np.reshape(model, (n_grid, n_grid)), 20, cmap='coolwarm')
    cb  = plt.colorbar(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], orientation="horizontal", pad=0.05)
    #cb  = plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    if model_name == "RVC":
        #cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
        #arr1 = plt.arrow(0, 0,  3.5, 6.5, head_width=0.5,
        #                 head_length=0.3, fc="xkcd:green", ec="xkcd:green", width = 0.2, label="colliding segment")
        #arr2 = plt.arrow(0, 0, 3.5, -0.8, head_width=0.5,
        #                 head_length=0.3, fc="xkcd:orange", ec="xkcd:orange", width=0.2, label="free segment")
        cs2 = plt.contour(X1, X2, np.reshape(upper_grid, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles="dashed", label="$G_2(x)$", linewidths=line_width)
        #plt.contour(X1, X2, np.reshape(upper_grid2, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles = "dotted", label="$G_2(x)$", linewidths=line_width)
        #plt.contour(X1, X2, np.reshape(upper_grid3, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
        #            linestyles="solid")
        cs3 = plt.contour(X1, X2, np.reshape(upper_grid4, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
                    linestyles="dashdot", label="$G_3(x)$", linewidths=line_width)
        cs1 = plt.contour(X1, X2, np.reshape(model, (n_grid, n_grid)), levels=[threshold_prob], linestyles = "dotted", cmap="Greys_r", label="$G_1(x)$", linewidths=line_width)
    else:
        plt.contour(X1, X2, np.reshape(model, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linewidths=line_width)


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
    pos_rv = plt.scatter(svrv[rvm.corrected_weights>0, 0], svrv[rvm.corrected_weights>0, 1], color = 'r', s = 60)
    neg_rv = plt.scatter(svrv[rvm.corrected_weights < 0, 0], svrv[rvm.corrected_weights < 0, 1], color = 'b', s = 60)

    print("All relevance vectors")
    for i in range(len(rvm.corrected_weights)):
        print(i, rvm.orig_weights[i], rvm.corrected_weights[i], svrv[i,:])
    #plt.plot(traj[:,0], traj[:,1],'go',markersize=4)
    # plt.plot()
    #plt.plot()
    #title = model_name
    #plt.title(title, fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.xlabel("x1", fontsize=20)
    # plt.ylabel("x2", fontsize=20)
    plt.xlim(min_x[0], max_x[0])
    # plt.ylim(min_x[1]+4,max_x[1]-2)
    plt.ylim(min_x[1], max_x[1])
    plt.legend(fontsize=25)
    #plt.xlabel("x1", fontsize = 20)
    #plt.ylabel("x2", fontsize = 20)

    if model_name == "RVC":
        h1, _ = cs1.legend_elements()
        h2, _ = cs2.legend_elements()
        h3, _ = cs3.legend_elements()
        plt.legend([pos_rv, neg_rv, h1[0], h2[0], h3[0]], \
                   ['pos. vec.', 'neg. vec.', '$G_1(x) = 0$', '$G_2(x) = 0$', '$G_3(x) = 0$'], loc = 2, fontsize=20)
        plt.savefig("/home/erl/repos/sklearn-bayes/figs/rvm_model_lines_horizontal.eps", bbox_inches='tight', pad_inches=0)
    else:
        plt.legend(fontsize=25, loc=2)



plt.figure(figsize = (10,10))
levels = np.arange(0,1, 0.0005)
#ax.contour3D(X1,X2,np.reshape(model,(n_grid,n_grid)), 100, cmap='coolwarm',
#                   linewidth=0, antialiased=False)
#model[model < threshold_prob] = threshold_prob
svrv = rvm.relevant_vectors_[0]
point_label = "relevant vecs"

plt.contourf(X1, X2, np.reshape(rv_grid, (n_grid, n_grid)), 20, cmap='coolwarm')
cb  = plt.colorbar(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], orientation="horizontal", pad=0.05)
# cb  = plt.colorbar()
cb.ax.tick_params(labelsize=20)
#arr1 = plt.arrow(0, 0, 3.5, 6.5, head_width=0.5,
#                 head_length=0.3, fc="xkcd:green", ec="xkcd:green", width=0.2, label="colliding segment")
#arr2 = plt.arrow(0, 0, 3.5, -0.8, head_width=0.5,
#                 head_length=0.3, fc="xkcd:orange", ec="xkcd:orange", width=0.2, label="free segment")
#cs2 = plt.contour(X1, X2, np.reshape(upper_grid, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
#                  linestyles="solid", label="$G_2(x)$", linewidths=line_width)
# plt.contour(X1, X2, np.reshape(upper_grid2, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles = "dotted", label="$G_2(x)$", linewidths=line_width)
cs2 = plt.contour(X1, X2, np.reshape(upper_grid3, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
            linestyles="dotted")
cs3 = plt.contour(X1, X2, np.reshape(upper_grid4, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
                  linestyles="dashdot", label="$G_3(x)$", linewidths=line_width)
cs4 = plt.contour(X1, X2, np.reshape(upper_grid5, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
                  linestyles="dashed", label="$G_3(x)$", linewidths=line_width)
cs1 = plt.contour(X1, X2, np.reshape(model, (n_grid, n_grid)), levels=[threshold_prob], linestyles="solid",
                  cmap="Greys_r", label="$G_1(x)$", linewidths=line_width)
pos_rv = plt.scatter(svrv[rvm.corrected_weights > 0, 0], svrv[rvm.corrected_weights > 0, 1], color='r', s=60)
neg_rv = plt.scatter(svrv[rvm.corrected_weights < 0, 0], svrv[rvm.corrected_weights < 0, 1], color='b', s=60)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.xlabel("x1", fontsize=20)
#plt.ylabel("x2", fontsize=20)
plt.xlim(min_x[0],max_x[0])
#plt.ylim(min_x[1]+4,max_x[1]-2)
plt.ylim(min_x[1],max_x[1])
plt.legend(fontsize=25)
h1, _ = cs1.legend_elements()
h2, _ = cs2.legend_elements()
h3, _ = cs3.legend_elements()
h4, _ = cs4.legend_elements()
plt.legend([pos_rv, neg_rv, h1[0], h2[0], h3[0],h4[0]], \
           ['pos. vec.', 'neg. vec.','$G_1(x) = 0,$ true boundary', '$G_3(x) = 0, n_1 = 5, n_2 = 1$', '$G_3(x) = 0, n_1 = 1, n_2 = 1$', '$G_3(x) = 0, n_1 = 1, n_2 = 1.5$'], loc=2,
           fontsize=17)
plt.savefig("/home/erl/repos/sklearn-bayes/figs/rvm_bounds_n1n2_horizontal.eps", bbox_inches='tight', pad_inches=0)
plt.show()
