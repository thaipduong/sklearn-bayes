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
import scipy
#from mpl_toolkits import mplot3d
plt.rcParams['text.usetex'] = True
#plt.rcParams['pdf.fonttype'] = 42
#plt.rcParams['ps.fonttype'] = 42
from sklearn.metrics import mean_squared_error


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
print ("RVC time:", str(rvm_time))

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
#A = np.array([-1.1, 0.9])
#B = np.array([0.0, -2.0])
A = np.array([-2.5, 1.0])
B = np.array([-2.5, 3.5])
#A = np.array([0.0, 0.0])
#B = np.array([-7.5, 2.5])
#A = np.array([0.0, 0.0])
#B = np.array([3.5, 6.5])

line_seg_x = np.linspace(A[0], B[0], 100)
line_seg_y = np.linspace(A[1], B[1], 100)
line_seg = np.vstack((line_seg_x, line_seg_y)).transpose()

upper_line, upper_line2, upper_line3, upper_line4, upper_line5 = rvm.predict_upperbound(line_seg)
rv_line, var_line, num, denom = rvm.predict_proba(line_seg)
upper_line5 = rvm.predict_upperbound_line(line_seg, A)
line_seg_rev = np.flipud(line_seg)
upper_line6 = rvm.predict_upperbound_line(line_seg_rev, B)
upper_line6 = np.flip(upper_line6)
upper_grid, upper_grid2, upper_grid3, upper_grid4, upper_grid5 = rvm.predict_upperbound(Xgrid)
upper_line7 = np.minimum(upper_line5, upper_line6)


#plt.plot(rv_line[:,1] - 0.5, label="GT")
a = np.where(upper_line7 >0)[0]

THRESHOLD = -0.01
UNKNOWN_PROB = -0.05
gt = num - THRESHOLD*denom

rv_grid = rv_grid[:,1]
'''
plt.figure(figsize=(12, 8))
plt.plot(t, gt, label="$G_1(x)$", linewidth=4)
plt.plot(t, upper_line, label="$G_2(x)$", linestyle="dashed", linewidth=4)
#plt.plot(upper_line2, label="Fastron bound", linestyle = "dotted")
plt.plot(t, upper_line4, label="$G_3(x)$",  linestyle="dotted", linewidth=4)
#plt.plot(upper_line7, label="triangle bound",  linestyle="dashdot")
#plt.plot(upper_line6, label="triangle bound",  linestyle="dashdot")
plt.plot(t, np.zeros(100), label="zero",  linestyle="dashdot", linewidth=4)
ax = plt.axes()
# Setting the background color
ax.set_facecolor("lightgrey")
plt.legend(fontsize = 20)
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
plt.savefig("/home/erl/repos/sklearn-bayes/figs/bounds_free.pdf", bbox_inches='tight', pad_inches=0)


#plt.show()

plt.figure(figsize = (12,8))

plt.plot(X[Y==1,0],X[Y==1,1],"rs", markersize = 4, label = 'occupied')
plt.plot(X[Y==0,0],X[Y==0,1],"bo", markersize = 4, label = 'free')
ax = plt.axes()
# Setting the background color
ax.set_facecolor("lightgrey")
plt.legend(fontsize = 20)
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
plt.xlim(min_x[0],max_x[0])
#plt.ylim(min_x[1]+4,max_x[1]-2)
plt.ylim(min_x[1],max_x[1])
plt.savefig("/home/erl/repos/sklearn-bayes/figs/inflated_samples.pdf", bbox_inches='tight', pad_inches=0)
#plt.show()
''''''
plt.figure(figsize = (12,8))
plt.plot(uninflated_laser_Xx[uninflated_laser_Yy==0,0],uninflated_laser_Xx[uninflated_laser_Yy==0,1],"bo", markersize = 4, label = 'free')
plt.plot(uninflated_laser_Xx[uninflated_laser_Yy==1,0],uninflated_laser_Xx[uninflated_laser_Yy==1,1],"rs", markersize = 4, label = 'occupied')
plt.legend(fontsize = 20)
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
ax = plt.axes()
# Setting the background color
ax.set_facecolor("lightgrey")
plt.xlim(min_x[0],max_x[0])
#plt.ylim(min_x[1]+4,max_x[1]-2)
plt.ylim(min_x[1],max_x[1])
plt.savefig("/home/erl/repos/sklearn-bayes/figs/uninflated_samples.pdf", bbox_inches='tight', pad_inches=0)



''''''
plt.figure(figsize = (12,8))
levels = np.arange(0,1, 0.0005)
plt.contourf(X1, X2, np.reshape(rv_grid, (n_grid, n_grid)), cmap='coolwarm')

#bounds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#norm = plt.colors.Normalize(vmin=0, vmax=1)
cb  = plt.colorbar()

#cb.set_ticks(0.075 + np.array([0.0, 0.15, 0.3, 0.45, 0.60, 0.75, 0.90]))
cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
cb.ax.tick_params(labelsize=20)
plt.contour(X1, X2, np.reshape(rv_grid, (n_grid, n_grid)), levels=[0.5], cmap="Greys_r")
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
plt.xlim(min_x[0]+3,max_x[0]-2)
#plt.ylim(min_x[1]+4,max_x[1]-2)
plt.ylim(min_x[1],max_x[1])
plt.legend(fontsize=20)
plt.savefig("/home/erl/repos/sklearn-bayes/figs/rvm_models.pdf", bbox_inches='tight', pad_inches=0)
plt.show()



models  = [rv_grid, upper_grid, upper_grid2, upper_grid4]
model_names = ["RVC", "upperbound", "upperbound_fastron", "upperbound AMGM"]
#plt.plot(rvm.relevant_vectors_,Y[rvm.active_],"co",markersize = 12,  label = "relevant vectors")
for model, model_name in zip(models, model_names):
    plt.figure(figsize = (12,8))
    levels = np.arange(0,1, 0.0005)
    #ax.contour3D(X1,X2,np.reshape(model,(n_grid,n_grid)), 100, cmap='coolwarm',
    #                   linewidth=0, antialiased=False)
    plt.contourf(X1, X2, np.reshape(model, (n_grid, n_grid)), cmap='coolwarm')
    cb  = plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    if model_name == "RVC":
        cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
        arr1 = plt.arrow(0, 0,  3.5, 6.5, head_width=0.3,
                         head_length=0.3, fc="xkcd:green", ec="xkcd:green", width = 0.1, label="colliding segment")
        arr2 = plt.arrow(0, 0, 3.5, -0.8, head_width=0.3,
                         head_length=0.3, fc="xkcd:orange", ec="xkcd:orange", width=0.1, label="free segment")
        cs2 = plt.contour(X1, X2, np.reshape(upper_grid, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles="dashed", label="$G_2(x)$")
        plt.contour(X1, X2, np.reshape(upper_grid2, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles = "dotted", label="$G_2(x)$")
        #plt.contour(X1, X2, np.reshape(upper_grid3, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
        #            linestyles="solid")
        cs3 = plt.contour(X1, X2, np.reshape(upper_grid4, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
                    linestyles="dashdot", label="$G_3(x)$")
        cs1 = plt.contour(X1, X2, np.reshape(model, (n_grid, n_grid)), levels=[0.4945], cmap="Greys_r", label="$G_1(x)$")
        #radius = rvm.get_radius(line_seg, A)
        #print("radius", radius)
        #ax = plt.axes()
        #circle = plt.Circle(A, radius, color='xkcd:orange', fill=False, label="free ball")
        #ax.add_artist(circle)

    else:
        plt.contour(X1, X2, np.reshape(model, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r")


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
    #plt.xlabel("x1", fontsize = 20)
    #plt.ylabel("x2", fontsize = 20)

    if model_name == "RVC":
        h1, _ = cs1.legend_elements()
        h2, _ = cs2.legend_elements()
        h3, _ = cs3.legend_elements()
        plt.legend([pos_rv, neg_rv, h1[0], h2[0], h3[0],arr1, arr2 ], \
                   ['pos. vec.', 'neg. vec.', '$G_1(x) = 0$', '$G_2(x) = 0$', '$G_3(x) = 0$', 'line 1', 'line 2'], loc = 2, fontsize=20,framealpha=0.5)
        plt.savefig("/home/erl/repos/sklearn-bayes/figs/rvm_model_lines.pdf", bbox_inches='tight', pad_inches=0)
    else:
        plt.legend(fontsize=20, loc=2)
plt.show()
'''

#################################################################################################################
########################## ILLUSTRATE COLLISION CHECKING LINE SEGMENTS #####################
''''''
plt.figure(figsize = (10,8))
#plt.figure()
levels = np.arange(0,1, 0.0005)
#ax.contour3D(X1,X2,np.reshape(model,(n_grid,n_grid)), 100, cmap='coolwarm',
#                   linewidth=0, antialiased=False)
plot_heatmap = True
prob_thres = (1+ scipy.special.erf(THRESHOLD))/2
print("THRESHOLD: ", prob_thres)
rv_grid [rv_grid < prob_thres] = prob_thres
#rv_grid [rv_grid > prob_thres] = 0.8
if plot_heatmap:
    plt.contourf(X1, X2, np.reshape(rv_grid, (n_grid, n_grid)), cmap='Greys')
    #cb  = plt.colorbar()
    #cb.ax.tick_params(labelsize=20)
    #cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
#arr1 = plt.arrow(0, 0,  3.5, 6.5, head_width=0.3,
#                 head_length=0.3, fc="xkcd:green", ec="xkcd:green", width = 0.1, label="colliding segment")
#arr2 = plt.arrow(0, 0, 3.5, -0.8, head_width=0.3,
#                 head_length=0.3, fc="xkcd:orange", ec="xkcd:orange", width=0.1, label="free segment")
#cs2 = plt.contour(X1, X2, np.reshape(upper_grid, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles="dashed", label="$G_2(x)$")
#cs2 = plt.contour(X1, X2, np.reshape(upper_grid, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles = "dotted", label="$G_2(x)$")
#plt.contour(X1, X2, np.reshape(upper_grid3, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
#            linestyles="solid")
cs3 = plt.contour(X1, X2, np.reshape(upper_grid3, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
                linestyles="dashdot", label="$G_3(x)$")
cs1 = plt.contour(X1, X2, np.reshape(rv_grid, (n_grid, n_grid)), levels=[prob_thres], cmap="Greys_r", label="$G_1(x)$")
#radius = rvm.get_radius(line_seg, A)
#print("radius", radius)
#ax = plt.axes()
#circle = plt.Circle(A, radius, color='xkcd:orange', fill=False, label="free ball")
#ax.add_artist(circle)

plot_rvs = False
if plot_rvs:
    svrv = None
    point_label = None
    svrv = rvm.relevant_vectors_[0]
    point_label = "relevant vecs"
    cm = ['b' if cw < 0 else 'r' for cw in rvm.corrected_weights]
    pos_rv = plt.scatter(svrv[rvm.corrected_weights>0, 0], svrv[rvm.corrected_weights>0, 1], color = 'r', s = 60)
    neg_rv = plt.scatter(svrv[rvm.corrected_weights < 0, 0], svrv[rvm.corrected_weights < 0, 1], color = 'b', s = 60)
    print("All relevance vectors")
    for i in range(len(rvm.corrected_weights)):
        print(i, rvm.orig_weights[i], rvm.corrected_weights[i], svrv[i,:])

v = 2.2*np.linalg.norm(B-A)
T = 20
for i in range(T):
    Bi = A + v*np.array([np.cos(2*i*np.pi/T), np.sin(2*i*np.pi/T)])
    arr1 = plt.arrow(A[0], A[1], Bi[0]-A[0], Bi[1]-A[1], head_width=0.25,
                     head_length=0.25, fc="xkcd:green", ec="xkcd:green", width=0.1, label="colliding segment")

for i in range(T):
    Bi = A + v*np.array([np.cos(2*i*np.pi/T), np.sin(2*i*np.pi/T)])
    intersect = rvm.predict_upperbound_line_check(line_seg, A, Bi,  n1 = 1.0, n2 = 1.0)
    print("intersect", intersect)
    if 1 > intersect > 0:
        #intersect = min(intersect,1.0)
        Bx = A + intersect*v*np.array([np.cos(2*i*np.pi/T), np.sin(2*i*np.pi/T)])
        x = plt.scatter(Bx[0], Bx[1], color = "xkcd:red", marker='x', s = 200, linewidths=10)


#plt.plot(traj[:,0], traj[:,1],'go',markersize=4)
# plt.plot()
#plt.plot()
#title = model_name
#plt.title(title, fontsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.xlabel("x1", fontsize = 20)
#plt.ylabel("x2", fontsize = 20)
h1, _ = cs1.legend_elements()
#h2, _ = cs2.legend_elements()
h3, _ = cs3.legend_elements()
plt.legend([h1[0], h3[0],arr1, x], \
           ['$G_1(x) = 0$', '$G_3(x) = 0$', 'line segments', 'intersection'], loc = 1, fontsize=22,framealpha=0.5)
plt.savefig("/home/erl/repos/sklearn-bayes/figs/rvm_model_checking_lines.eps", bbox_inches='tight', pad_inches=0)

plt.show()




# Solve this equation: (a1*t^2 + b1*t + c1 - x1)**2 + (a2*t^2 + b2*t + c2 - x2)**2 = r**2
# x1 = s1(curr_t), x2 = s2(curr_t)
def solve_next_t(a1, b1, c1, a2, b2, c2, curr_t, r):
    x1 = a1*curr_t**2 + b1*curr_t + c1
    x2 = a2 * curr_t ** 2 + b2 * curr_t + c2
    p = []
    p.append(a1**2 + a2**2)
    p.append(2*a1*b1 + 2*a2*b2)
    p.append(b1**2 + 2*a1*(c1-x1) + b2**2 + 2*a2*(c2-x2))
    p.append(2*b1*(c1-x1) + 2*b2*(c2-x2))
    p.append((c1-x1)**2 + (c2-x2)**2 - r**2)
    roots = np.roots(p)
    #print(roots)
    real_roots = roots.real[abs(roots.imag) < 1e-2]
    print(real_roots)
    min_t = -1
    for root in real_roots:
        if root > curr_t:
            if min_t < 0 or root < min_t:
                min_t = root
    return min_t


''''''
########################## ILLUSTRATE COLLISION CHECKING A CURVE #####################
plt.figure(figsize = (10,8))
#plt.figure()
levels = np.arange(0,1, 0.0005)
#ax.contour3D(X1,X2,np.reshape(model,(n_grid,n_grid)), 100, cmap='coolwarm',
#                   linewidth=0, antialiased=False)

plot_heatmap = True
prob_thres = (1+ scipy.special.erf(THRESHOLD))/2
print("THRESHOLD: ", prob_thres)
rv_grid [rv_grid < prob_thres] = prob_thres
#rv_grid [rv_grid > prob_thres] = 0.8
if plot_heatmap:
    plt.contourf(X1, X2, np.reshape(rv_grid, (n_grid, n_grid)), cmap='Greys')
    #cb  = plt.colorbar()
    #cb.ax.tick_params(labelsize=20)
    #cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])



#arr1 = plt.arrow(0, 0,  3.5, 6.5, head_width=0.3,
#                 head_length=0.3, fc="xkcd:green", ec="xkcd:green", width = 0.1, label="colliding segment")
#arr2 = plt.arrow(0, 0, 3.5, -0.8, head_width=0.3,
#                 head_length=0.3, fc="xkcd:orange", ec="xkcd:orange", width=0.1, label="free segment")
#cs2 = plt.contour(X1, X2, np.reshape(upper_grid, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles="dashed", label="$G_2(x)$")
cs2 = plt.contour(X1, X2, np.reshape(upper_grid, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles = "dotted", label="$G_2(x)$")
#plt.contour(X1, X2, np.reshape(upper_grid3, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
#            linestyles="solid")
cs3 = plt.contour(X1, X2, np.reshape(upper_grid3, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
                linestyles="dashdot", label="$G_3(x)$")
cs1 = plt.contour(X1, X2, np.reshape(rv_grid, (n_grid, n_grid)), levels=[prob_thres], cmap="Greys_r", label="$G_1(x)$")
#radius = rvm.get_radius(line_seg, A)
#print("radius", radius)
#ax = plt.axes()
#circle = plt.Circle(A, radius, color='xkcd:orange', fill=False, label="free ball")
#ax.add_artist(circle)

plot_rvs = False
if plot_rvs:
    svrv = None
    point_label = None
    svrv = rvm.relevant_vectors_[0]
    point_label = "relevant vecs"
    cm = ['b' if cw < 0 else 'r' for cw in rvm.corrected_weights]
    pos_rv = plt.scatter(svrv[rvm.corrected_weights>0, 0], svrv[rvm.corrected_weights>0, 1], color = 'r', s = 60)
    neg_rv = plt.scatter(svrv[rvm.corrected_weights < 0, 0], svrv[rvm.corrected_weights < 0, 1], color = 'b', s = 60)
    print("All relevance vectors")
    for i in range(len(rvm.corrected_weights)):
        print(i, rvm.orig_weights[i], rvm.corrected_weights[i], svrv[i,:])

max_t = 1.05
t = np.linspace(0,max_t, 1000)
C = np.array([-2.5, 7.0])
c1 = [C[0], 7.5, 1.0]
c2 = [C[1], 10.5, -15.0]
curve_x = c1[0] + c1[1]*t + c1[2]*t**2
curve_y = c2[0] + c2[1]*t + c2[2]*t**2

curve_colliding = plt.plot(curve_x, curve_y, 'xkcd:red', markersize=1, label="colliding", linewidth=3)

radius = rvm.get_radius(line_seg, C, n1 = 1.0, n2 = 1.0)
print("radius", radius)
next_t = 0.0
curr_t = 0.0
ax = plt.axes()
curr_x = c1[0]
curr_y = c2[0]
''''''
while radius > 0.1 and curr_t < max_t:
    circle = plt.Circle([curr_x, curr_y], radius, color='xkcd:red', fill=False, label="free ball", linewidth=3)
    plt.scatter(curr_x, curr_y, color="xkcd:blue", marker='x', s=200, linewidths=20)
    print("center", [curr_x, curr_y])
    ax.add_artist(circle)
    next_t = solve_next_t(c1[2], c1[1], c1[0], c2[2], c2[1], c2[0], curr_t, radius)
    print("next_t", next_t, "curr_t", curr_t)
    curr_x = c1[0] + c1[1] * next_t + c1[2] * next_t ** 2
    curr_y = c2[0] + c2[1] * next_t + c2[2] * next_t ** 2
    radius = rvm.get_radius(line_seg, np.array([curr_x, curr_y]))
    print("radius = ", radius)
    curr_t = next_t

max_t = 0.85
t = np.linspace(0,max_t, 1000)
C = np.array([-2.5, 2.0])
c1 = [C[0], 15.5, 1.0]
c2 = [C[1], -11.5, 11.0]
curve_x = c1[0] + c1[1]*t + c1[2]*t**2
curve_y = c2[0] + c2[1]*t + c2[2]*t**2


curve_free = plt.plot(curve_x, curve_y, 'xkcd:green', markersize=1, label="free", linewidth=3)

radius = rvm.get_radius(line_seg, C, n1 = 1.0, n2 = 1.0)
print("radius", radius)
next_t = 0.0
curr_t = 0.0
ax = plt.axes()
curr_x = c1[0]
curr_y = c2[0]
''''''
while radius > 0.1 and curr_t < max_t:
    circle = plt.Circle([curr_x, curr_y], radius, color='xkcd:green', fill=False, label="free ball", linewidth=3)
    plt.scatter(curr_x, curr_y, color="xkcd:blue", marker='x', s=200, linewidths=20)
    print("center", [curr_x, curr_y])
    ax.add_artist(circle)
    next_t = solve_next_t(c1[2], c1[1], c1[0], c2[2], c2[1], c2[0], curr_t, radius)
    print("next_t", next_t, "curr_t", curr_t)
    curr_x = c1[0] + c1[1] * next_t + c1[2] * next_t ** 2
    curr_y = c2[0] + c2[1] * next_t + c2[2] * next_t ** 2
    radius = rvm.get_radius(line_seg, np.array([curr_x, curr_y]), n1 = 1.0, n2 = 1.0)
    print("radius = ", radius)
    curr_t = next_t


#plt.plot(traj[:,0], traj[:,1],'go',markersize=4)
# plt.plot()
#plt.plot()
#title = model_name
#plt.title(title, fontsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.xlabel("x1", fontsize = 20)
#plt.ylabel("x2", fontsize = 20)
h1, _ = cs1.legend_elements()
h2, _ = cs2.legend_elements()
h3, _ = cs3.legend_elements()
plt.legend([h1[0], h2[0], h3[0], curve_colliding[0], curve_free[0]], \
           ['$G_1(x) = 0$','$G_2(x) = 0$', '$G_3(x) = 0$', 'colliding curve', "free curve"], loc = 1, fontsize=22,framealpha=0.5)
plt.savefig("/home/erl/repos/sklearn-bayes/figs/rvm_model_checking_curves.eps", bbox_inches='tight', pad_inches=0)

plt.show()


#######################################################################################################################
########################## ILLUSTRATE COLLISION CHECKING A COLLIDING CURVE #####################
plt.figure(figsize = (10,8))
#plt.figure()
levels = np.arange(0,1, 0.0005)
#ax.contour3D(X1,X2,np.reshape(model,(n_grid,n_grid)), 100, cmap='coolwarm',
#                   linewidth=0, antialiased=False)

plot_heatmap = True
prob_thres = (1+ scipy.special.erf(THRESHOLD))/2
print("THRESHOLD: ", prob_thres)
rv_grid [rv_grid < prob_thres] = prob_thres
#rv_grid [rv_grid > prob_thres] = 0.8
if plot_heatmap:
    plt.contourf(X1, X2, np.reshape(rv_grid, (n_grid, n_grid)), cmap='Greys')
    #cb  = plt.colorbar()
    #cb.ax.tick_params(labelsize=20)
    #cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])



#arr1 = plt.arrow(0, 0,  3.5, 6.5, head_width=0.3,
#                 head_length=0.3, fc="xkcd:green", ec="xkcd:green", width = 0.1, label="colliding segment")
#arr2 = plt.arrow(0, 0, 3.5, -0.8, head_width=0.3,
#                 head_length=0.3, fc="xkcd:orange", ec="xkcd:orange", width=0.1, label="free segment")
#cs2 = plt.contour(X1, X2, np.reshape(upper_grid, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles="dashed", label="$G_2(x)$")
#plt.contour(X1, X2, np.reshape(upper_grid2, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles = "dotted", label="$G_2(x)$")
#plt.contour(X1, X2, np.reshape(upper_grid3, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
#            linestyles="solid")
cs3 = plt.contour(X1, X2, np.reshape(upper_grid3, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
                linestyles="dashdot", label="$G_3(x)$")
cs1 = plt.contour(X1, X2, np.reshape(rv_grid, (n_grid, n_grid)), levels=[prob_thres], cmap="Greys_r", label="$G_1(x)$")

cs2 = plt.contour(X1, X2, np.reshape(upper_grid, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles="dotted", label="$G_2(x)$")
#radius = rvm.get_radius(line_seg, A)
#print("radius", radius)
#ax = plt.axes()
#circle = plt.Circle(A, radius, color='xkcd:orange', fill=False, label="free ball")
#ax.add_artist(circle)

plot_rvs = False
if plot_rvs:
    svrv = None
    point_label = None
    svrv = rvm.relevant_vectors_[0]
    point_label = "relevant vecs"
    cm = ['b' if cw < 0 else 'r' for cw in rvm.corrected_weights]
    pos_rv = plt.scatter(svrv[rvm.corrected_weights>0, 0], svrv[rvm.corrected_weights>0, 1], color = 'r', s = 60)
    neg_rv = plt.scatter(svrv[rvm.corrected_weights < 0, 0], svrv[rvm.corrected_weights < 0, 1], color = 'b', s = 60)
    print("All relevance vectors")
    for i in range(len(rvm.corrected_weights)):
        print(i, rvm.orig_weights[i], rvm.corrected_weights[i], svrv[i,:])

colliding_curve = True
if colliding_curve:
    max_t = 1.1
    t = np.linspace(0,max_t, 1000)
    C = np.array([-6.0, 1.0])
    #c1 = [C[0], 7.5, 1.0]
    #c2 = [C[1], -10.5, +15.0]
    c1 = [C[0], 7.5, 1.0]
    c2 = [C[1], 11.5, -15.0]
    curve_x = c1[0] + c1[1]*t + c1[2]*t**2
    curve_y = c2[0] + c2[1]*t + c2[2]*t**2

    curve_colliding = plt.plot(curve_x, curve_y, 'xkcd:red', markersize=1, label="colliding", linewidth=3)

    radius = rvm.get_radius(line_seg, C, n1 = 1.0, n2 = 1.0)
    print("radius", radius)
    next_t = 0.0
    curr_t = 0.0
    ax = plt.axes()
    curr_x = c1[0]
    curr_y = c2[0]
    ''''''
    while radius > 0.1 and curr_t < max_t:
        circle = plt.Circle([curr_x, curr_y], radius, color='xkcd:red', fill=False, label="free ball", linewidth=3)
        plt.scatter(curr_x, curr_y, color="xkcd:blue", marker='x', s=200, linewidths=20)
        print("center", [curr_x, curr_y])
        ax.add_artist(circle)
        next_t = solve_next_t(c1[2], c1[1], c1[0], c2[2], c2[1], c2[0], curr_t, radius)
        print("next_t", next_t, "curr_t", curr_t)
        curr_x = c1[0] + c1[1] * next_t + c1[2] * next_t ** 2
        curr_y = c2[0] + c2[1] * next_t + c2[2] * next_t ** 2
        radius = rvm.get_radius(line_seg, np.array([curr_x, curr_y]), n1 = 1.0, n2 = 1.0)
        print("radius = ", radius)
        curr_t = next_t
else:
    max_t = 0.85
    t = np.linspace(0,max_t, 1000)
    C = np.array([-2.5, 2.0])
    c1 = [C[0], 15.5, 1.0]
    c2 = [C[1], -11.5, 11.0]
    curve_x = c1[0] + c1[1]*t + c1[2]*t**2
    curve_y = c2[0] + c2[1]*t + c2[2]*t**2


    curve_free = plt.plot(curve_x, curve_y, 'xkcd:green', markersize=1, label="free", linewidth=3)

    radius = rvm.get_radius(line_seg, C)
    print("radius", radius)
    next_t = 0.0
    curr_t = 0.0
    ax = plt.axes()
    curr_x = c1[0]
    curr_y = c2[0]
    ''''''
    while radius > 0.1 and curr_t < max_t:
        circle = plt.Circle([curr_x, curr_y], radius, color='xkcd:green', fill=False, label="free ball", linewidth=3)
        plt.scatter(curr_x, curr_y, color="xkcd:blue", marker='x', s=200, linewidths=20)
        print("center", [curr_x, curr_y])
        ax.add_artist(circle)
        next_t = solve_next_t(c1[2], c1[1], c1[0], c2[2], c2[1], c2[0], curr_t, radius)
        print("next_t", next_t, "curr_t", curr_t)
        curr_x = c1[0] + c1[1] * next_t + c1[2] * next_t ** 2
        curr_y = c2[0] + c2[1] * next_t + c2[2] * next_t ** 2
        radius = rvm.get_radius(line_seg, np.array([curr_x, curr_y]), n1 = 1.0, n2 = 1.0)
        print("radius = ", radius)
        curr_t = next_t


#plt.plot(traj[:,0], traj[:,1],'go',markersize=4)
# plt.plot()
#plt.plot()
#title = model_name
#plt.title(title, fontsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.xlabel("x1", fontsize = 20)
#plt.ylabel("x2", fontsize = 20)
h1, _ = cs1.legend_elements()
h2, _ = cs2.legend_elements()
h3, _ = cs3.legend_elements()
if colliding_curve:
    plt.legend([h1[0], h2[0], h3[0], curve_colliding[0]], \
               ['$G_1(x) = 0$','$G_2(x) = 0$', '$G_3(x) = 0$', 'colliding curve'], loc = 1, fontsize=22,framealpha=0.5)
    plt.savefig("/home/erl/repos/sklearn-bayes/figs/rvm_model_checking_curves_colliding.eps", bbox_inches='tight', pad_inches=0)
else:
    plt.legend([h1[0], h2[0], h3[0], curve_free[0]], \
               ['$G_1(x) = 0$','$G_2(x) = 0$', '$G_3(x) = 0$', "free curve"], loc=1, fontsize=22, framealpha=0.5)
    plt.savefig("/home/erl/repos/sklearn-bayes/figs/rvm_model_checking_curves_free.eps", bbox_inches='tight',
                pad_inches=0)

plt.show()



#######################################################################################################################
########################## ILLUSTRATE COLLISION CHECKING A FREE CURVE #####################
plt.figure(figsize = (10,8))
#plt.figure()
levels = np.arange(0,1, 0.0005)
#ax.contour3D(X1,X2,np.reshape(model,(n_grid,n_grid)), 100, cmap='coolwarm',
#                   linewidth=0, antialiased=False)

plot_heatmap = True
prob_thres = (1+ scipy.special.erf(THRESHOLD))/2
print("THRESHOLD: ", prob_thres)
rv_grid [rv_grid < prob_thres] = prob_thres
#rv_grid [rv_grid > prob_thres] = 0.8
if plot_heatmap:
    plt.contourf(X1, X2, np.reshape(rv_grid, (n_grid, n_grid)), cmap='Greys')
    #cb  = plt.colorbar()
    #cb.ax.tick_params(labelsize=20)
    #cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])



#arr1 = plt.arrow(0, 0,  3.5, 6.5, head_width=0.3,
#                 head_length=0.3, fc="xkcd:green", ec="xkcd:green", width = 0.1, label="colliding segment")
#arr2 = plt.arrow(0, 0, 3.5, -0.8, head_width=0.3,
#                 head_length=0.3, fc="xkcd:orange", ec="xkcd:orange", width=0.1, label="free segment")
#cs2 = plt.contour(X1, X2, np.reshape(upper_grid, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles="dashed", label="$G_2(x)$")
#plt.contour(X1, X2, np.reshape(upper_grid2, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles = "dotted", label="$G_2(x)$")
#plt.contour(X1, X2, np.reshape(upper_grid3, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
#            linestyles="solid")
cs3 = plt.contour(X1, X2, np.reshape(upper_grid3, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r",
                linestyles="dashdot", label="$G_3(x)$")
cs1 = plt.contour(X1, X2, np.reshape(rv_grid, (n_grid, n_grid)), levels=[prob_thres], cmap="Greys_r", label="$G_1(x)$")

cs2 = plt.contour(X1, X2, np.reshape(upper_grid, (n_grid, n_grid)), levels=[0.0], cmap="Greys_r", linestyles="dotted", label="$G_2(x)$")
#radius = rvm.get_radius(line_seg, A)
#print("radius", radius)
#ax = plt.axes()
#circle = plt.Circle(A, radius, color='xkcd:orange', fill=False, label="free ball")
#ax.add_artist(circle)

plot_rvs = False
if plot_rvs:
    svrv = None
    point_label = None
    svrv = rvm.relevant_vectors_[0]
    point_label = "relevant vecs"
    cm = ['b' if cw < 0 else 'r' for cw in rvm.corrected_weights]
    pos_rv = plt.scatter(svrv[rvm.corrected_weights>0, 0], svrv[rvm.corrected_weights>0, 1], color = 'r', s = 60)
    neg_rv = plt.scatter(svrv[rvm.corrected_weights < 0, 0], svrv[rvm.corrected_weights < 0, 1], color = 'b', s = 60)
    print("All relevance vectors")
    for i in range(len(rvm.corrected_weights)):
        print(i, rvm.orig_weights[i], rvm.corrected_weights[i], svrv[i,:])

colliding_curve = False
if colliding_curve:
    max_t = 1.1
    t = np.linspace(0,max_t, 1000)
    C = np.array([-6.0, 1.0])
    #c1 = [C[0], 7.5, 1.0]
    #c2 = [C[1], -10.5, +15.0]
    c1 = [C[0], 7.5, 1.0]
    c2 = [C[1], 11.5, -15.0]
    curve_x = c1[0] + c1[1]*t + c1[2]*t**2
    curve_y = c2[0] + c2[1]*t + c2[2]*t**2

    curve_colliding = plt.plot(curve_x, curve_y, 'xkcd:red', markersize=1, label="colliding", linewidth=3)

    radius = rvm.get_radius(line_seg, C, n1 = 1.0, n2 = 1.0)
    print("radius", radius)
    next_t = 0.0
    curr_t = 0.0
    ax = plt.axes()
    curr_x = c1[0]
    curr_y = c2[0]
    ''''''
    while radius > 0.1 and curr_t < max_t:
        circle = plt.Circle([curr_x, curr_y], radius, color='xkcd:red', fill=False, label="free ball", linewidth=3)
        plt.scatter(curr_x, curr_y, color="xkcd:blue", marker='x', s=200, linewidths=20)
        print("center", [curr_x, curr_y])
        ax.add_artist(circle)
        next_t = solve_next_t(c1[2], c1[1], c1[0], c2[2], c2[1], c2[0], curr_t, radius)
        print("next_t", next_t, "curr_t", curr_t)
        curr_x = c1[0] + c1[1] * next_t + c1[2] * next_t ** 2
        curr_y = c2[0] + c2[1] * next_t + c2[2] * next_t ** 2
        radius = rvm.get_radius(line_seg, np.array([curr_x, curr_y]), n1 = 1.0, n2 = 1.0)
        print("radius = ", radius)
        curr_t = next_t
else:
    max_t = 0.85
    t = np.linspace(0,max_t, 1000)
    C = np.array([-2.5, 2.0])
    c1 = [C[0], 15.5, 1.0]
    c2 = [C[1], -11.5, 11.0]
    curve_x = c1[0] + c1[1]*t + c1[2]*t**2
    curve_y = c2[0] + c2[1]*t + c2[2]*t**2


    curve_free = plt.plot(curve_x, curve_y, 'xkcd:green', markersize=1, label="free", linewidth=3)

    radius = rvm.get_radius(line_seg, C, n1 = 1.0, n2 = 1.0)
    print("radius", radius)
    next_t = 0.0
    curr_t = 0.0
    ax = plt.axes()
    curr_x = c1[0]
    curr_y = c2[0]
    ''''''
    while radius > 0.1 and curr_t < max_t:
        circle = plt.Circle([curr_x, curr_y], radius, color='xkcd:green', fill=False, label="free ball", linewidth=3)
        plt.scatter(curr_x, curr_y, color="xkcd:blue", marker='x', s=200, linewidths=20)
        print("center", [curr_x, curr_y])
        ax.add_artist(circle)
        next_t = solve_next_t(c1[2], c1[1], c1[0], c2[2], c2[1], c2[0], curr_t, radius)
        print("next_t", next_t, "curr_t", curr_t)
        curr_x = c1[0] + c1[1] * next_t + c1[2] * next_t ** 2
        curr_y = c2[0] + c2[1] * next_t + c2[2] * next_t ** 2
        radius = rvm.get_radius(line_seg, np.array([curr_x, curr_y]), n1 = 1.0, n2 = 1.0)
        print("radius = ", radius)
        curr_t = next_t


#plt.plot(traj[:,0], traj[:,1],'go',markersize=4)
# plt.plot()
#plt.plot()
#title = model_name
#plt.title(title, fontsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.xlabel("x1", fontsize = 20)
#plt.ylabel("x2", fontsize = 20)
h1, _ = cs1.legend_elements()
h2, _ = cs2.legend_elements()
h3, _ = cs3.legend_elements()
if colliding_curve:
    plt.legend([h1[0], h2[0], h3[0], curve_colliding[0]], \
               ['$G_1(x) = 0$','$G_2(x) = 0$', '$G_3(x) = 0$', 'colliding curve'], loc = 1, fontsize=22,framealpha=0.5)
    plt.savefig("/home/erl/repos/sklearn-bayes/figs/rvm_model_checking_curves_colliding.eps", bbox_inches='tight', pad_inches=0)
else:
    plt.legend([h1[0], h2[0], h3[0], curve_free[0]], \
               ['$G_1(x) = 0$','$G_2(x) = 0$', '$G_3(x) = 0$', "free curve"], loc=1, fontsize=22, framealpha=0.5)
    plt.savefig("/home/erl/repos/sklearn-bayes/figs/rvm_model_checking_curves_free.eps", bbox_inches='tight',
                pad_inches=0)

plt.show()



