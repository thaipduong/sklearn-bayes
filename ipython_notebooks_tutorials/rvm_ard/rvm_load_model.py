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
import load_gt_rosbag
from sklearn.metrics import mean_squared_error


from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import classification_report
import pickle

f = open("/home/erl/repos/sklearn-bayes/data/results/good5/rosbag_rvm5laser_samples_seq_25pos50negres_inflated.pkl","rb")
#f = open("/home/erl/repos/sklearn-bayes/data/results/good4/rosbag_rvm4.pkl","rb")
#f = open("/home/erl/repos/sklearn-bayes/data/results/good1/rosbag_rvm_good.pkl","rb")
rvm = pickle.load(f)
print("Load rvm model successfully:" + str(len(rvm.Mn)) + "relevance vectors")
all_rv_X, all_rv_y, all_rv_A = rvm.recover_posterior_dist(tol_factor=0.01, a_thres=1.0)
print("Done approx posterior dist using Laplace approx:" + str(len(rvm.Mn)) + "relevance vectors")
#max_x      = np.array([65,26])#10*np.max(X,axis = 0)
#min_x      = np.array([-20,-5])#10*np.min(X,axis = 0)
max_x      = np.array([65.25,26.25])#10*np.max(X,axis = 0)
min_x      = np.array([-20.25,-5.25])#10*np.min(X,axis = 0)

n_grid_x = int((65.25+20.25)/0.25) + 1
n_grid_y = int((26.25+5.25)/0.25) + 1
X1         = np.linspace(min_x[0],max_x[0],n_grid_x)
X2         = np.linspace(min_x[1],max_x[1],n_grid_y)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid_x*n_grid_y,2])
Xgrid[:,0] = np.reshape(x1,(n_grid_x*n_grid_y,))
Xgrid[:,1] = np.reshape(x2,(n_grid_x*n_grid_y,))

rv_grid, var_grid, _, _ = rvm.predict_proba(Xgrid)
rv_grid = rv_grid[:,1]
threshold = 0.67
rv_grid_bin = rv_grid > threshold

ratio = int(100*n_grid_x/n_grid_y)

# Plot relevance vectors
svrv = np.array(rvm.all_rv_X)
w = rvm.Mn
plt.figure(figsize=(ratio/10, 10))
pos_rv = plt.scatter(svrv[w > 0, 0], svrv[w > 0, 1], color='r', s=40)
neg_rv = plt.scatter(svrv[w < 0, 0], svrv[w < 0, 1], color='b', s=40)
plt.savefig("/home/erl/repos/sklearn-bayes/data/results/rosbag_rvs.pdf", bbox_inches='tight', pad_inches=0)

# Illustrate collision checking
plt.figure(figsize=(ratio/10, 10))
levels = np.arange(0,1, 0.0005)
plt.contourf(X1, X2, np.reshape(rv_grid, (n_grid_y, n_grid_x)), cmap='coolwarm')
cb  = plt.colorbar()
cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
cb.ax.tick_params(labelsize=20)
plt.savefig("/home/erl/repos/sklearn-bayes/data/results/rosbag_rvmmap.pdf", bbox_inches='tight', pad_inches=0)


plt.figure(figsize=(ratio/10, 10))
levels = np.arange(0,1, 0.0005)
plt.contourf(X1, X2, np.reshape(rv_grid_bin, (n_grid_y, n_grid_x)), cmap="Greys")
cb  = plt.colorbar()
cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
cb.ax.tick_params(labelsize=20)
plt.savefig("/home/erl/repos/sklearn-bayes/data/results/rosbag_rvmmap_bin.pdf", bbox_inches='tight', pad_inches=0)

gtmap = load_gt_rosbag.get_goundtruth_map()
rvm_map_bin = np.reshape(rv_grid_bin, (n_grid_y, n_grid_x))
plt.figure(figsize=(ratio/10, 10))
levels = np.arange(0,1, 0.0005)
plt.contourf(X1, X2, np.reshape(gtmap, (n_grid_y, n_grid_x)), cmap="Greys")
cb  = plt.colorbar()
cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
cb.ax.tick_params(labelsize=20)
plt.savefig("/home/erl/repos/sklearn-bayes/data/results/rosbag_gt.pdf", bbox_inches='tight', pad_inches=0)

print("Loaded groundtruth map...")

print("#################################################################################")
drift_allowance = 2
error = load_gt_rosbag.compare(gtmap, rvm_map_bin, drift_allowance = drift_allowance)
tpr_error, count_true = load_gt_rosbag.compare_tpr(gtmap, rvm_map_bin, drift_allowance = drift_allowance)
print("ACCURACY:", 100*(n_grid_y*n_grid_x - error)/(n_grid_y*n_grid_x), "%")
print("RECALL:", 100*(count_true - tpr_error)/count_true, "%")

print("#################################################################################")
plt.show()