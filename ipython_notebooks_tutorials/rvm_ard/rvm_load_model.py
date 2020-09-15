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

plt.rcParams['text.usetex'] = True
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import classification_report
import pickle
filename = "laser_samples_seq_25pos50negres_inflated"
laser_data = np.load("/home/erl/repos/sklearn-bayes/data/"+ filename + ".npz", allow_pickle=True,  encoding='latin1')
robot_xy = laser_data['robot_xy']

f = open("/home/erl/repos/sklearn-bayes/data/results/good19_uninflated_3.0_dup/rosbag_rvm19laser_samples_seq_25pos50negres.pkl","rb")
#f = open("/home/erl/repos/sklearn-bayes/data/results/good20_uninflated_2.0_dup/rosbag_rvm20laser_samples_seq_25pos50negres.pkl","rb")

rvm = pickle.load(f)
print("Load rvm model successfully:" + str(len(rvm.Mn)) + "relevance vectors")
svrv = np.array(rvm.all_rv_X)
w = rvm.Mn
S = rvm.Sn


#threshold = -0.01
lambda_max_sqrt = np.sqrt(np.linalg.norm(S, ord=2))
#w = w - threshold*lambda_max_sqrt
#weights = w.tolist()
np.savez("/home/erl/repos/sklearn-bayes/data/results/rvm_model.npz", relvecs = svrv, Mn = w, Sn = S, lambda_max_sqrt = lambda_max_sqrt, unknown_prob = rvm.fixed_intercept)
#all_rv_X, all_rv_y, all_rv_A = rvm.recover_posterior_dist(tol_factor=0.01, a_thres=1.0)
#print("Done approx posterior dist using Laplace approx:" + str(len(rvm.Mn)) + "relevance vectors")
#max_x      = np.array([65,26])#10*np.max(X,axis = 0)
#min_x      = np.array([-20,-5])#10*np.min(X,axis = 0)
max_x      = np.array([65.25,26.25])#10*np.max(X,axis = 0)
min_x      = np.array([-18.5,-5.25])#10*np.min(X,axis = 0)

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
threshold = 0.5
rv_grid_bin = rv_grid > threshold

ratio = int(100*n_grid_x/n_grid_y)

# Plot relevance vectors
svrv = np.array(rvm.all_rv_X)
w = rvm.Mn
plt.figure(figsize=(ratio/30, 4))
pos_rv = plt.scatter(svrv[w > 0, 0], svrv[w > 0, 1], color='r', s=2)
neg_rv = plt.scatter(svrv[w < 0, 0], svrv[w < 0, 1], color='b', s=2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("/home/erl/repos/sklearn-bayes/data/results/rosbag_rvs.eps", pad_inches=0.05)

# Illustrate collision checking
plt.figure(figsize=(ratio/30, 5))
levels = np.arange(0,1, 0.0005)
min_val = np.min(rv_grid)
max_val = np.max(rv_grid)
#rv_grid = (rv_grid - min_val)/(max_val - min_val)
im = plt.contourf(X1, X2, np.reshape(rv_grid, (n_grid_y, n_grid_x)), 11, cmap='coolwarm',vmin=0.0, vmax=1.0)
cb  = plt.colorbar(im, orientation="horizontal", pad=0.01)
#cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
cb.ax.tick_params(labelsize=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("/home/erl/repos/sklearn-bayes/data/results/rosbag_rvmmap.eps", pad_inches=0.05)

plt.figure(figsize=(ratio*1.2/30, 4))
levels = np.arange(0,1, 0.0005)
min_val = np.min(rv_grid)
max_val = np.max(rv_grid)
#rv_grid = (rv_grid - min_val)/(max_val - min_val)
im = plt.contourf(X1, X2, np.reshape(rv_grid, (n_grid_y, n_grid_x)), 11, cmap='coolwarm',vmin=0.0, vmax=1.0)
cb  = plt.colorbar(im, orientation="vertical", pad=0.01)
#cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
cb.ax.tick_params(labelsize=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("/home/erl/repos/sklearn-bayes/data/results/rosbag_rvmmap_vertical.eps", pad_inches=0.05)

print("ratio aspect:", ratio/30, 4)
plt.figure(figsize=(ratio/30, 4))
levels = np.arange(0,1, 0.0005)
im = plt.contourf(X1, X2, np.reshape(rv_grid_bin, (n_grid_y, n_grid_x)), cmap="Greys")
cb  = plt.colorbar(im, orientation="horizontal", pad=0.2)
#cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
cb.ax.tick_params(labelsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("/home/erl/repos/sklearn-bayes/data/results/rosbag_rvmmap_bin.eps", pad_inches=0.05)

gtmap = load_gt_rosbag.get_goundtruth_map()
gtinterior = load_gt_rosbag.get_interior(gtmap)
gt_interior = gtmap + 0.4*gtinterior
gt_interior[gt_interior > 0.9] = 1.0
rvm_map_bin = np.reshape(rv_grid_bin, (n_grid_y, n_grid_x))
plt.figure(figsize=(ratio/30, 4))
levels = np.arange(0,1, 0.0005)
plt.contourf(X1, X2, np.reshape(gt_interior, (n_grid_y, n_grid_x)), cmap="Greys")
for i in range(1, len(robot_xy)-1, 2):
	plt.scatter(robot_xy[i, 0]-0.5, robot_xy[i, 1], color='b', s=2)
plt.scatter(robot_xy[0, 0]-1, robot_xy[0, 1], color='r', s=60)
plt.scatter(robot_xy[-1, 0]-1, robot_xy[-1, 1], color='g', s=60)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("/home/erl/repos/sklearn-bayes/data/results/rosbag_gt.eps", pad_inches=0.05)

print("Loaded groundtruth map...")

print("#################################################################################")
drift_allowance = 2
gt_interior = gtmap + gtinterior
gt_interior[gt_interior > 0] = 1.0
#error, total = load_gt_rosbag.compare(gtmap, rvm_map_bin, drift_allowance = drift_allowance, excluded = gtinterior)
error, total = load_gt_rosbag.compare(gt_interior, rvm_map_bin, drift_allowance = drift_allowance, excluded = None)
tpr_error, count_true = load_gt_rosbag.compare_tpr(gt_interior, rvm_map_bin, drift_allowance = drift_allowance)
print(error, total, n_grid_y*n_grid_x, count_true, tpr_error)
print("ACCURACY:", 100*(total - error)/total, "%")
print("RECALL:", 100*(count_true - tpr_error)/count_true, "%")

print("#################################################################################")
plt.show()