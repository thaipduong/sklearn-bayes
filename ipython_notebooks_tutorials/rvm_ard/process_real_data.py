import numpy as np
import matplotlib.pyplot as plt
from skbayes.rvm_ard_models import RVC4
from sklearn.metrics.pairwise import pairwise_kernels
prefix = "/home/erl/repos/sklearn-bayes/data/env_map27/"
#prefix = "/home/erl/repos/sklearn-bayes/data/env_map207/"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def get_kernel(X, Y, gamma, degree, coef0, kernel, kernel_params):
	'''
	Calculates kernelised features for RVR and RVC
	'''
	if callable(kernel):
		params = kernel_params or {}
	else:
		params = {"gamma": gamma,
				  "degree": degree,
				  "coef0": coef0}
	return pairwise_kernels(X, Y, metric=kernel,
							filter_params=True, **params)

############################################# LOAD DATA #############################################################
rel_vec_data = np.load(prefix + "relevance_vecs_w.npz", allow_pickle=True,  encoding='latin1')
rel_vec_xi_data = np.load(prefix + "relevance_vecs_xi.npz", allow_pickle=True,  encoding='latin1')
astar_data = np.load(prefix + "astar_stats.npz", allow_pickle=True,  encoding='latin1')
support_data = np.load(prefix + "support_stats.npz", allow_pickle=True,  encoding='latin1')
print("Data loaded")
############################################# EXTRACT DATA #############################################################
final_idx = -6
#robot_poses = rel_vec_data['robot_poses']
robot_poses = np.load(prefix + "robot_poses.npy")
all_rel_vecs_label_w = rel_vec_data['all_rel_vecs_label_w']
all_rel_vecs = rel_vec_xi_data['all_rel_vecs']
all_rel_vecs_xi = rel_vec_xi_data['all_rel_vecs_xi']
final_rel_vecs = all_rel_vecs[final_idx]
final_rel_vecs_label = np.array([1.0 if l > 0 else 0.0 for l in all_rel_vecs_label_w[final_idx]])
final_rel_vecs_label2 = np.array(all_rel_vecs_label_w[final_idx])
final_rel_vecs_xi = np.array(all_rel_vecs_xi[final_idx])
print("Extracted data")

############################################# RECOVER MAP #############################################################
rvm = RVC4(n_iter = 50, tol = 1e-1, n_iter_solver = 20, tol_solver = 1e-2, kernel = 'rbf', gamma = 3.0)
final_rel_K = get_kernel(final_rel_vecs, final_rel_vecs, rvm.gamma, rvm.degree, rvm.coef0,
                              rvm.kernel, rvm.kernel_params)
Mn, Sn, B, t_hat, cholesky = rvm._posterior_dist(final_rel_K, final_rel_vecs_label, final_rel_vecs_xi, keep_prev_mean=False,
                                                          tol_mul=0.001)
if cholesky:
	Sn = np.dot(Sn.T, Sn)
print("Laplace approximation done!")

############################################# PLOTTING FIGURES ##########################################################
################ MAP ####################
max_x      = np.array([25.0, 25.0])#10*np.max(X,axis = 0)
min_x      = np.array([-5.0, -25.0])#10*np.min(X,axis = 0)

n_grid_x = int((max_x[0] - min_x[0])/0.25) + 1
n_grid_y = int((max_x[1] - min_x[1])/0.25) + 1
X1         = np.linspace(min_x[0],max_x[0],n_grid_x)
X2         = np.linspace(min_x[1],max_x[1],n_grid_y)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid_x*n_grid_y,2])
Xgrid[:,0] = np.reshape(x1,(n_grid_x*n_grid_y,))
Xgrid[:,1] = np.reshape(x2,(n_grid_x*n_grid_y,))
rv_grid, var_grid, _, _  = rvm.predict_proba_rvs_userdefined( Xgrid, final_rel_vecs, Mn, Sn, -0.5)
rv_grid = rv_grid[:,1]
ratio = n_grid_y/n_grid_x

final_rel_vecs_label2 = Mn
plt.figure(figsize=(ratio*4, 4))
pos_rv = plt.scatter(-final_rel_vecs[final_rel_vecs_label2 > 0, 1], final_rel_vecs[final_rel_vecs_label2 > 0, 0], color='r', s=10, label="pos. relevance vectors")
neg_rv = plt.scatter(-final_rel_vecs[final_rel_vecs_label2 < 0, 1], final_rel_vecs[final_rel_vecs_label2 < 0, 0], color='b', s=10, label="neg. relevance vectors")
plt.legend(loc = 2, framealpha=0.5)#
ax = plt.axes()
# Setting the background color
ax.set_facecolor("lightgrey")
plt.savefig(prefix + "realexp_rvs.pdf", bbox_inches='tight', pad_inches=0)


plt.figure(figsize=(ratio*4*1.185, 4))
levels = np.arange(0,1, 0.0005)
min_val = np.min(rv_grid)
max_val = np.max(rv_grid)
#rv_grid = (rv_grid - min_val)/(max_val - min_val)

im = plt.contourf(-X2, X1, np.reshape(rv_grid, (n_grid_y, n_grid_x)).transpose(), 11, cmap='coolwarm',vmin=0.0, vmax=1.0)
plt.plot(-robot_poses[:, 1], robot_poses[:, 0], color='xkcd:cyan', linewidth=2, label="robot trajectory")
plt.scatter(-robot_poses[0, 1], robot_poses[0, 0], color='r', s=30, label="start")
plt.scatter(-robot_poses[-1, 1], robot_poses[-1, 0], color='b', s=30, label = "end")
cb  = plt.colorbar(im, orientation="vertical", pad=0.01)
#cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
#cb.ax.tick_params(labelsize=10)
ax = plt.axes()
# Setting the background color
ax.set_facecolor("lightgrey")
plt.legend(loc = 2, framealpha=1.5)#
plt.savefig(prefix + "realexp_rvmmap.pdf", bbox_inches='tight', pad_inches=0)
#plt.show()
################ ASTAR STATS ####################

astar_info = astar_data['astar_info']
astar_time = astar_data['astar_time']
astar_time = astar_time[astar_info[:,3] > 0]
astar_info = astar_info[astar_info[:,3] > 0, :]
time_per_node = 1000000*astar_info[:,0]/astar_info[:,3]
print("Average checking time/motion primitive: ", np.average(time_per_node))

astar_info = astar_data['astar_info']
astar_time = astar_data['astar_time']
time_per_node = 1000000*astar_info[:,0]/astar_info[:,3]
#avg_time[avg_time>4.0] = 4.0
#print("Avg astart time per motion primitive: ", np.mean(time_per_node))
f, ax = plt.subplots(figsize=(ratio*4, 4))
ax.plot(astar_time, time_per_node, 'g',label='A* time per expanded node')
ax.set_xlabel("time(s)", size='x-large')
ax.set_ylabel("time($\mu$s)", size='x-large')
ax.legend(facecolor='xkcd:silver',fontsize='large')
plt.savefig(prefix + "realcar_astartime_permotion.pdf",bbox_inches='tight', pad_inches = 0)


################ MAP UPDATE STATS################
support_info = support_data['support_info']
stime = support_data['support_time']
#avg_time = 1000*astar_info[:,0]/astar_info[:,2]
#avg_time[avg_time>4.0] = 4.0
f, ax = plt.subplots(figsize=(ratio*4, 4))
ax.plot(stime, support_info[:,0], 'g',label='Map update time')
ax.set_xlabel("time(s)", size='x-large')
ax.set_ylabel("time(s)", size='x-large')
ax.legend(facecolor='xkcd:silver',fontsize='large')
plt.savefig(prefix + "realcar_map_update.pdf",bbox_inches='tight', pad_inches = 0)
#################################################
plt.show()