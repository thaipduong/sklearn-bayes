import numpy as np
import matplotlib.pyplot as plt
from skbayes.rvm_ard_models import RVC4
from sklearn.metrics.pairwise import pairwise_kernels
prefix = "/home/erl/repos/sklearn-bayes/data/env_map27/video/"
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
rel_vec_xi_data = np.load(prefix + "relevance_vecs_xi_rerun.npz", allow_pickle=True,  encoding='latin1')
poses_path = np.load(prefix + "robot_poses_path_rerun.npz", allow_pickle=True,  encoding='latin1')
print("Data loaded")
############################################# EXTRACT DATA #############################################################
#final_idx = -1
robot_poses = poses_path['robot_poses']
odom_time = poses_path['odom_time']
all_path = poses_path['path']
all_rel_vecs = rel_vec_xi_data['all_rel_vecs']
all_rel_vecs_xi = rel_vec_xi_data['all_rel_vecs_xi']
all_rel_vecs_label = rel_vec_xi_data['all_rel_vecs_y']
vec_pub_time = rel_vec_xi_data['pub_time']




############################################# RECOVER MAP #############################################################
rvm = RVC4(n_iter = 50, tol = 1e-1, n_iter_solver = 20, tol_solver = 1e-2, kernel = 'rbf', gamma = 3.0)
#################### MAP CONFIG
max_x      = np.array([25.0, 35.0])#10*np.max(X,axis = 0)
min_x      = np.array([-14.0, -35.0])#10*np.min(X,axis = 0)

n_grid_x = int((max_x[0] - min_x[0])/0.25) + 1
n_grid_y = int((max_x[1] - min_x[1])/0.25) + 1
X1         = np.linspace(min_x[0],max_x[0],n_grid_x)
X2         = np.linspace(min_x[1],max_x[1],n_grid_y)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid_x*n_grid_y,2])
Xgrid[:,0] = np.reshape(x1,(n_grid_x*n_grid_y,))
Xgrid[:,1] = np.reshape(x2,(n_grid_x*n_grid_y,))
ratio = n_grid_y / n_grid_x
goal1_idx = 1600
##############################################################################################
vec_idx = 0
rv_grid = None
for idx in range(0,len(robot_poses),1):#len(robot_poses)):
	pose_time = odom_time[idx]
	vec_time = vec_pub_time[vec_idx]
	print(idx, vec_time, pose_time, len(vec_pub_time))
	if (pose_time > vec_time and vec_idx < len(vec_pub_time)) or (vec_idx == 0 and rv_grid is None):
		final_rel_vecs = all_rel_vecs[vec_idx]
		final_rel_vecs_label = np.array([1.0 if l > 0 else 0.0 for l in all_rel_vecs_label[vec_idx]])
		final_rel_vecs_label2 = np.array(all_rel_vecs_label[vec_idx])
		final_rel_vecs_xi = np.array(all_rel_vecs_xi[vec_idx])
		print("Extracted data")
		print("Number of relevance vectors: ", len(final_rel_vecs))
		final_rel_K = get_kernel(final_rel_vecs, final_rel_vecs, rvm.gamma, rvm.degree, rvm.coef0, rvm.kernel, rvm.kernel_params)
		Mn, Sn, B, t_hat, cholesky = rvm._posterior_dist(final_rel_K, final_rel_vecs_label, final_rel_vecs_xi, keep_prev_mean=False, tol_mul=0.01)
		if cholesky:
			Sn = np.dot(Sn.T, Sn)
		print("Laplace approximation done!")
		rv_grid, var_grid, _, _  = rvm.predict_proba_rvs_userdefined( Xgrid, final_rel_vecs, Mn, Sn, -0.05)
		rv_grid = rv_grid[:,1]
		vec_idx = vec_idx + 1
		if vec_idx >= len(vec_pub_time):
			vec_idx = len(vec_pub_time) - 1
		vec_time = vec_pub_time[vec_idx]
		min_val = np.min(rv_grid)
		max_val = np.max(rv_grid)
		#rv_grid[rv_grid > 0.8] = 1.0
		#rv_grid[rv_grid < 0.2] = 0.0
		# rv_grid = (rv_grid - min_val)/(max_val - min_val)
		rv_grid = (rv_grid - np.min(rv_grid)) / (np.max(rv_grid) - np.min(rv_grid))
	elif rv_grid is None:
		rv_grid = 0.5*np.ones(n_grid_x*n_grid_y)
	'''
	######################################################## HORIZONTAL COLOR BAR
	plt.figure(figsize=(ratio*4, 4*1.1))
	pos_rv = plt.scatter(-final_rel_vecs[final_rel_vecs_label2 > 0, 1], final_rel_vecs[final_rel_vecs_label2 > 0, 0], color='r', s=2, label="pos. relevance vectors")
	neg_rv = plt.scatter(-final_rel_vecs[final_rel_vecs_label2 < 0, 1], final_rel_vecs[final_rel_vecs_label2 < 0, 0], color='b', s=2, label="neg. relevance vectors")
	plt.legend(loc = 2, framealpha=0.5)
	axes = plt.gca()
	axes.set_ylim([-17.0,max_x[0]])
	axes.set_xlim([min_x[1],max_x[1]])
	ax = plt.axes()
	# Setting the background color
	ax.set_facecolor("lightgrey")
	plt.savefig(prefix + "realexp_rvs_rerun.pdf", bbox_inches='tight', pad_inches=0)
	

	plt.figure(figsize=(ratio*4, 4*1.2))
	levels = np.arange(0,1, 0.0005)
	min_val = np.min(rv_grid)
	max_val = np.max(rv_grid)
	#rv_grid = (rv_grid - min_val)/(max_val - min_val)
	
	im = plt.contourf(-X2, X1, np.reshape(rv_grid, (n_grid_y, n_grid_x)).transpose(), 11, cmap='coolwarm',vmin=0.0, vmax=1.0)
	plt.plot(-robot_poses[:, 1], robot_poses[:, 0], color='xkcd:cyan', linewidth=2, label="robot trajectory")
	plt.scatter(-robot_poses[0, 1], robot_poses[0, 0], color='r', s=30, label="start")
	plt.scatter(-robot_poses[goal1_idx, 1], robot_poses[goal1_idx, 0], color='xkcd:purple', s=30, label = "goal 1")
	plt.scatter(-robot_poses[-1, 1], robot_poses[-1, 0], color='b', s=30, label = "goal 2")
	cb  = plt.colorbar(im, orientation="horizontal", pad=0.02)
	#cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
	#cb.ax.tick_params(labelsize=10)
	ax = plt.axes()
	# Setting the background color
	ax.set_facecolor("lightgrey")
	plt.legend(loc =3, framealpha=1.5, facecolor='xkcd:silver')
	plt.savefig(prefix + "realexp_rvmmap_rerun.pdf", bbox_inches='tight', pad_inches=0)
	#plt.show()
	'''
	######################################################## VERTICAL COLOR BAR
	plt.figure(figsize=(18,9))
	levels = np.arange(0,1, 0.0005)

	im = plt.contourf(-X2, X1, np.reshape(rv_grid, (n_grid_y, n_grid_x)).transpose(), 10, cmap='coolwarm',vmin=0.0, vmax=1.0)
	cb  = plt.colorbar(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], orientation="vertical", pad=0.02)
	plt.plot(-robot_poses[:idx+1, 1], robot_poses[:idx+1, 0], color='xkcd:cyan', linewidth=2, label="robot trajectory")
	cur_path = np.array(all_path[idx])
	if len(cur_path) > 0:
		plt.plot(-cur_path[:, 1], cur_path[:, 0], color='purple', linewidth=2,
				 label="planned path")
	plt.scatter(-robot_poses[idx, 1], robot_poses[idx, 0], color='xkcd:cyan', s=40, label="robot")
	plt.scatter(-robot_poses[0, 1], robot_poses[0, 0], color='r', marker='^', s=40, label="start")
	#plt.scatter(-robot_poses[goal1_idx, 1], robot_poses[goal1_idx, 0], marker='*', color='xkcd:purple', s=30, label = "goal 1")
	#plt.scatter(-robot_poses[-1, 1], robot_poses[-1, 0], color='b', s=30, marker='*', label = "goal 2")
	pos_rv = plt.scatter(-final_rel_vecs[final_rel_vecs_label2 > 0, 1], final_rel_vecs[final_rel_vecs_label2 > 0, 0],
						 color='r', s=20, label="pos. relevance vectors")
	neg_rv = plt.scatter(-final_rel_vecs[final_rel_vecs_label2 < 0, 1], final_rel_vecs[final_rel_vecs_label2 < 0, 0],
						 color='b', s=20, label="neg. relevance vectors")
	#cb  = plt.colorbar(im, orientation="vertical", pad=0.02)

	#cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
	#cb.ax.tick_params(labelsize=10)
	ax = plt.axes()
	# Setting the background color
	ax.set_facecolor("lightgrey")
	plt.legend(loc =3, framealpha=1.5, facecolor='xkcd:silver', prop={'size': 20})
	plt.savefig(prefix + "imgs/realexp_rvmmap_rerun_vertical"+str(idx).zfill(4)+".png", bbox_inches='tight', pad_inches=0)
	plt.close()
	#plt.show()

	#################################################
	#plt.show()