import numpy as np
import matplotlib.pyplot as plt
import load_gt_rosbag


##################################################################################################
max_x      = np.array([65.25 +0.75,26.25])#10*np.max(X,axis = 0)
min_x      = np.array([-20.25 +0.75,-5.25])#10*np.min(X,axis = 0)
n_grid_x = int((65.25+20.25)/0.25) + 1
n_grid_y = int((26.25+5.25)/0.25) + 1
X1         = np.linspace(min_x[0],max_x[0],n_grid_x)
X2         = np.linspace(min_x[1],max_x[1],n_grid_y)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid_x*n_grid_y,2])
Xgrid[:,0] = np.reshape(x1,(n_grid_x*n_grid_y,))
Xgrid[:,1] = np.reshape(x2,(n_grid_x*n_grid_y,))
gtmap = load_gt_rosbag.get_goundtruth_map()
gtinterior = load_gt_rosbag.get_interior(gtmap)
###################################################################################################

BUILDMAP = True
res = 0.25
bin_map = np.load("/home/erl/repos/sklearn-bayes/data/results/octomap/octomap.npy")
print("Done loading map!")
bin_map = bin_map.transpose()
bin_map[bin_map > 0] = 1.0
bin_map[bin_map <= 0] = 0.0
bin_map_extended = np.zeros(gtmap.shape)
bin_map_extended[3:3+bin_map.shape[0], 9:9+bin_map.shape[1]] = bin_map
##################################################################################################
ratio = int(100*n_grid_x/n_grid_y)
plt.figure(figsize=(ratio/10, 10))
levels = np.arange(0,1, 0.0005)
plt.contourf(bin_map_extended-gtmap, cmap="Greys")
#plt.contourf(X1, X2, gtmap, cmap="Greys")
#cb  = plt.colorbar()
#cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
#cb.ax.tick_params(labelsize=20)
#plt.savefig("/home/erl/repos/sklearn-bayes/data/results/rosbag_fastronmap_bin.pdf", bbox_inches='tight', pad_inches=0)


print("#################################################################################")
drift_allowance = 2
error, total = load_gt_rosbag.compare(gtmap, bin_map_extended, drift_allowance = drift_allowance, excluded = gtinterior)
tpr_error, count_true = load_gt_rosbag.compare_tpr(gtmap, bin_map_extended, drift_allowance = drift_allowance)
print(error, total, n_grid_y*n_grid_x, count_true, tpr_error)
print("ACCURACY:", 100*(total - error)/(total), "%")
print("RECALL:", 100*(count_true - tpr_error)/count_true, "%")

plt.show()