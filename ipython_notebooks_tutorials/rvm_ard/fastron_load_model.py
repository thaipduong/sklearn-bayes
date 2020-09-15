import numpy as np
import matplotlib.pyplot as plt
import load_gt_rosbag

def G2(x, y, gamma = 40, offset = 0.0):
	k = np.exp(-gamma*((np.linalg.norm(x-y, axis= 1))**2 - offset))
	#print("###########################distance:", np.linalg.norm(x-y)**2)
	return k

def calculate_score(x_test, support_points, support_weights):
	score = np.zeros([x_test.shape[0], 1])
	for i in range(len(x_test)):
		x = x_test[i,:]

		dist = np.linalg.norm((x - support_points), axis=1)
		#print("dist:", dist.shape)
		min_dist = np.min(dist)
		#g = G2(x, support_points, offset=min_dist**2)
		#print("g:", g.shape)
		score_i = support_weights*G2(x, support_points, offset=min_dist**2)
		#print("score_i:", score_i.shape)
		score[i] = np.sum(score_i)
		#print("score:", score.shape)
	label = np.sign(score)
	#print("label:", label.shape)
	return score, label

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
if BUILDMAP:
	support_vecs = np.load("/home/erl/repos/sklearn-bayes/data/results/fastronmap/support_vecs.npy")
	print("Loaded support vectors!")
	support_points = res*support_vecs[:,0:2]
	support_weights = support_vecs[:,2]
	score, bin_map = calculate_score(Xgrid, support_points, support_weights)
	bin_map[bin_map < 0] = 0.0
	bin_map = np.reshape(bin_map, (n_grid_y, n_grid_x))
	np.save("/home/erl/repos/sklearn-bayes/data/results/fastronmap/fastron_map.npy", bin_map)
	print("Done calculating scores and labels")
else:
	bin_map = np.load("/home/erl/repos/sklearn-bayes/data/results/fastronmap/fastron_map.npy")
	print("Done loading map!")

##################################################################################################
ratio = int(100*n_grid_x/n_grid_y)

plt.figure(figsize=(ratio/10, 10))
levels = np.arange(0,1, 0.0005)
plt.contourf(X1, X2, bin_map, cmap="Greys")
#plt.contourf(X1, X2, gtmap, cmap="Greys")


plt.figure(figsize=(ratio/10, 10))
levels = np.arange(0,1, 0.0005)
plt.contourf(X1, X2, gtinterior-gtmap, cmap="Greys")
#plt.contourf(X1, X2, gtmap, cmap="Greys")

#cb  = plt.colorbar()
#cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
#cb.ax.tick_params(labelsize=20)
#plt.savefig("/home/erl/repos/sklearn-bayes/data/results/rosbag_fastronmap_bin.pdf", bbox_inches='tight', pad_inches=0)

f, ax = plt.subplots(figsize=(ratio/10, 10))
#gmap = np.transpose(gmap)
ax.imshow(gtmap, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='none')
#ax.set_xlabel("length(m)", size='x-large')
#ax.set_ylabel("width(m)", size='x-large')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)

f, ax = plt.subplots(figsize=(ratio/10, 10))
#gmap = np.transpose(gmap)
ax.imshow(gtinterior, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='none')
#ax.set_xlabel("length(m)", size='x-large')
#ax.set_ylabel("width(m)", size='x-large')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)

print("#################################################################################")
drift_allowance = 1
error, total = load_gt_rosbag.compare(gtmap, bin_map, drift_allowance = drift_allowance, excluded = None)
tpr_error, count_true = load_gt_rosbag.compare_tpr(gtmap, bin_map, drift_allowance = drift_allowance)
print(error, total, n_grid_y*n_grid_x, count_true, tpr_error)
print("ACCURACY:", 100*(total - error)/(total), "%")
#print("ACCURACY:", 100*(n_grid_y*n_grid_x - error)/(n_grid_y*n_grid_x), "%")
print("RECALL:", 100*(count_true - tpr_error)/count_true, "%")

plt.show()