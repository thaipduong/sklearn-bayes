import pickle
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from scipy.stats import norm
# pl.rcParams['pdf.fonttype'] = 42
# pl.rcParams['ps.fonttype'] = 42
pl.rcParams['text.usetex'] = True
sbhm_fit_time =  np.load("/home/erl/repos/Bayesian_Hilbert_Maps/Outputs/test6/updatetime_sbhm.npz")
sbhm_fit_time = sbhm_fit_time["update_time"]
#pl.figure(figsize=(6.5,5))
#pl.plot(bhm_mdl.fit_time)
#pl.plot(bhm_mdl.fit_loop_time)
#pl.show()

def predict_proba_approx(rvm, X):
	'''
	Predicts probabilities of targets.

	Theoretical Note
	================
	Current version of method does not use MacKay's approximation
	to convolution of Gaussian and sigmoid. This results in less accurate
	estimation of class probabilities and therefore possible increase
	in misclassification error for multiclass problems (prediction accuracy
	for binary classification problems is not changed)

	Parameters
	----------
	X: array-like of size (n_samples_test,n_features)
	   Matrix of explanatory variables

	Returns
	-------
	probs: numpy array of size (n_samples_test,)
	   Estimated probabilities of target classes
	'''
	decision, var, K = rvm.decision_function(X)
	# print("Decision value!!!!!!!!!!!!!!!", decision)
	# print("Var value!!!!!!!!!!!!!!!", var)

	prob = norm.cdf(decision)
	prob[decision < 0] = norm.cdf(
		decision[decision < 0] / np.sqrt((1/rvm.lambda_max) * np.sum(K[decision < 0, :] ** 2, axis=1) + 1))
	if prob.ndim == 1:
		prob = np.vstack([1 - prob, prob]).T
	prob = prob / np.reshape(np.sum(prob, axis=1), (prob.shape[0], 1))
	return prob, var, decision, np.sqrt(var + 1)


run = 22
f = open("../../Outputs/test" + str(run) + "/rvm_model.pkl","rb")
rvm = pickle.load(f)
print("Number of relevance vectors:", len(rvm.Mn))
print("Average training time + kernel cal per scan: ", np.average(rvm.kernelcal_fit_time))
print("Average training time per scan: ", np.average(rvm.fit_time))
print("Average full posterior time per scan: ", np.average(rvm.full_posterior_time))
print("Average approx lambda max time per scan: ", np.average(rvm.lambda_max_time))
f, ax = pl.subplots(figsize=(6.5,5))
width = 2
#pl.plot(rvm.fit_time)
ax.plot(np.array(rvm.full_posterior_time) + np.array(rvm.fit_time),linewidth=width,  label = "SBKM - full cov.")
ax.plot(np.array(rvm.lambda_max_time) + np.array(rvm.fit_time),linewidth=width,  label = "SBKM - $\lambda_{max}$.")
ax.plot(sbhm_fit_time,linewidth=width, label = "SBHM - full cov.")
ax.set_xlabel("Time steps", size='x-large')
ax.set_ylabel("Map update time(s)", size='x-large')
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.legend(facecolor='xkcd:silver',fontsize='x-large')
pl.savefig("/home/erl/repos/sklearn-bayes/figs/update_time_intel.eps",bbox_inches='tight', pad_inches = 0)
pl.show()


#rvm.recover_posterior_dist(tol_factor = 0.001)
#rvm.recover_posterior_dist(tol_factor = 0.01)
#rvm.recover_posterior_dist(tol_factor = 0.01)
#rvm.recover_posterior_dist(tol_factor = 0.01)

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


fn_train, cell_resolution, cell_max_min, skip, thresh, gamma = load_parameters('intel')

############################################### PLOT MAP ###############################################################
''''''
q_resolution = 0.25
X1 = np.arange(cell_max_min[0], cell_max_min[1]+1, q_resolution)
X2 = np.arange(cell_max_min[2], cell_max_min[3]+1, q_resolution)
xx, yy = np.meshgrid(X1, X2)
X_query = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
Y_query, _, _, _ = rvm.predict_proba(X_query)
Y_query = Y_query[:,1]
#plot
pl.figure(figsize=(6.5,5))
#pl.title('Final RVM')
#pl.scatter(X_query[:, 0], X_query[:, 1], c=Y_query, cmap='jet', s=10, marker='8',edgecolors='')
print("Y_query: ", np.min(Y_query), np.max(Y_query))
im = pl.contourf(X1, X2, np.reshape(Y_query, (len(X2),len(X1))), 5,  cmap='coolwarm',vmin=0.0, vmax=1.0)
#pl.imshow(Y_query.reshape(xx.shape))
cb  = pl.colorbar(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], orientation="vertical")
pl.xlim([cell_max_min[0], cell_max_min[1]]); pl.ylim([cell_max_min[2], cell_max_min[3]])
pl.savefig("/home/erl/repos/sklearn-bayes/figs/rvmmap_intel_test" + str(run) + ".eps", bbox_inches='tight', pad_inches=0)
#pl.show()

################################################# CALCULATE ACCURACY ###################################################
# read data
g = pd.read_csv(fn_train, delimiter=',').values
print('shapes:', g.shape)
X_train = np.float_(g[:, 0:3])
Y_train = np.int_(g[:, 3][:, np.newaxis]).ravel()  # * 2 - 1
# X_test = np.float_(g[np.mod(g[:, 0], 10) == 0, 0:3])
# Y_test = np.float_(g[np.mod(g[:, 0], 10) == 0, 3][:, np.newaxis]).ravel()  # * 2 - 1
X_test = np.float_(g[::10, 0:3])
Y_test = np.float_(g[::10, 3][:, np.newaxis]).ravel()  # * 2 - 1
###################################### PLOT DATA
plot_gt = False
if plot_gt:
	pl.figure(figsize=(6.5, 5))
	pl.scatter(X_train[:, 1], X_train[:, 2], c=Y_train, s=2, cmap='coolwarm')
	#pl.title('Training data')
	pl.xlim([cell_max_min[0], cell_max_min[1]]);
	pl.ylim([cell_max_min[2], cell_max_min[3]])
	cb  = pl.colorbar(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], orientation="vertical")
	pl.savefig("/home/erl/repos/sklearn-bayes/figs/intel_data.png", bbox_inches='tight', pad_inches=0, dpi=300)
	pl.show()
####################################### QUERY PROBABILISTIC MAP
step = 10
start = 0
Y_query, _, _, _ = rvm.predict_proba_approx(X_test[:,1:])#predict_proba_approx(rvm, X_train[start:-1:step,1:])#
Y_query = Y_query[:,1]
true_Y = Y_test

######################################## PLOT AUC
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN
res = 0.01
thres = np.arange(0.0,1.0+res, res)
tpr = []
fpr = []
for i in range(len(thres)):
	th = thres[i]
	label = [1 if y_query >= th else 0 for y_query in Y_query]
	TP, FP, TN, FN = perf_measure(true_Y, label)
	tpr.append(TP/(TP+FN))
	fpr.append(FP/(FP+TN))
	check_acc = [1 if label[j] != true_Y[j] else 0 for j in range(len(Y_query))]
	#print("i = ", i, "accuracy = ", 100-100*sum(check_acc)/len(Y_query))
pl.figure()
pl.plot(fpr, tpr)
#################### CALCULATE AUC ######################
area = 0.0
print(tpr)
print(fpr)
for i in range(1,len(thres)):
	area = area + 0.5*(tpr[i-1] + tpr[i])*(-fpr[i] + fpr[i-1])
	#print(-(fpr[i] - fpr[i-1]))
print("AUC = ", area)
##################### CALCULATE NLL #####################
nll = 0.0
for i in range(len(Y_query)):
	if true_Y[i] > 0.5:
		nll = nll - np.log(Y_query[i]+1e-100)
	else:
		nll = nll - np.log(1 - Y_query[i] + 1e-100)
nll = nll/len(Y_query)
print("NLL = ", nll)
######################################################################################################################
pl.show()



