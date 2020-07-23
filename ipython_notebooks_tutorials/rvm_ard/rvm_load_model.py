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

from sklearn.metrics import mean_squared_error


from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import classification_report
import pickle
f = open("/home/erl/repos/sklearn-bayes/data/results/good1/rosbag_rvm_good.pkl","rb")
rvm = pickle.load(f)

max_x      = np.array([65,26])#10*np.max(X,axis = 0)
min_x      = np.array([-20,-5])#10*np.min(X,axis = 0)

n_grid_x = int((65+20)/0.25) + 1
n_grid_y = int((26+5)/0.25) + 1
X1         = np.linspace(min_x[0],max_x[0],n_grid_x)
X2         = np.linspace(min_x[1],max_x[1],n_grid_y)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid_x*n_grid_y,2])
Xgrid[:,0] = np.reshape(x1,(n_grid_x*n_grid_y,))
Xgrid[:,1] = np.reshape(x2,(n_grid_x*n_grid_y,))

rv_grid, var_grid, _, _ = rvm.predict_proba(Xgrid)
rv_grid = rv_grid[:,1]

print("Load rvm model successfully")
ratio = int(100*n_grid_x/n_grid_y)
# Illustrate collision checking
plt.figure(figsize=(ratio/10, 10))
levels = np.arange(0,1, 0.0005)
plt.contourf(X1, X2, np.reshape(rv_grid, (n_grid_y, n_grid_x)), cmap='coolwarm')


cb  = plt.colorbar()

#cb.set_ticks(0.075 + np.array([0.0, 0.15, 0.3, 0.45, 0.60, 0.75, 0.90]))
cb.ax.set_yticklabels(['0.0', '0.14', '0.28', '0.42', '0.56', '0.70', '0.85', '1.0'])
cb.ax.tick_params(labelsize=20)
plt.savefig("/home/erl/repos/sklearn-bayes/figs/rosbag_rvmmap.pdf", bbox_inches='tight', pad_inches=0)

plt.show()
