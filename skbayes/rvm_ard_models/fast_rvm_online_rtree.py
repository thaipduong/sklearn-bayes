import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.externals import six
from sklearn.linear_model.base import LinearModel, LinearClassifierMixin
from sklearn.utils import check_X_y,check_array,as_float_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.extmath import pinvh,log_logistic,safe_sparse_dot 
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from scipy.special import expit
from scipy.special import erf
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from scipy.stats import logistic
from scipy.stats import norm
from numpy.linalg import LinAlgError
from scipy.stats import multivariate_normal
from numpy import linalg as LA
import scipy.sparse
import warnings
import time
from rtree import index
from numbers import Number

INTERCEPT = False
THRESHOLD = -0.01
UNKNOWN_PROB = -0.05

#TODO: predict_proba for RVC with Laplace Approximation


def update_precisions(Q,S,q,s,A,active,tol,n_samples,clf_bias, skip = 0):
    '''
    Selects one feature to be added/recomputed/deleted to model based on 
    effect it will have on value of log marginal likelihood.
    '''
    # initialise vector holding changes in log marginal likelihood
    deltaL = np.zeros(Q.shape[0])
    
    # identify features that can be added , recomputed and deleted in model
    theta        =  q**2 - s 
    add          =  (theta > 0) * (active == False)
    recompute    =  (theta > 0) * (active == True)
    delete       = ~(add + recompute)
    
    # compute sparsity & quality parameters corresponding to features in 
    # three groups identified above
    Qadd,Sadd      = Q[add], S[add]
    Qrec,Srec,Arec = Q[recompute], S[recompute], A[recompute]
    Qdel,Sdel,Adel = Q[delete], S[delete], A[delete]
    
    # compute new alpha's (precision parameters) for features that are 
    # currently in model and will be recomputed
    Anew           = s[recompute]**2/ ( theta[recompute] + np.finfo(np.float32).eps)
    delta_alpha    = (1./Anew - 1./Arec)
    
    # compute change in log marginal likelihood 
    deltaL[add]       = ( Qadd**2 - Sadd ) / Sadd + np.log(Sadd/Qadd**2 )
    deltaL[recompute] = Qrec**2 / (Srec + 1. / delta_alpha) - np.log(1 + Srec*delta_alpha)
    deltaL[delete]    = Qdel**2 / (Sdel - Adel) - np.log(1 - Sdel / Adel)
    deltaL            = deltaL  / n_samples
    
    # find feature which caused largest change in likelihood
    feature_index = skip + np.argmax(deltaL[skip:])
             
    # no deletions or additions
    same_features  = np.sum( theta[~recompute] > 0) == 0 #False #
    
    # changes in precision for features already in model is below threshold
    no_delta       = np.sum( abs( Anew - Arec ) > tol ) == 0
    
    # check convergence: if no features to add or delete and small change in 
    #                    precision for current features then terminate
    converged = False
    if same_features and no_delta:
        converged = True
        print("Converge!")
        #print(sum(abs( Anew - Arec )), tol)
        return [A,converged]
    
    # if not converged update precision parameter of weights and return
    if theta[feature_index] > 0:
        A[feature_index] = s[feature_index]**2 / theta[feature_index]
        if active[feature_index] == False:
            active[feature_index] = True
    else:
        # at least two active features
        if active[feature_index] == True and np.sum(active) >= 2:
            # do not remove bias term in classification 
            # (in regression it is factored in through centering)
            if not (feature_index == 0 and clf_bias):
               active[feature_index] = False
               A[feature_index]      = np.PINF
                
    return [A,converged]


###############################################################################
#                ARD REGRESSION AND CLASSIFICATION
###############################################################################



#----------------------- Classification ARD -----------------------------------
     
     
def _logistic_cost_grad(X,Y,w,diagA):
    '''
    Calculates cost and gradient for logistic regression
    '''
    n     = X.shape[0]
    Xw    = np.dot(X,w)
    s     = expit(Xw)
    wdA   = w*diagA
    wdA[0] = 1e-3 # broad prior for bias term => almost no regularization
    cost = np.sum( Xw* (1-Y) - log_logistic(Xw)) + np.sum(w*wdA)/2 
    grad  = np.dot(X.T, s - Y) + wdA
    return [cost/n,grad/n]

def _gaussian_cost_grad(X,Y,w,diagA):
    '''
        Calculates cost and gradient for probit regression
        '''
    n = X.shape[0]
    Xw = np.dot(X, w)
    t = (Y-0.5)*2
    s = norm.cdf(Xw)
    cost = -(np.sum(np.log(s[Y==1]), 0) + \
             np.sum(np.log(1-s[Y==0]), 0))
    cost = cost + 0.5*(diagA*(w**2))

    temp = norm.pdf(Xw*t)*t/norm.cdf(Xw*t)
    grad = diagA*w - np.dot(X.T, temp)
    return cost / n, grad / n

def _gaussian_cost(X,Y,w,diagA):
    '''
        Calculates cost and gradient for probit regression
        '''
    n = X.shape[0]
    Xw = np.dot(X, w)
    t = (Y-0.5)*2
    s = norm.cdf(Xw)
    #tt = np.sum(np.log(s[Y==1]), 0)
    #tt2 = np.sum(np.log(1-s[Y==0]), 0)
    #tt3 = 0.5*np.sum((diagA*(w**2)))
    cost = -(np.sum(np.log(s[Y==1]), 0) + \
             np.sum(np.log(1-s[Y==0]), 0))
    cost = cost + 0.5*np.sum((diagA*(w**2)))#0.5*(diagA*(w**2))
    return cost / n

def _gaussian_grad(X,Y,w,diagA):
    '''
        Calculates cost and gradient for probit regression
        '''
    n = X.shape[0]
    Xw = np.dot(X, w)
    t = (Y-0.5)*2
    temp = norm.pdf(Xw*t)*t/norm.cdf(Xw*t)
    grad = diagA*w - np.dot(X.T, temp)
    return grad / n

def _gaussian_hess(X,Y,w,diagA):
    '''
        Calculates cost and gradient for probit regression
        '''
    n = X.shape[0]
    Xw = np.dot(X, w)
    t = (Y-0.5)*2
    eta = norm.pdf(Xw*t)*t/norm.cdf(Xw*t)
    B = eta*(Xw + eta)
    S = np.dot(X.T * B, X)
    np.fill_diagonal(S, np.diag(S) + diagA)
    return S / n
    

        
class ClassificationARD4(BaseEstimator,LinearClassifierMixin):
    '''
    Logistic Regression with Automatic Relevance determination (Fast Version uses 
    Sparse Bayesian Learning)
    
    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 100)
        Maximum number of iterations before termination
        
    tol: float, optional (DEFAULT = 1e-3)
        If absolute change in precision parameter for weights is below threshold
        algorithm terminates.
        
    normalize: bool, optional (DEFAULT = True)
        If True normalizes features
              
    n_iter_solver: int, optional (DEFAULT = 20)
        Maximum number of iterations before termination of solver
        
    tol_solver: float, optional (DEFAULT = 1e-5)
        Convergence threshold for solver (it is used in estimating posterior
        distribution)
        
    fit_intercept : bool, optional ( DEFAULT = True )
        If True will use intercept in the model. If set
        to false, no intercept will be used in calculations
   
    verbose : boolean, optional (DEFAULT = True)
        Verbose mode when fitting the model
        
        
    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
        
    lambda_ : float
       estimated precisions of weights
       
    active_ : array, dtype = np.bool, shape = (n_features)
       True for non-zero coefficients, False otherwise
       
    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients
        
        
    References
    ----------
    [1] Fast marginal likelihood maximisation for sparse Bayesian models (Tipping & Faul 2003)
        (http://www.miketipping.com/papers/met-fastsbl.pdf)
    [2] Analysis of sparse Bayesian learning (Tipping & Faul 2001)
        (http://www.miketipping.com/abstracts.htm#Faul:NIPS01)
    '''
    def __init__(self, n_iter=50, tol=1e-4, n_iter_solver=15, normalize=False,
                 tol_solver=1e-2, fit_intercept=INTERCEPT, fixed_intercept =UNKNOWN_PROB, verbose=False):
        self.n_iter             = n_iter
        self.tol                = tol
        print("Init ", self.tol)
        self.n_iter_solver      = n_iter_solver
        self.normalize          = normalize
        self.tol_solver         = tol_solver
        self.fit_intercept      = fit_intercept
        self.fixed_intercept = fixed_intercept
        self.verbose            = verbose
        #self.relevant_vectors = None
        self.prev_trained = False
        self.rtree_index = index.Index()
        self.relevant_vectors_dict = {}
        self.relevant_vectors_local = None
    
    
    def fit(self,X,y):
        '''
        Fits Logistic Regression with ARD
        
        Parameters
        ----------
        X: array-like of size [n_samples, n_features]
           Training data, matrix of explanatory variables
        
        y: array-like of size [n_samples] 
           Target values
           
        Returns
        -------
        self : object
            Returns self.
        '''



        X, y = check_X_y(X, y, accept_sparse = False, dtype=np.float64)

        # normalize, if required
        if self.normalize:
            self._x_mean = np.mean(X,0)
            self._x_std  = np.std(X,0)
            X            = (X - self._x_mean) / self._x_std

        # add bias term if required
        if self.fit_intercept:
            X = np.concatenate((np.ones([X.shape[0],1]),X),1)

        # preprocess targets
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_classes = 2#len(self.classes_)
        if n_classes < 2:
            raise ValueError("Need samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % self.classes_[0])
        
        # if multiclass use OVR (i.e. fit classifier for each class)
        if n_classes < 2:
            raise ValueError("Need samples of at least 2 classes")
        if n_classes > 2:
            self.coef_, self.sigma_        = [0]*n_classes,[0]*n_classes
            self.intercept_ , self.active_ = [0]*n_classes, [0]*n_classes
            self.lambda_                   = [0]*n_classes
        else:
            self.coef_, self.sigma_, self.intercept_,self.active_ = [0],[[0]],[0],[0]
            self.lambda_                                          = [0]
         
        for i in range(len(self.classes_)):
            if n_classes == 2:
                pos_class = self.classes_[1]
            else:
                pos_class = self.classes_[i]
            mask = (y == pos_class)
            y_bin = np.zeros(y.shape, dtype=np.float64)
            y_bin[mask] = 1
            active, lambda_ = self._fit(X,y_bin)
            self.intercept_[i] = self.fixed_intercept
            self.active_[i], self.lambda_[i] = active, lambda_
            # in case of binary classification fit only one classifier
            if n_classes == 2:
                break  
        self.intercept_ = np.asarray(self.intercept_)
        self.prev_trained = True
        return self
        
    
    def _fit(self,X,y):
        '''
        Fits binary classification
        '''
        n_samples,n_features = X.shape
        A         = np.PINF * np.ones(n_features)
        active    = np.zeros(n_features , dtype = np.bool)

        if self.prev_trained: # i.e. there is an existing model trained previously.
            active[0:self.prev_rvcount] = True
            A[0:self.prev_rvcount] = self.prev_A#1e-3*np.ones(self.prev_rvcount)
            active[self.prev_rvcount] = True
            A[self.prev_rvcount] = 1
            #self.prev_sigma = self.sigma_
            #self.prev_mu = self.coef_[0][self.active_[0]]
        else:
            active[0] = True
            A[0] = 1e-3  # np.finfo(np.float16).eps
        # if we fit intercept, make it active from the beginning
        #if self.fit_intercept:
        #    active[0] = True
        #    A[0]      = np.finfo(np.float16).eps


        warning_flag = 0
        start = time.time()
        for i in range(self.n_iter):
            cur = time.time()
            #print("___fit ", i, " = ", cur - start)
            Xa      =  X[:,active]
            Aa      =  A[active]

            # mean & precision of posterior distribution
            Mn,Sn,B,t_hat, cholesky = self._posterior_dist(Xa,y, Aa, keep_prev_mean = False)

            cur1 = time.time()
            print("___fit_cur1 ", i, "(approximation) = ", cur1 - cur)
            if not cholesky:
                warning_flag += 1
            
            # raise warning in case cholesky failes (but only once)
            if warning_flag == 1:
                warnings.warn(("Cholesky decomposition failed ! Algorithm uses pinvh, "
                               "which is significantly slower, if you use RVC it "
                               "is advised to change parameters of kernel"))
            #if self.prev_trained:
            #    if cholesky:
            #        Sn2 = np.dot(Sn.T, Sn)
            #        Sn2[0:self.prev_rvcount, 0:self.prev_rvcount] = self.prev_sigma[0]
            #        Sn = np.linalg.cholesky(Sn2)
            # compute quality & sparsity parameters
            s,q,S,Q = self._sparsity_quality(X,Xa,t_hat,B,A,Aa,active,Sn,cholesky)
            cur2 = time.time()
            print("___fit_cur2 ", i, "(approximation) = ", cur2 - cur1)
            # update precision parameters of coefficients
            #print self.tol
            skip = self.prev_rvcount if self.prev_trained else 0
            A,converged  = update_precisions(Q,S,q,s,A,active,self.tol,n_samples,self.fit_intercept, skip=skip)
            cur3 = time.time()
            print("___fit_cur3 ", i, "(approximation) = ", cur3 - cur2)
            print("___fit_oneloop ", i, "(approximation) = ", cur3 - cur)
            # terminate if converged
            if converged or i == self.n_iter - 1:
                break
        finish_loop = time.time()
        print("___fit_finish_loop ", i, " = ", finish_loop - start)
        finish = time.time()
        print("___fit_finish ", i, " = ", finish - start)
        return active, A
   
        
    def predict(self,X):
        '''
        Estimates target values on test set
        
        Parameters
        ----------
        X: array-like of size (n_samples_test, n_features)
           Matrix of explanatory variables
           
        Returns
        -------
        y_pred: numpy arra of size (n_samples_test,)
           Predicted values of targets
        '''
        probs, var, _, _ = self.predict_proba(X)
        indices = np.argmax(probs, axis = 1)
        y_pred  = self.classes_[indices]
        return y_pred

        
    def _decision_function_active(self,X,coef_,active_,intercept_):
        ''' Constructs decision function using only relevant features '''
        if self.normalize:
            X = (X - self._x_mean[active_]) / self._x_std[active_]
        decision = safe_sparse_dot(X,coef_[active_]) + intercept_
        return decision
        
        
    def decision_function(self,X):
        ''' 
        Computes distance to separating hyperplane between classes. The larger 
        is the absolute value of the decision function further data point is 
        from the decision boundary.
        
        Parameters
        ----------
        X: array-like of size (n_samples_test,n_features)
           Matrix of explanatory variables
          
        Returns
        -------
        decision: numpy array of size (n_samples_test,)
           Distance to decision boundary
        '''
        check_is_fitted(self, 'coef_') 
        X = check_array(X, accept_sparse=False, dtype = np.float64)
        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))
        decision = [self._decision_function_active(X[:,active],coef,active,bias) for 
                    coef,active,bias in zip(self.coef_,self.active_,self.intercept_)]
        decision = np.asarray(decision).squeeze().T
        return decision
        

    def predict_proba(self,X):
        '''
        Predicts probabilities of targets for test set using probit 
        function to approximate convolution of sigmoid and Gaussian.
        
        Parameters
        ----------
        X: array-like of size (n_samples_test,n_features)
           Matrix of explanatory variables
           
        Returns
        -------
        probs: numpy array of size (n_samples_test,)
           Estimated probabilities of target classes
        '''
        y_hat = self.decision_function(X)
        X = check_array(X, accept_sparse=False, dtype = np.float64)
        if self.normalize:
            X = (X - self._x_mean) / self._x_std
        if self.fit_intercept:
            X    = np.concatenate((np.ones([X.shape[0],1]), X),1)
        if y_hat.ndim == 1:
            pr   = self._convo_approx(X[:,self.lambda_[0]!=np.PINF],
                                           y_hat,self.sigma_[0])
            prob = np.vstack([1 - pr, pr]).T
        else:
            pr   = [self._convo_approx(X[:,idx != np.PINF],y_hat[:,i],
                        self.sigma_[i]) for i,idx in enumerate(self.lambda_) ]
            pr   = np.asarray(pr).T
            prob = pr / np.reshape(np.sum(pr, axis = 1), (pr.shape[0],1))
        return prob
        
    def _convo_approx(self,X,y_hat,sigma):
        ''' Computes approximation to convolution of sigmoid and gaussian'''
        var = np.sum(np.dot(X,sigma)*X,1)
        ks  = 1. / ( 1. + np.pi * var/ 8)**0.5
        pr  = expit(y_hat * ks)
        return pr
        
    def _sparsity_quality(self, X, Xa, y, B, A, Aa, active, Sn, cholesky):
        '''
        Calculates sparsity & quality parameters for each feature
        '''
        cur1 = time.time()
        XB = X.T * B
        #XB2 = np.matmul(X.T,np.diag(B))
        #Xdiff = XB - XB2
        bxx = np.matmul(B, X ** 2)
        #bxx2 = np.sum(XB*X.T,1)
        bxy = np.matmul(XB, y)
        YB = y * B
        cur2 = time.time()
        #print("_____sparsity___cur2_____fit ", cur2 - cur1)
        if cholesky:
            # Here Sn is inverse of lower triangular matrix, obtained from
            # cholesky decomposition
            #XB.astype(np.float32)
            #Xa.astype(np.float32)
            XBX = np.matmul(XB, Xa)
            cur21 = time.time()
            print("_______________cholesky___cur21_ shape ", XB.shape, Xa.shape)
            print("_______________cholesky___cur21_____fit ", cur21 - cur2)
            XBXS = np.matmul(XBX, Sn.T)
            cur22 = time.time()
            print("_______________cholesky___cur22_____fit ", cur22 - cur21)
            SXBY = np.matmul(Sn, np.matmul(Xa.T,YB))
            cur23 = time.time()
            print("_______________cholesky___cur23_____fit ", cur23 - cur22)
            S = bxx - np.sum(XBXS ** 2, 1)
            Q = bxy - np.sum(XBXS*SXBY.T,1)
            #S = bxx - np.sum(np.matmul(XBXS, XBXS.T), 1)
        else:
            XSX = np.dot(np.dot(Xa, Sn), Xa.T)
            S = bxx - np.sum(np.dot(XB, XSX) * XB, 1)
            XBXSX = np.matmul(XB, XSX)

            Q = bxy - np.matmul(XBXSX, YB)
        cur3 = time.time()
        print("_______________sparsity___cur3_____fit ", cur3 - cur21)
        qi = np.copy(Q)
        si = np.copy(S)
        Qa, Sa = Q[active], S[active]
        cur4 = time.time()
        print("_______________sparsity___cur4_____fit ", cur4 - cur3)
        qi[active] = Aa * Qa / (Aa - Sa)
        si[active] = Aa * Sa / (Aa - Sa)
        cur5 = time.time()
        print("_______________sparsity___cur5_____fit ", cur5 - cur4)
        return [si, qi, S, Q]

    def _posterior_dist(self, X, y, A, keep_prev_mean = True, tol_mul = 1.0):
        '''
        Uses Laplace approximation for calculating posterior distribution
        '''
        f_full = lambda w: _gaussian_cost_grad(X, y, w, A)
        f = lambda w: _gaussian_cost(X, y, w, A)
        f_grad = lambda w: _gaussian_grad(X, y, w, A)
        f_hess = lambda  w: _gaussian_hess(X, y, w, A)
        w_init = np.random.random(X.shape[1])
        #if self.prev_trained:
        #    w_init[0:self.prev_rvcount] = self.prev_mu
        #    if len(self.prev_mu) != self.prev_rvcount:
        #        print(len(self.prev_mu), self.prev_rvcount)
        # print w_init.shape
        # print X.shape
        ##print y.shape
        # print A.shape
        Mn = fmin_l_bfgs_b(f_full, x0=w_init, pgtol=tol_mul*self.tol_solver,
                           maxiter=int(self.n_iter_solver/tol_mul))[0]

        #bb = f(w_init)
        #aa = f_grad(w_init)
        #cc = f_full(w_init)
        #dd = f_hess(w_init)
        #opts = {'xtol': self.tol_solver, 'maxiter': self.n_iter_solver}
        #Mn = minimize(f, x0=w_init,  method="Newton-CG", jac=f_grad, hess=f_hess, options=opts)
        #Mn = Mn.x
        #if self.prev_trained and keep_prev_mean:
        #    Mn[0:self.prev_rvcount] = self.prev_mu
        Xm_nobias = np.dot(X, Mn)
        Xm = Xm_nobias + self.fixed_intercept
        #s = norm.cdf(Xm)
        t = (y - 0.5) * 2
        eta = norm.pdf(t * Xm) * t / norm.cdf(Xm * t) + 1e-300
        #B = eta * (Xm_nobias + eta)
        B = eta * (Xm + eta)
        # B         = logistic._pdf(Xm) # avoids underflow
        S = np.dot(X.T * B, X)
        np.fill_diagonal(S, np.diag(S) + A)
        t_hat = Xm_nobias + eta / B
        # t_hat = y - s
        cholesky = True
        # try using Cholesky , if it fails then fall back on pinvh
        try:
            R = np.linalg.cholesky(S)
            Sn = solve_triangular(R, np.eye(A.shape[0]),
                                  check_finite=False, lower=True)
        except LinAlgError:
            Sn = pinvh(S)
            cholesky = False
        return [Mn, Sn, B, t_hat, cholesky]

###############################################################################
#                  Relevance Vector Machine: RVR and RVC
###############################################################################



def get_kernel( X, Y, gamma, degree, coef0, kernel, kernel_params ):
    '''
    Calculates kernelised features for RVR and RVC
    '''
    if callable(kernel):
        params = kernel_params or {}
    else:
        params = {"gamma": gamma,
                  "degree": degree,
                  "coef0": coef0  }
    return pairwise_kernels(X, Y, metric=kernel,
                            filter_params=True, **params)
                            

class RVC4(ClassificationARD4):
    '''
    Relevance Vector Classifier (Fast Version, uses Sparse Bayesian Learning )
        
    
    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 100)
        Maximum number of iterations before termination
        
    tol: float, optional (DEFAULT = 1e-4)
        If absolute change in precision parameter for weights is below tol, then
        the algorithm terminates.

    n_iter_solver: int, optional (DEFAULT = 15)
        Maximum number of iterations before termination of solver
        
    tol_solver: float, optional (DEFAULT = 1e-4)
        Convergence threshold for solver (it is used in estimating posterior
        distribution)
        
    fit_intercept : bool, optional ( DEFAULT = True )
        If True will use intercept in the model

    verbose : boolean, optional (DEFAULT = True)
        Verbose mode when fitting the model
        
    kernel: str, optional (DEFAULT = 'rbf')
        Type of kernel to be used (all kernels: ['rbf' | 'poly' | 'sigmoid']
    
    degree : int, (DEFAULT = 3)
        Degree for poly kernels. Ignored by other kernels.
        
    gamma : float, optional (DEFAULT = 1/n_features)
        Kernel coefficient for rbf and poly kernels, ignored by other kernels
        
    coef0 : float, optional (DEFAULT = 0.1)
        Independent term in poly and sigmoid kernels, ignored by other kernels
        
    kernel_params : mapping of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as
        callable object, ignored by other kernels
        
        
    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the model (mean of posterior distribution)
        
    lambda_ : float
       Estimated precisions of weights
       
    active_ : array, dtype = np.bool, shape = (n_features)
       True for non-zero coefficients, False otherwise
       
    sigma_ : array, shape = (n_features, n_features)
       Estimated covariance matrix of the weights, computed only for non-zero 
       coefficients
       
       
    References
    ----------
    [1] Fast marginal likelihood maximisation for sparse Bayesian models (Tipping & Faul 2003)
        (http://www.miketipping.com/papers/met-fastsbl.pdf)
    [2] Analysis of sparse Bayesian learning (Tipping & Faul 2001)
        (http://www.miketipping.com/abstracts.htm#Faul:NIPS01)
    '''
    
    def __init__(self, n_iter = 200, tol = 1e-1, n_iter_solver = 50, tol_solver = 1e-3,
                 fit_intercept = INTERCEPT, fixed_intercept = UNKNOWN_PROB, verbose = False, kernel = 'rbf', degree = 2,
                 gamma  = None, coef0  = 0, kernel_params = None):
        super(RVC4,self).__init__(n_iter,tol,n_iter_solver,False,tol_solver,
                                 fit_intercept, fixed_intercept, verbose)
        print("Init RVC", self.tol)
        self.kernel        = kernel
        self.degree        = degree
        self.gamma         = gamma
        self.coef0         = coef0
        self.kernel_params = kernel_params
        self.prev_X = None
        self.prev_y = None
        
        
    def fit(self,X,y):
        '''
        Fit Relevance Vector Classifier
        
        Parameters
        -----------
        X: array-like of size [n_samples, n_features]
           Training data, matrix of explanatory variables
        
        y: array-like of size [n_samples, n_features] 
           Target values
           
        Returns
        -------
        self: object
           self
        '''
        X_orig = np.copy(X)
        y_orig = np.copy(y)
        self.cur_X = X_orig.tolist()
        self.cur_y = y_orig.tolist()

        if self.prev_trained:
            #xa = X[0,:].tolist()
            #xa2 = X[0, :]
            #a = [7.75, -10.0] not in self.prev_X
            X_mean = np.mean(X, axis=0)
            nearest_rv = self.rtree_index.nearest((X_mean[0], X_mean[1], X_mean[0], X_mean[1]), 200, objects=True)
            local_rv = []
            local_rv_label = []
            local_A = []
            for rv in nearest_rv:
                local_rv.append([rv.bbox[0], rv.bbox[1]])
                local_rv_label.append(self.relevant_vectors_dict[(rv.bbox[0], rv.bbox[1])][0])
                local_A.append(self.relevant_vectors_dict[(rv.bbox[0], rv.bbox[1])][1])
            self.relevant_vectors_local = local_rv
            self.prev_rvcount = len(local_rv)
            self.prev_A = local_A
            #prev_rv = self.relevant_vectors_[0].tolist()
            Xdiff = np.array([X[i,:] if X[i,:].tolist() not in self.prev_X and X[i,:].tolist() not in local_rv else None for i in range(len(X))]) #np.setdiff1d(X, self.prev_X)
            ydiff = np.array([y[i] if X[i, :].tolist() not in self.prev_X and X[i,:].tolist() not in local_rv else None for i in range(len(X))])
            Xdiff = Xdiff[ydiff != None]
            ydiff = ydiff[ydiff != None]
            if len(ydiff) > 0:
                print("New data")
            else:
                print("Data is the same! Skip...")
                #self.prev_X = X_orig
                #self.prev_y = y_orig
                return self
            newdatalen = len(Xdiff)
            Xdiff = np.concatenate(Xdiff).reshape((newdatalen,2))
            ydiff = ydiff.astype(np.int64)
            ydiff_pos = ydiff[ydiff > 0]
            Xdiff_pos = Xdiff[ydiff > 0, :]
            pos_portion = sum(ydiff[ydiff > 0])
            neg_portion = len(ydiff) - pos_portion
            #if (pos_portion > 0):
            #    r = max(int(neg_portion / pos_portion), 0)
            #    for i in range(r):
            #        Xdiff = np.vstack((Xdiff, Xdiff_pos))
            #        ydiff = np.hstack((ydiff, ydiff_pos))
            X = np.vstack((local_rv, Xdiff))
            y = np.hstack((local_rv_label, ydiff))
            #X = np.vstack((self.relevant_vectors_[0], X_orig))
            #y = np.hstack((self.rv_labels[0], y_orig))

        X,y = check_X_y(X,y, accept_sparse = False, dtype = np.float64)


        # kernelise features
        K = get_kernel( X, X, self.gamma, self.degree, self.coef0, 
                       self.kernel, self.kernel_params)
        # use fit method of ClassificationARD
        _ = super(RVC4,self).fit(K,y)
        self.relevant_  = [np.where(active==True)[0] for active in self.active_]
        if X.ndim == 1:
            self.relevant_vectors_ = [ X[relevant_] for relevant_ in self.relevant_]
            self.rv_labels = [ y[relevant_] for relevant_ in self.relevant_]
            self.rv_A = [self.lambda_[0][relevant_] for relevant_ in self.relevant_]
        else:
            self.relevant_vectors_ = [ X[relevant_,:] for relevant_ in self.relevant_ ]
            self.rv_labels = [y[relevant_] for relevant_ in self.relevant_]
            self.rv_A = [self.lambda_[0][relevant_] for relevant_ in self.relevant_]


        rv_local_list = self.relevant_vectors_[0].tolist()

        count_dict = {}
        for i in range(len(rv_local_list)):
            r = (rv_local_list[i][0], rv_local_list[i][1])
            if (r[0], r[1]) not in count_dict:
                count_dict[(r[0], r[1])] = 1
            else:
                count_dict[(r[0], r[1])] = count_dict[(r[0], r[1])] + 1
            if r not in self.relevant_vectors_dict.keys():
                self.relevant_vectors_dict[(r[0], r[1])] = (self.rv_labels[0][i], self.rv_A[0][i])
                self.rtree_index.insert(int(2*(r[0]*10000 + 2*r[1])), (r[0], r[1], r[0], r[1]))
        tree_nodes = list(self.rtree_index.intersection((-100000.0, -100000.0, 100000.0, 100000.0)))
        print("############################################### rv_local_list length = ", len(rv_local_list))
        print("############################################### rv_global length = ", len(self.relevant_vectors_dict.keys()))
        if len(rv_local_list) != len(self.relevant_vectors_dict.keys()):
            print("smt not right")
            #count_dict = {}
            #for i in range(len(X)):
            #    r = (X[i][0], X[i][1])
            #    if (r[0], r[1]) not in count_dict:
            #        count_dict[(r[0], r[1])] = 1
            #    else:
            #        count_dict[(r[0], r[1])] = count_dict[(r[0], r[1])] + 1
            #print("smt not right")
        ############################## Recalculate weight mean and cov
            # kernelise features
        stime_ext = time.time()
        self.recover_posterior_dist()
        etime_ext = time.time()
        print("@@@@@@@@@@@@@@@@@@@@@@@@ EXTRA TIME:", etime_ext - stime_ext)
        self.prev_X = self.cur_X
        self.prev_y = self.cur_y
        return self

    def recover_posterior_dist(self, tol_factor = 0.1, a_thres = 1.0):
        self.all_rv_X = []
        self.all_rv_y = []
        self.all_rv_A = []
        keys_list = list(self.relevant_vectors_dict.keys())
        for r in keys_list:
            if self.relevant_vectors_dict[r][1] > a_thres:
                self.relevant_vectors_dict.pop(r)
                self.rtree_index.delete(int(2 * (r[0] * 10000 + 2 * r[1])), (r[0], r[1], r[0], r[1]))
        for r in self.relevant_vectors_dict.keys():
            self.all_rv_X.append(np.array([r[0], r[1]]))
            self.all_rv_y .append(self.relevant_vectors_dict[r][0])
            self.all_rv_A.append(self.relevant_vectors_dict[r][1])
        all_rv_K = get_kernel(self.all_rv_X, self.all_rv_X, self.gamma, self.degree, self.coef0,
                              self.kernel, self.kernel_params)
        all_rv_y = np.array(self.all_rv_y)
        all_rv_A = np.array(self.all_rv_A)
        Mn, Sn, B, t_hat, cholesky = self._posterior_dist(all_rv_K, all_rv_y, all_rv_A, keep_prev_mean=False,
                                                          tol_mul=tol_factor)
        # in case Sn is inverse of lower triangular matrix of Cholesky decomposition
        # compute covariance using formula Sn  = np.dot(Rinverse.T , Rinverse)
        if cholesky:
            Sn = np.dot(Sn.T, Sn)
        self.Mn = Mn
        self.Sn = Sn
        return self.all_rv_X, all_rv_y, all_rv_A

    def _decision_function_active(self,X,coef_,intercept_):
        ''' Constructs decision function using only relevant features '''
        decision = safe_sparse_dot(X,self.Mn) + intercept_
        return decision

    def decision_function(self, X):
        '''
        Computes distance to separating hyperplane between classes. The larger
        is the absolute value of the decision function further data point is
        from the decision boundary.

        Parameters
        ----------
        X: array-like of size (n_samples_test,n_features)
           Matrix of explanatory variables

        Returns
        -------
        decision: numpy array of size (n_samples_test,)
           Distance to decision boundary
        '''
        check_is_fitted(self, 'coef_')
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        n_features = len(self.all_rv_X[0])#self.relevant_vectors_[0].shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))
        kernel = lambda rvs: get_kernel(X, rvs, self.gamma, self.degree,
                                        self.coef0, self.kernel, self.kernel_params)
        decision = []
        K = []
        if len(self.all_rv_X) == 0:
            decision.append(np.ones(X.shape[0]) * self.intercept_)
        else:
            k = kernel(self.all_rv_X)
            decision.append(self._decision_function_active(k, self.Mn, self.intercept_))
            K.append(k)
        decision = np.asarray(decision).squeeze().T
        K = np.array(K[0])
        if self.fit_intercept:
            K = np.concatenate((np.ones([K.shape[0], 1]), K), 1)
        # print(K.shape)
        # covar = np.matmul(np.matmul(K, S), K.T)
        # var2 = np.diag(covar)
        var = np.sum(np.matmul(K, self.Sn) * K, axis=1)
        # diff = var - var2
        return decision, var

    def get_feature(self, X):
        '''
        Computes distance to separating hyperplane between classes. The larger
        is the absolute value of the decision function further data point is
        from the decision boundary.

        Parameters
        ----------
        X: array-like of size (n_samples_test,n_features)
           Matrix of explanatory variables

        Returns
        -------
        decision: numpy array of size (n_samples_test,)
           Distance to decision boundary
        '''
        check_is_fitted(self, 'coef_')
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        n_features = len(self.all_rv_X[0])  # self.relevant_vectors_[0].shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))
        kernel = lambda rvs: get_kernel(X, rvs, self.gamma, self.degree,
                                        self.coef0, self.kernel, self.kernel_params)
        K = kernel(self.all_rv_X)
        w = self.Mn
        if self.fit_intercept:
            K = np.concatenate((np.ones([K.shape[0], 1]), K), 1)
        mu = self.all_rv_X[0]
        return K, w, mu

    def predict_proba(self, X):
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
        decision, var = self.decision_function(X)
        #print("Decision value!!!!!!!!!!!!!!!", decision)
        #print("Var value!!!!!!!!!!!!!!!", var)
        prob = norm.cdf(decision / np.sqrt(var + 1))
        if prob.ndim == 1:
            prob = np.vstack([1 - prob, prob]).T
        prob = prob / np.reshape(np.sum(prob, axis=1), (prob.shape[0], 1))
        return prob, var, decision,  np.sqrt(var + 1)

#    def predict_proba_grad(self, X, dX):
#
    def predict_upperbound(self, X, c = THRESHOLD):
        K, w, mu = self.get_feature(X)
        S =  np.abs(self.Sn)
        #trace = np.trace(S)
        #sum_col = np.sum(S,axis=1)
        #lambda_max_sqrt = np.sqrt(np.max(sum_col))
        lambda_max_sqrt = np.sqrt(LA.norm(S, ord=2))
        #temp1 = norm.cdf(-0.1)
        #temp2 = 0.5*(1 + erf(-0.1/np.sqrt(2)))
        #upperbound = np.matmul(K, w)
        #upperbound = upperbound + self.intercept_
        w2 = w - c*lambda_max_sqrt
        upperbound = np.matmul(K,w2)
        upperbound = upperbound + self.intercept_ - c
        self.corrected_weights = w2
        self.orig_weights = w
        upperbound2, pos_term, neg_term, _, _, _ = self.collision_checking(K, mu, w2, X)
        upperbound2 = upperbound2.flatten()
        upperbound2 = upperbound2 + self.intercept_ - c
        upperbound4 = pos_term - 2*np.sqrt(neg_term*(self.intercept_ - c))
        #n = np.max(1, (self.intercept_ - c)/neg_term) + 1
        n = np.array([1 if n_i < 1 else n_i for n_i in (self.intercept_ - c)/neg_term]) + 1
        n = 1e189*np.ones(len(upperbound))
        temp = (c - self.intercept_)/(n-1)
        upperbound3 = pos_term[:,0] - n*np.power(-neg_term[:,0], 1/n)*np.power(temp, (n-1)/n)
        return upperbound, upperbound2, upperbound3, upperbound4

    def predict_upperbound_line(self, X, A, B, e=THRESHOLD):
        K, w, mu = self.get_feature(X)
        S = self.Sn
        # trace = np.trace(S)
        # sum_col = np.sum(S,axis=1)
        # lambda_max_sqrt = np.sqrt(np.max(sum_col))
        lambda_max_sqrt = np.sqrt(LA.norm(S, ord=2))
        # temp1 = norm.cdf(-0.1)
        # temp2 = 0.5*(1 + erf(-0.1/np.sqrt(2)))
        # upperbound = np.matmul(K, w)
        # upperbound = upperbound + self.intercept_
        w2 = w - e * lambda_max_sqrt
        _, pos_term, neg_term, neg_coff, neg_dist, _ = self.collision_checking(K, mu, w2, X)
        A = A.reshape([1,2])
        dist_to_A = get_kernel( X, A, self.gamma, self.degree, self.coef0,
                       self.kernel, self.kernel_params)
        upperbound5 = pos_term - 2 * np.sqrt(neg_coff * (e - self.intercept_))*dist_to_A*neg_dist[0]
        #### Find intersection
        v = B - A
        intersection = self.check_line(A, w2, v[0], e=e)
        return upperbound5, intersection
    def get_radius(self, X, A, e=THRESHOLD):
        K, w, mu = self.get_feature(X)
        S = self.Sn
        # trace = np.trace(S)
        # sum_col = np.sum(S,axis=1)
        # lambda_max_sqrt = np.sqrt(np.max(sum_col))
        lambda_max_sqrt = np.sqrt(LA.norm(S, ord=2))
        # temp1 = norm.cdf(-0.1)
        # temp2 = 0.5*(1 + erf(-0.1/np.sqrt(2)))
        # upperbound = np.matmul(K, w)
        # upperbound = upperbound + self.intercept_
        w2 = w - e * lambda_max_sqrt
        radius = self.check_radius(A, w2, e=e)
        return radius
    '''
    def predict_upperbound_line_cs(self, X, A, c=-0.1):
        K, w, mu = self.get_feature(X)
        S = np.abs(self.sigma_[0])
        # trace = np.trace(S)
        # sum_col = np.sum(S,axis=1)
        # lambda_max_sqrt = np.sqrt(np.max(sum_col))
        lambda_max_sqrt = np.sqrt(LA.norm(S, ord=2))
        # temp1 = norm.cdf(-0.1)
        # temp2 = 0.5*(1 + erf(-0.1/np.sqrt(2)))
        # upperbound = np.matmul(K, w)
        # upperbound = upperbound + self.intercept_
        w2 = w - c * lambda_max_sqrt
        _, pos_term, neg_term, neg_coff, neg_dist, neg_min_idx = self.collision_checking(K, mu, w2, X)
        A = A.reshape([1,2])
        dist_to_A = get_kernel( X, A, self.gamma, self.degree, self.coef0,
                       self.kernel, self.kernel_params)
        upperbound6 = pos_term - 2 * np.sqrt(neg_coff * (c - self.intercept_))*dist_to_A*neg_dist[0]
        return upperbound6
    '''
    def check_line(self, s0, alpha, v, e=THRESHOLD, n = 1, tighter_bound = True):
        total_plus = np.sum(alpha[alpha > 0])
        rvs = self.all_rv_X
        tau = None
        for j in range(len(alpha)):
            if alpha[j] < 0:
                continue
            if tighter_bound:
                temp_max = -1
                for k in range(len(alpha)):
                    if alpha[k] > 0:
                        continue
                    beta = np.log(n+1) + (n/(n+1))*np.log((e - self.intercept_[0])/n) + np.log(-alpha[k])/(n+1) -np.log(total_plus)
                    # Quadratic conditions
                    a = -n*(v[0]**2 + v[1]**2)*self.gamma
                    m1 = n*s0
                    m2 = rvs[k]
                    m3 = (n+1)*rvs[j]
                    temp1 = n*s0 + rvs[k] - (n+1)*rvs[j]
                    b = -2*np.dot(v,temp1[0])*self.gamma
                    temp = s0 - rvs[j]
                    temp2 = np.linalg.norm(temp)**2
                    c = (-(n+1)*(np.linalg.norm(s0 - rvs[j])**2) + np.linalg.norm(s0 - rvs[k])**2)*self.gamma - (n+1)*beta
                    delta = b**2 - 4*a*c
                    if delta <= 0:
                        t1 = 100000
                    else:
                        t1 = (-b + np.sqrt(delta))/(2*a)
                        t2 = (-b - np.sqrt(delta)) / (2 * a)

                    if t1 >= 0:
                        if t1 > temp_max:
                            temp_max = t1
                    elif t2 < 0:
                        temp_max = 100000
                if temp_max < 0:
                    tau = -1
                    break
                elif tau is None or temp_max < tau:
                    tau = temp_max
        return tau

    def check_radius(self, s0, alpha, e=THRESHOLD, n = 1, tighter_bound = True):
        total_plus = np.sum(alpha[alpha > 0])
        rvs = self.all_rv_X
        tau = None
        for j in range(len(alpha)):
            if alpha[j] < 0:
                continue
            if tighter_bound:
                temp_max = -1
                for k in range(len(alpha)):
                    if alpha[k] > 0:
                        continue
                    beta = np.log(n+1) + (n/(n+1))*np.log((e - self.intercept_[0])/n) + np.log(-alpha[k])/(n+1) -np.log(total_plus)
                    # Quadratic conditions
                    a = -n
                    temp1 = n*s0 + rvs[k] - (n+1)*rvs[j]
                    b = 2*np.linalg.norm(temp1)*np.sqrt(self.gamma)
                    temp = s0 - rvs[j]
                    temp2 = np.linalg.norm(temp)**2
                    c = (-(n+1)*(np.linalg.norm(s0 - rvs[j])**2) + np.linalg.norm(s0 - rvs[k])**2)*self.gamma - (n+1)*beta
                    delta = b**2 - 4*a*c
                    if delta <= 0:
                        t1 = 100000
                    else:
                        t1 = (-b + np.sqrt(delta))/(2*a)
                        t2 = (-b - np.sqrt(delta)) / (2 * a)

                    if t1 >= 0:
                        if t1 > temp_max:
                            temp_max = t1
                    elif t2 < 0:
                        temp_max = 100000
                if temp_max < 0:
                    tau = -1
                    break
                elif tau is None or temp_max < tau:
                    tau = temp_max
        return tau/np.sqrt(self.gamma)








    def collision_checking(self, K, mu, alpha, X_test):
        f_predict_plus = np.zeros([X_test.shape[0], 1], dtype=np.float64)
        f_predict_minus = np.zeros([X_test.shape[0], 1], dtype=np.float64)
        f_predict_upperbound = np.zeros([X_test.shape[0], 1], dtype=np.float64)
        f_predict_upperbound_pos = np.zeros([X_test.shape[0], 1], dtype=np.float64)
        f_predict_upperbound_neg = np.zeros([X_test.shape[0], 1], dtype=np.float64)
        neg_coff = np.zeros([X_test.shape[0], 1], dtype=np.float64)
        neg_dist = np.zeros([X_test.shape[0], 1], dtype=np.float64)
        neg_min_idx = np.zeros([X_test.shape[0], 1], dtype=np.float64)
        minus_apha = alpha[alpha < 0]
        plus_apha = alpha[alpha > 0]
        plus_apha_total = np.sum(plus_apha)
        minus_alpha_total = np.abs(np.sum(minus_apha))
        for i in range(X_test.shape[0]):
            G_row = np.zeros([len(alpha), 1])
            for j in range(len(alpha)):
                if alpha[j] >= 0:
                    continue
                G_row[j] = K[i, j]
                # G_row[j] = self.kernel(X[j, :], x_test, gamma=alpha[j]/plus_alpha_total) if alpha[j] > 0 else 0
            # print("total_dist", total_dist)
            f_predict_minus[i][0] = np.max(G_row)  # minus_alpha_total*np.exp(-self.gamma*total_dist)
            minus_idx_min = np.argmax(G_row)
            G_row = np.zeros([len(alpha), 1])
            for j in range(len(alpha)):
                if alpha[j] <= 0:
                    continue
                G_row[j] = K[i, j]
            f_predict_plus[i][0] = np.max(G_row)
            f_predict_upperbound[i][0] = plus_apha_total * f_predict_plus[i][0] + alpha[minus_idx_min] * \
                                         f_predict_minus[i][0]
            f_predict_upperbound_pos[i][0] = plus_apha_total * f_predict_plus[i][0]
            f_predict_upperbound_neg[i][0] = alpha[minus_idx_min] * f_predict_minus[i][0]
            neg_coff[i][0] = -alpha[minus_idx_min]
            neg_dist[i][0] = f_predict_minus[i][0]
            neg_min_idx[i][0] = minus_idx_min
        return f_predict_upperbound, f_predict_upperbound_pos , f_predict_upperbound_neg, neg_coff, neg_dist, neg_min_idx





