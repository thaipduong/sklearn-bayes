from sklearn.utils.estimator_checks import check_estimator
from skbayes.rvm_ard_models import RVR,RVC
check_estimator(RVC)
check_estimator(RVR)
print("All test are passed ...")