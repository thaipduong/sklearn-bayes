# -*- coding: utf-8 -*-
'''
Fast Relevance Vector Machine & ARD ( Tipping and Faul (2003) )

    IMPLEMENTED ALGORITHMS:
    =======================
    Relevance Vector Regression : RVR
    Relevance Vector Classifier : RVC
    Classification ARD          : ClassificationARD
    Regression ARD              : RegressionARD
    Variational Regression ARD  : VBRegressionARD
    Variational Logistic 
                 Regression ARD : VBClassificationARD
'''
from .fast_rvm import RVR,RVC,ClassificationARD,RegressionARD
from .fast_rvm2 import RVR2,RVC2,ClassificationARD2,RegressionARD2
from .fast_rvm_online import RVR3,RVC3,ClassificationARD3,RegressionARD3, RVSet
from .fast_rvm_online_rtree import ClassificationARD4,RVC4
from .fast_rvm_online_rtree_intel import ClassificationARD5,RVC5
from .vrvm import VBRegressionARD, VBClassificationARD

__all__ = ['RVR','RVC','ClassificationARD','RegressionARD', 'RVR2','RVC2','ClassificationARD2','RegressionARD2', 'RVR3','RVC3','ClassificationARD3','RegressionARD3','VBRegressionARD', 'RVSet',
           'VBClassificationARD']


