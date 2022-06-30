# -*- coding: utf-8 -*-
"""
# Check the versions of libraries

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
"""

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, LeavePGroupsOut, GroupKFold, cross_validate, cross_val_predict, GridSearchCV, LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import ElasticNet, BayesianRidge, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
import numpy as np
from numpy import arange
import pandas as pd
import random

# Load dataset
#path = 'variables2.3_strongCorrelation.csv'
path = 'STRIDE_ARRA.csv'
#data = read_csv(path)
data = read_csv(path, dtype='float', na_values='?')
array0 = data.values

# Delete rows with missing values
#r=np.where(array0=='?')
#array = np.delete(array0, r[0], 0)
#array = array0[~np.isnan(array0).any(axis=1)]
array = array0

# Hold-out test set (one random participant)
random.seed(10)
p=random.randint(1, 50)
array_test = array[array[:,0]==p] #participant 37
array_train = array[array[:,0]!=p]
x_test=array_test[:,2:]
y_test=array_test[:,1]
groups_test=array_test[:,0]
#x_train=array_train[:,2:]
x_train=array_train[:,[2,3,6,7,8]] #without stance time and gait cycle speed parameters
y_train=array_train[:,1]
groups_train=array_train[:,0]

# Counter instances
"""
pyplot.bar(arange(15,35,1), [np.count_nonzero(y_train==15),
                np.count_nonzero(y_train==16),
                np.count_nonzero(y_train==17),
                np.count_nonzero(y_train==18),
                np.count_nonzero(y_train==19),
                np.count_nonzero(y_train==20),
                np.count_nonzero(y_train==21),
                np.count_nonzero(y_train==22),
                np.count_nonzero(y_train==23),
                np.count_nonzero(y_train==24),
                np.count_nonzero(y_train==25),
                np.count_nonzero(y_train==26),
                np.count_nonzero(y_train==27),
                np.count_nonzero(y_train==28),
                np.count_nonzero(y_train==29),
                np.count_nonzero(y_train==30),
                np.count_nonzero(y_train==31),
                np.count_nonzero(y_train==32),
                np.count_nonzero(y_train==33),
                np.count_nonzero(y_train==34)])
pyplot.xlabel('Measured FM-LE')
pyplot.ylabel('Count instances')
pyplot.show()
"""

# Counter subjects
"""
pyplot.bar(arange(15,35,1), [len(set(array_train[y_train==15,0])),
                len(set(array_train[y_train==16,0])),
                len(set(array_train[y_train==17,0])),
                len(set(array_train[y_train==18,0])),
                len(set(array_train[y_train==19,0])),
                len(set(array_train[y_train==20,0])),
                len(set(array_train[y_train==21,0])),
                len(set(array_train[y_train==22,0])),
                len(set(array_train[y_train==23,0])),
                len(set(array_train[y_train==24,0])),
                len(set(array_train[y_train==25,0])),
                len(set(array_train[y_train==26,0])),
                len(set(array_train[y_train==27,0])),
                len(set(array_train[y_train==28,0])),
                len(set(array_train[y_train==29,0])),
                len(set(array_train[y_train==30,0])),
                len(set(array_train[y_train==31,0])),
                len(set(array_train[y_train==32,0])),
                len(set(array_train[y_train==33,0])),
                len(set(array_train[y_train==34,0]))])
pyplot.xlabel('Measured FM-LE')
pyplot.ylabel('Count subjects')
pyplot.show()
"""

# boxplot for each Fugl-Meyer and input parameter 
"""
r=np.where(array[:,0]==1)
array_subj = np.mean(array[r], axis=0).reshape(1,-1)
i=2
while i < 56:
    r=np.where(array[:,0]==i)
    if len(r[0])>0:
        array_subj = np.concatenate((array_subj,np.mean(array[r], axis=0).reshape(1,-1)))
    i=i+1
array_subj_=array_subj[:,2:]
#x_train=array_subj_
x_train=array_subj[:,[2,3,6,7,8]] #without stance time and gait cycle speed
y_train=array_subj[:,1]
groups_train=array_subj[:,0]
metrics=["step_length_NP", "step_length_P", "stance_time_NP", "stance_time_P", "walking_speed", "stride_length", "stride_time_P", "gait_cycle_speed"]
no_met = len(metrics)
i=0
while i < no_met:  
    #all gait cycles
    pyplot.boxplot([x_train[y_train==15,i],
                    x_train[y_train==16,i],
                    x_train[y_train==17,i],
                    x_train[y_train==18,i],
                    x_train[y_train==19,i],
                    x_train[y_train==20,i],
                    x_train[y_train==21,i],
                    x_train[y_train==22,i],
                    x_train[y_train==23,i],
                    x_train[y_train==24,i],
                    x_train[y_train==25,i],
                    x_train[y_train==26,i],
                    x_train[y_train==27,i],
                    x_train[y_train==28,i],
                    x_train[y_train==29,i],
                    x_train[y_train==30,i],
                    x_train[y_train==31,i],
                    x_train[y_train==32,i],
                    x_train[y_train==33,i],
                    x_train[y_train==34,i]], labels=arange(15,35,1))
    #mean gait cycles per subject
    pyplot.boxplot([array_subj_[array_subj[:,1]==15,i],
                    array_subj_[array_subj[:,1]==16,i],
                    array_subj_[array_subj[:,1]==17,i],
                    array_subj_[array_subj[:,1]==18,i],
                    array_subj_[array_subj[:,1]==19,i],
                    array_subj_[array_subj[:,1]==20,i],
                    array_subj_[array_subj[:,1]==21,i],
                    array_subj_[array_subj[:,1]==22,i],
                    array_subj_[array_subj[:,1]==23,i],
                    array_subj_[array_subj[:,1]==24,i],
                    array_subj_[array_subj[:,1]==25,i],
                    array_subj_[array_subj[:,1]==26,i],
                    array_subj_[array_subj[:,1]==27,i],
                    array_subj_[array_subj[:,1]==28,i],
                    array_subj_[array_subj[:,1]==29,i],
                    array_subj_[array_subj[:,1]==30,i],
                    array_subj_[array_subj[:,1]==31,i],
                    array_subj_[array_subj[:,1]==32,i],
                    array_subj_[array_subj[:,1]==33,i],
                    array_subj_[array_subj[:,1]==34,i]], labels=arange(15,35,1))
    pyplot.xlabel('Measured FM-LE')
    pyplot.ylabel(metrics[i])
    pyplot.title('Mean gait cycles per subject')
    pyplot.show()
    i=i+1
"""

# Leave one subject out cross-validation (45 folds)
#logo=LeaveOneGroupOut()
#print(logo.get_n_splits(groups=groups_train))
logo=LeaveOneOut()
"""
for train_index, val_index in logo.split(x_train, y_train, groups_train):
    print("TRAIN:", train_index[0], "VAL:", val_index[0])
    x_train2, x_val = x_train[train_index], x_train[val_index]
    y_train2, y_val = y_train[train_index], y_train[val_index]
    #print(x_train2, x_val, y_train2, y_val)
"""

# Models
models = []
"""
fit_intercept = True (constant value of linear equation)
"""
#models.append(('LR', LinearRegression(),{})) #alpha=0 
"""
alpha = 1.0 (Regularization strength)
fit_intercept = True (constant value of linear equation)
max_inter = 1000
tol = 1e-4 (precision of solution)
random_state = 10 (used when solver == ‘sag’ or ‘saga’)
gridsearch reference: https://machinelearningmastery.com/lasso-regression-with-python/
"""
#models.append(('Lasso', Lasso(random_state=10), {"estimator__alpha": arange(0, 1, 0.01)})) # l1 regularization
"""
alpha = 1.0 (Regularization strength)
fit_intercept = True (constant value of linear equation)
max_inter (For ‘sparse_cg’ and ‘lsqr’ solvers, the default value is determined by scipy.sparse.linalg. For ‘sag’ solver, the default value is 1000. For ‘lbfgs’ solver, the default value is 15000.)
tol = 1e-3 (precision of solution)
solver = auto (chooses the solver automatically based on the type of data)
random_state = 10 (used when solver == ‘sag’ or ‘saga’)
gridsearch reference: https://machinelearningmastery.com/ridge-regression-with-python/
"""
#list_ = list(arange(1000, 20000, 1000))
#list_.append(0)
#models.append(('Ridge', Ridge(random_state=10), {"estimator__alpha": list_})) # l2 regularization
"""
alpha = 1.0 (Regularization strength)
l1_ratio = 0.5 (The ElasticNet mixing parameter. For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty.)
fit_intercept = True (constant value of linear equation)
max_inter = 1000
tol = 1e-4 (precision of solution)
random_state = 10 (used when solver == ‘sag’ or ‘saga’)
gridsearch reference: https://machinelearningmastery.com/elastic-net-regression-in-python/ (removed alpha=0 because of a warning; removed l1_ratio<=0.01 because https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
"""
#models.append(('EN', ElasticNet(random_state=10), {"estimator__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]})) # l1 & l2 regularization
#models.append(('EN', ElasticNet(random_state=10), {"estimator__l1_ratio": arange(0.02, 1, 0.01)})) # l1 & l2 regularization
"""
max_inter = 300
tol = 1e-3 (precision of solution)
lambda_init (Initial value for lambda (precision of the weights). If not set, lambda_init is 1.)
alpha_init (Initial value for alpha (precision of the noise). If not set, alpha_init is 1/Var(y).)
alpha_1 = 1e-6
alpha_2 = 1e-6
lambda_1 = 1e-6
lambda_2 = 1e-6
fit_intercept = True (constant value of linear equation)
"""
#models.append(('B_Ridge', BayesianRidge(), {"estimator__alpha_1": arange(0, 1, 0.01)})) 
#list_ = list(arange(10000000, 100000000, 10000000))
#list_.append(0)
#models.append(('B_Ridge', BayesianRidge(alpha_1=0), {"estimator__alpha_2": list_})) 
#models.append(('B_Ridge', BayesianRidge(alpha_1=0, alpha_2=10000000), {"estimator__lambda_1": arange(0, 1, 0.01)})) 
#models.append(('B_Ridge', BayesianRidge(alpha_1=0, alpha_2=10000000, lambda_1=0.02), {"estimator__lambda_2": arange(0, 1, 0.01)})) 
"""
criterion = squared_error (The function to measure the quality of a split)
splitter = best (The strategy used to choose the split at each node.)
max_depth = None (If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.)
min_samples_split = 2 (The minimum number of samples required to split an internal node.)
min_samples_leaf = 1 (The minimum number of samples required to be at a leaf node.)
min_weight_fraction_leaf = 0.0 (The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.)
max_features = None (If None, then max_features=n_features.)
random_state = 10
max_leaf_nodes = None (If None then unlimited number of leaf nodes.)
min_impurity_decrease = 0.0 (A node will be split if this split induces a decrease of the impurity greater than or equal to this value.)
ccp_alpha = 0.0 (Complexity parameter used for Minimal Cost-Complexity Pruning.)
"""
#models.append(('DT', DecisionTreeRegressor(random_state=10), {'estimator__criterion': ["friedman_mse", "poisson", "mse", "mae"]})) 
clf = DecisionTreeRegressor(criterion='friedman_mse', random_state=10)
path = clf.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
#models.append(('DT', clf, {'estimator__ccp_alpha': ccp_alphas})) 
models.append(('DT', DecisionTreeRegressor(criterion='friedman_mse', random_state=10, ccp_alpha=0.25), {})) 
#models.append(('DT', DecisionTreeRegressor(random_state=10), {'estimator__criterion': ["friedman_mse", "poisson", "mse", "mae"], 'estimator__ccp_alpha': ccp_alphas})) #test all gridsearch possibilities
"""
n_estimators = 100 (The number of trees in the forest.)
criterion = squared_error (The function to measure the quality of a split)
splitter = best (The strategy used to choose the split at each node.)
max_depth = None (If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.)
min_samples_split = 2 (The minimum number of samples required to split an internal node.)
min_samples_leaf = 1 (The minimum number of samples required to be at a leaf node.)
min_weight_fraction_leaf = 0.0 (The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.)
max_features = None (If None, then max_features=n_features.)
random_state = 10
max_leaf_nodes = None (If None then unlimited number of leaf nodes.)
min_impurity_decrease = 0.0 (A node will be split if this split induces a decrease of the impurity greater than or equal to this value.)
bootstrap = True (Whether bootstrap samples are used when building trees.)
oob_score = False (Whether to use out-of-bag samples to estimate the generalization score.)
ccp_alpha = 0.0 (Complexity parameter used for Minimal Cost-Complexity Pruning.)
max_samples = None (Number of samples to draw from X to train each base estimator. If None, then draw X.shape[0] samples.)
gridsearch reference: https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
"""
#models.append(('RF', RandomForestRegressor(criterion='friedman_mse', oob_score=True, random_state=10, ccp_alpha=0.25), {'estimator__n_estimators': arange(30, 1000, 100)}))
"""
kernel = rbf (Specifies the kernel type to be used in the algorithm.)
degree = 3 (Degree of the polynomial kernel function.)
gamma = scale (Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma.)
coef0 = 0.0 (Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.)
tol = 1e-3 (precision of solution)
C = 1.0 (Regularization parameter.)
epsilon = 0.1 (Epsilon in the epsilon-SVR model.)
shrinking = True (Whether to use the shrinking heuristic.)
cache_size = 200 (Specify the size of the kernel cache.)
max_iter = -1 (Hard limit on iterations within solver, or -1 for no limit.)
gridsearch reference 1: https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
gridsearch reference 2: https://xiaoqizheng.github.io/machine_learning_by_R/SVM_and_SVR.html
"""
#models.append(('SVR', SVR(), {'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']}))
#models.append(('SVR', SVR(kernel='linear'), {'estimator__C': arange(0.001,1,0.1)}))
#models.append(('SVR', SVR(kernel='linear', C=0.001), {'estimator__epsilon': arange(0,1,0.1)}))
#models.append(('SVR', SVR(kernel='rbf'), {'estimator__C': arange(0.001,1,0.1)}))
#models.append(('SVR', SVR(kernel='rbf', C=0.101), {'estimator__epsilon': arange(0,1,0.1)}))
"""
n_neighbors = 5 
weights = uniform (Weight function used in prediction.)
algorithm = auto (‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.)
p = 2 (Power parameter for the Minkowski metric. Euclidean_distance (l2) for p = 2.)
metric = minkowski (distance metric to use for the tree)
gridsearch reference: https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
"""
#models.append(('KNR', KNeighborsRegressor(), {'estimator__n_neighbors': arange(1,1000,50)}))
#models.append(('KNR', KNeighborsRegressor(n_neighbors=701), {'estimator__metric': ['euclidean', 'manhattan', 'minkowski']}))
#models.append(('KNR', KNeighborsRegressor(n_neighbors=701, metric='manhattan'), {'estimator__weights': ['uniform', 'distance']}))
results = []
names = []
scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error']
scaler = StandardScaler()
#print(search.best_score_)
#print(search.best_params_)
for name, model, param in models:
    pipeline1 = Pipeline([('transformer', scaler), ('estimator', model)])
    #cv_results = cross_validate(pipeline, x_train, y_train, scoring=scoring, cv=logo, groups=groups_train, return_train_score=True)
    #cv_predict = cross_val_predict(pipeline, x_train, y_train, cv=logo, groups=groups_train)
    search = GridSearchCV(pipeline1, param, scoring=scoring, refit='neg_mean_absolute_error', cv=logo, return_train_score=True)
    search.fit(x_train, y_train, groups_train)
    cv_results = search.cv_results_
    index = search.best_index_
    pipeline2 = Pipeline([('transformer', scaler), ('estimator', search.best_estimator_)])
    cv_predict = cross_val_predict(pipeline2, x_train, y_train, cv=logo, groups=groups_train)
    """
    pyplot.plot(cv_results.get('train_neg_mean_squared_error'))
    pyplot.xlabel('Train Folds')
    pyplot.ylabel('Neg Mean Squared Error')
    pyplot.title(name)
    pyplot.show()
    pyplot.plot(cv_results.get('train_neg_root_mean_squared_error'))
    pyplot.xlabel('Train Folds')
    pyplot.ylabel('Neg Root Mean Squared Error')
    pyplot.title(name)
    pyplot.show()
    pyplot.plot(cv_results.get('train_neg_mean_absolute_error'))
    pyplot.xlabel('Train Folds')
    pyplot.ylabel('Neg Mean Absolute Error')
    pyplot.title(name)
    pyplot.show()
    pyplot.plot(cv_results.get('test_neg_mean_squared_error'))
    pyplot.xlabel('Test Folds (subjects)')
    pyplot.ylabel('Neg Mean Squared Error')
    pyplot.title(name)
    pyplot.show()
    pyplot.plot(cv_results.get('test_neg_root_mean_squared_error'))
    pyplot.xlabel('Test Folds (subjects)')
    pyplot.ylabel('Neg Root Mean Squared Error')
    pyplot.title(name)
    pyplot.show()
    pyplot.plot(cv_results.get('test_neg_mean_absolute_error'))
    pyplot.xlabel('Test Folds (subjects)')
    pyplot.ylabel('Neg Mean Absolute Error')
    pyplot.title(name)
    pyplot.show()
    """
    """
    print('%s (train): %f (%f)' % (name, cv_results.get('train_neg_mean_squared_error').mean(), cv_results.get('train_neg_mean_squared_error').std()))
    print('%s (test): %f (%f)' % (name, cv_results.get('test_neg_mean_squared_error').mean(), cv_results.get('test_neg_mean_squared_error').std()))
    print('%s (train): %f (%f)' % (name, cv_results.get('train_neg_root_mean_squared_error').mean(), cv_results.get('train_neg_root_mean_squared_error').std()))
    print('%s (test): %f (%f)' % (name, cv_results.get('test_neg_root_mean_squared_error').mean(), cv_results.get('test_neg_root_mean_squared_error').std()))
    print('%s (train): %f (%f)' % (name, cv_results.get('train_neg_mean_absolute_error').mean(), cv_results.get('train_neg_mean_absolute_error').std()))
    print('%s (test): %f (%f)' % (name, cv_results.get('test_neg_mean_absolute_error').mean(), cv_results.get('test_neg_mean_absolute_error').std()))
    pyplot.boxplot([cv_results.get('train_neg_mean_squared_error'),cv_results.get('test_neg_mean_squared_error')],labels=['train','test'])
    pyplot.ylabel('neg_mean_squared_error')
    pyplot.title(name)
    pyplot.show()
    pyplot.boxplot([cv_results.get('train_neg_root_mean_squared_error'),cv_results.get('test_neg_root_mean_squared_error')],labels=['train','test'])
    pyplot.ylabel('neg_root_mean_squared_error')
    pyplot.title(name)
    pyplot.show()
    pyplot.boxplot([cv_results.get('train_neg_mean_absolute_error'),cv_results.get('test_neg_mean_absolute_error')],labels=['train','test'])
    pyplot.ylabel('neg_mean_absolute_error')
    pyplot.title(name)
    pyplot.show()
    """
    pyplot.scatter(y_train,cv_predict)
    pyplot.xlabel('Measured FM-LE')
    pyplot.ylabel('Predicted FM-LE')
    pyplot.ylim([15,34])
    pyplot.title(name)
    pyplot.show()
    error=y_train-cv_predict
    pyplot.boxplot([error[y_train==15],
                    error[y_train==16],
                    error[y_train==17],
                    error[y_train==18],
                    error[y_train==19],
                    error[y_train==20],
                    error[y_train==21],
                    error[y_train==22],
                    error[y_train==23],
                    error[y_train==24],
                    error[y_train==25],
                    error[y_train==26],
                    error[y_train==27],
                    error[y_train==28],
                    error[y_train==29],
                    error[y_train==30],
                    error[y_train==31],
                    error[y_train==32],
                    error[y_train==33],
                    error[y_train==34]], labels=arange(15,35,1))
    pyplot.xlabel('Measured FM-LE')
    pyplot.ylabel('Measured-Predicted FM-LE')
    pyplot.title(name)
    pyplot.show()
    print('%s (train): %f (%f)' % (name, cv_results.get('mean_train_neg_mean_squared_error')[index], cv_results.get('std_train_neg_mean_squared_error')[index]))
    print('%s (test): %f (%f)' % (name, cv_results.get('mean_test_neg_mean_squared_error')[index], cv_results.get('std_test_neg_mean_squared_error')[index]))
    print('%s (train): %f (%f)' % (name, cv_results.get('mean_train_neg_root_mean_squared_error')[index], cv_results.get('std_train_neg_root_mean_squared_error')[index]))
    print('%s (test): %f (%f)' % (name, cv_results.get('mean_test_neg_root_mean_squared_error')[index], cv_results.get('std_test_neg_root_mean_squared_error')[index]))
    print('%s (train): %f (%f)' % (name, cv_results.get('mean_train_neg_mean_absolute_error')[index], cv_results.get('std_train_neg_mean_absolute_error')[index]))
    print('%s (test): %f (%f)' % (name, cv_results.get('mean_test_neg_mean_absolute_error')[index], cv_results.get('std_test_neg_mean_absolute_error')[index]))
    print(search.best_params_)
    """
    pyplot.plot(cv_results.get('param_'+list(param.keys())[0]), cv_results.get('mean_test_neg_root_mean_squared_error'),'o')
    pyplot.xlabel('param_'+list(param.keys())[0])
    pyplot.ylabel('mean_test_neg_root_mean_squared_error')
    pyplot.title(name)
    pyplot.show()
    """
    train_neg_mean_squared_error=[]
    train_neg_root_mean_squared_error=[]
    train_neg_mean_absolute_error=[]
    test_neg_mean_squared_error=[]
    test_neg_root_mean_squared_error=[]
    test_neg_mean_absolute_error=[]
    i=0
    while i < 45:
        train_neg_mean_squared_error.append(cv_results.get('split'+str(i)+'_train_neg_mean_squared_error')[index])
        train_neg_root_mean_squared_error.append(cv_results.get('split'+str(i)+'_train_neg_root_mean_squared_error')[index])
        train_neg_mean_absolute_error.append(cv_results.get('split'+str(i)+'_train_neg_mean_absolute_error')[index])
        test_neg_mean_squared_error.append(cv_results.get('split'+str(i)+'_test_neg_mean_squared_error')[index])
        test_neg_root_mean_squared_error.append(cv_results.get('split'+str(i)+'_test_neg_root_mean_squared_error')[index])
        test_neg_mean_absolute_error.append(cv_results.get('split'+str(i)+'_test_neg_mean_absolute_error')[index])
        i=i+1
        print(i)
    pyplot.boxplot([train_neg_mean_squared_error,test_neg_mean_squared_error],labels=['train','test'])
    pyplot.ylabel('neg_mean_squared_error')
    pyplot.title(name)
    pyplot.show()
    pyplot.boxplot([train_neg_root_mean_squared_error,test_neg_root_mean_squared_error],labels=['train','test'])
    pyplot.ylabel('neg_root_mean_squared_error')
    pyplot.title(name)
    pyplot.show()
    pyplot.boxplot([train_neg_mean_absolute_error,test_neg_mean_absolute_error],labels=['train','test'])
    pyplot.ylabel('neg_mean_absolute_error')
    pyplot.title(name)
    pyplot.show()
"""
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
"""