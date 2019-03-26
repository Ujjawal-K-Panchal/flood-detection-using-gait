# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:09:59 2019

@author: uchih
"""

#importing libraries.
import pandas as pd
import numpy as np
import pickle
import os

#Importing dataset.

#os.chdir(r'..\..\..\data\windowed')
dataset  = pd.read_csv(os.path.join('data', 'windowed', 'window_50_stride_25_data.csv'))# Caution uses merged window. If you wish to use the same configuration, please set random_state to 1
#os.chdir(r'..\..\Code\Models\Support Vector Machines')

#Removing some unwanted features.
dataset = dataset.drop([col for col in dataset.columns if  not col.find('MAGNETIC')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('ORIENTATION')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('ACCELEROMETER')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('GRAVITY_Y_mean')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('LINEAR_ACCELERATION_Y_median')], axis = 1)

dataset = dataset.drop([col for col in dataset.columns if  not col.find('std_dev')==-1], axis = 1)#input *std_dev for removing substring with std_dev

data1 = dataset[dataset["OUT"] == 0]
data2 = dataset[dataset["OUT"] == 0.19]

dataset = pd.concat([data1 , data2])

X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]
feature_names = list(X.columns)
feature_names

#Assigning labels for heights.
CLR = list(range(len(Y)))
for i in range(len(Y)):
    if(Y[i] == 0):
        CLR[i] = 'a'
    elif(Y[i] == 0.19):
        CLR[i] = 'b'
    elif(Y[i] == 2.5):
        CLR[i] = 'c'
    elif(Y[i] == 4.5):
        CLR[i] = 'd'

Y = CLR

#Standard Scaling the data.
from sklearn.preprocessing import StandardScaler as SS
ss = SS()
X = ss.fit_transform(X)



#train_test splitting for analysis of optimal number of parameters.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)
X_cv, X_test, Y_cv, Y_test =train_test_split(X_test, Y_test, test_size = 0.5, random_state = 0)

#Modelling

#hyper parameter tuning.
from sklearn.svm import SVC

"""
from sklearn.model_selection import GridSearchCV

Cs = [ 0.01, 0.03, 0.1, 0.3, 1, 3]
gammas = [0.01,0.03, 0.1 , 0.3, 1,3]
kernels = ['linear']
param_grid = {'C': Cs, 'gamma' : gammas, 'kernel' : kernels}
grid_search = GridSearchCV(SVC(), param_grid, cv=3, verbose = 10, n_jobs = 4)
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
"""
model = SVC(C = 0.3, gamma = 0.01, kernel = 'linear')
model.fit(X_train, Y_train)
    
#10 Cross validation

from sklearn.model_selection import KFold, cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
#previously tuned h params values : n_estimators = 36, criterion = 'entropy',

l1 = cross_val_score(model, X, Y, cv=k_fold, n_jobs=1)


print('List of Accuracies of 10-Cross-Validation :\n'+str(l1))
print('10-Cross-Validation-Accuracy mean : %.4f' %(np.sum(l1)/len(l1)) )

coeffs = model.coef_
coeff_dict = dict()
coeff_dict_mean = dict()
coeff_dict_median = dict()
sorted_for_each_plane = dict()

for i in range(len(feature_names)):
    coeff_dict[feature_names[i]] = coeffs[:,i]
    coeff_dict_mean[feature_names[i]] = np.abs(np.mean(coeffs[:,i]))
    coeff_dict_mean[feature_names[i]] = np.abs(np.median(coeffs[:,i]))
    
sorted_feature_names = sorted(coeff_dict_mean , key = coeff_dict_mean.get)

sorted_per_plane = list()

for plane in range(len(coeffs[:,0])):
    coeff_dict = {}

    for i in range(len(feature_names)): coeff_dict[feature_names[i]] = coeffs[plane,i]

    sorted_per_plane =[]

    sorted_feature_names = sorted(coeff_dict , key = coeff_dict.get)
    
    for name in sorted_feature_names:
        sorted_per_plane.append([name, coeff_dict[name]])
    sorted_per_plane.reverse()

    sorted_for_each_plane[plane] = sorted_per_plane

print(type(sorted_for_each_plane[0]))
