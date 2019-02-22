# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:28:45 2019

@author: Ujjawal.K.Panchal
"""
#importing libraries.
import pandas as pd
import numpy as np
import pickle
import os

#Importing dataset.

os.chdir(r'..\..\..\data\windowed')
dataset  = pd.read_csv('window_50_stride_25.csv')# Caution uses merged window. If you wish to use the same configuration, please set random_state to 1
os.chdir(r'..\..\Code\Models\Support Vector Machines')

dataset = dataset.drop([col for col in dataset.columns if  not col.find('MAGNETIC')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('std_dev')], axis = 1)#input *std_dev for removing substring with std_dev
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

'''
#Fitting the PCA algorithm with our Data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Getting Orthogonal Components.
pca = PCA(n_components = 60) 
X = pca.fit_transform(X)
print(len(feature_names))
print(pca.components_.shape)
'''




"""
pca = PCA().fit(X)
#Plotting the Cumulative Summation of the Explained Variance to check variance preserved on no of attributes.
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.show()
#The minimum number of components to keep variance of 99% variance is 46.

"""
'''
importance_dict = dict()
components = pca.components_.T # Now each row contains different features, and each column, their information in transformed components. 
for i in range(len(feature_names)):
    importance_dict[ feature_names[i] ] = components[i]
'''
#writing importance of features in a file.
#df = pd.DataFrame(pca.components_,columns = feature_names)
#df.to_csv('PCA_feature_variances_46-components.csv')
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
kernels = ['rbf','poly', 'linear']
param_grid = {'C': Cs, 'gamma' : gammas, 'kernel' : kernels}
grid_search = GridSearchCV(SVC(), param_grid, cv=3, verbose = 10, n_jobs = 4)
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
"""
#modelling.
#model PARAMETERS = SVC(C = 3, gamma = 0.1, kernel = 'rbf') #will be chosen from pickle. 
# With PCA : {'C': 3, 'gamma': 0.03, 'kernel': 'rbf'}
#from sklearn.metrics import accuracy_score # we will use accuracy score for scoring.

"""
#evaluating performance for selection of feature selection calculation.
from sklearn.feature_selection import SelectKBest, f_classif

l1 = list()

for i in range(1,X.shape[1]+1):
    #print(i)
    sel = SelectKBest(score_func = f_classif,k=i)
    X_1 = sel.fit_transform(X_train,Y_train)
    X_2 = sel.transform(X_cv)
    model.fit(X_1,Y_train)
    Y_pred = model.predict(X_2)
    acc = accuracy_score(Y_cv,Y_pred)
    #print(acc)
    l1.append(acc)
    
#Checking the number of best attributes to take for the best accuracy.
max_i,max_x = 0,0
for i,x in enumerate(l1):
    if(x > max_x):
        max_x = x
        max_i = i
print('The best accuracy is',max_x,' which is received on selecting',max_i,'attributes')

#This is experimentally found to be 28. So we modify all our splits to contain only 28 of the best attribs.
#X_train = sel.fit_transform(X_train,Y_train)
#X_cv = sel.transform(X_cv)
#X_test = sel.transform(X_test)
#X = sel.transform(X)
"""
#Now, we train and test our model on the training, cross validation and test sets.
#model = pickle.load(open('SVC_92_04_46-attribs-PCAd.sav','rb'))
#pickle.dump(model, open('SVC_92_04_46-attribs-PCAd.sav','wb'))
model = SVC(C = 3, gamma = 0.03, kernel = 'rbf')
'''
#cross val accuracy.
Y_cv_pred = model.predict(X_cv)
print('Cross validation set accuracy = ',accuracy_score(Y_cv,Y_cv_pred))
#testing accuracy.
Y_test_pred = model.predict(X_test)
print('Test set accuracy = ',accuracy_score(Y_test,Y_test_pred))
print('\n')
'''
#10 Cross validation

from sklearn.model_selection import KFold, cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
#previously tuned h params values : n_estimators = 36, criterion = 'entropy',

l1 = cross_val_score(model, X, Y, cv=k_fold, n_jobs=1)


print('List of Accuracies of 10-Cross-Validation :\n'+str(l1))
print('10-Cross-Validation-Accuracy mean : %.4f' %(np.sum(l1)/len(l1)) )



    
