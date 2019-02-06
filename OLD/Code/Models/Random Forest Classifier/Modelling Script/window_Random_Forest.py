# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 12:10:39 2018

@author: Ujjawal.K.Panchal
Note! Please use Merged_Window.csv with me.
"""

import pandas as pd
import numpy as np
import csv
import pickle
dataset  = pd.read_csv('merged_windowed.csv')# Caution uses merged window. If you wish to use the same configuration, please set random_state to 1
dataset = dataset.drop([col for col in dataset.columns if  not col.find('MAGNETIC')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('std_dev')], axis = 1)#input *std_dev for removing substring with std_dev
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]

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
'''
train_test_dev
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)#Don't change random state for keeping standardized.

X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
'''
#10 Cross validation
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
k_fold = KFold(len(Y), n_folds=10, shuffle=True, random_state=0)
#previously tuned h params values : n_estimators = 36, criterion = 'entropy',
clf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',verbose = 1, n_jobs = -1)
l1 = cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1)
print(l1)
print(np.sum(l1)/len(l1))
'''
Writing to files

X_train.to_csv('X_train.csv')
Y_train.to_csv('Y_train.csv')
X_test.to_csv('X_test.csv')
Y_test.to_csv('Y_test.csv')
X_dev.to_csv('X_dev.csv')
Y_dev.to_csv('Y_dev.csv')
'''
'''
#Modelling
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators = 36, criterion = 'entropy', verbose = 1)

clf.fit(X_train, Y_train)

pickle.dump(clf , open('rfc98_73-10crossval.sav', 'wb'))
'''
'''
#Load Model
clf = pickle.load(open('rfc98_31dev.sav', 'rb'))
#Dev Prediction
Y_pred= clf.predict(X_dev)
'''
'''
#Dev Evaluation
print('\nCross Validation set:\n')
from sklearn.metrics import  precision_score, accuracy_score, recall_score
print('Confusion Matrix')
print(pd.crosstab(np.array(Y_dev),np.array(Y_pred), margins = False))
print('Accuracy : ',accuracy_score(Y_dev,Y_pred)*100, ' %' )
print('Precision : ', precision_score(Y_dev,Y_pred, average = 'macro'))
print('Recall : ', recall_score(Y_dev, Y_pred, average = 'macro'))
'''
'''
#from https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd
#Feature Importance
feature_list = dataset.columns
feature_list = feature_list[:-1]
importances = list(clf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 9)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
'''
'''
#Test Prediction
Y_pred= clf.predict(X_test)

# Test Evaluation 
print('\nTest set:\n')
print('Confusion Matrix')
print(pd.crosstab(np.array(Y_test),np.array(Y_pred), margins = False))
print('Accuracy : ',accuracy_score(Y_test,Y_pred)*100, ' %' )
print('Precision : ', precision_score(Y_test,Y_pred, average = 'macro'))
print('Recall : ', recall_score(Y_test, Y_pred, average = 'macro'))
'''
