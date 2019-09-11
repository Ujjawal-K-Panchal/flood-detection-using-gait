# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:04:23 2019

@author: uchih
"""

#importing libraries.
import pandas as pd
import numpy as np
import pickle
import os
import sys


#Importing dataset.

os.chdir(r'..\..\..\data\windowed')
dataset  = pd.read_csv('windowed_new_data.csv')# Caution uses merged window. If you wish to use the same configuration, please set random_state to 1
os.chdir(r'..\..\Code\models\Naive Bayes Classifier')
#dataset  = pd.read_csv(os.path.join('data', 'windowed', 'window_50_stride_25_data.csv'))


#Removing some unwanted features.
dataset = dataset.drop([col for col in dataset.columns if  not col.find('MAGNETIC')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('ORIENTATION')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('ACCELEROMETER')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('PRT')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('std_dev')==-1], axis = 1)#input *std_dev for removing substring with std_dev

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


#Fitting the PCA algorithm with our Data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Getting Orthogonal Components.
pca = PCA() 
X = pca.fit_transform(X)
print(len(feature_names))
print(pca.components_.shape)

pca = PCA().fit(X)
#Plotting the Cumulative Summation of the Explained Variance to check variance preserved on no of attributes.
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.show()

#The minimum number of components to keep variance of 99% variance is 6.

#nbc modelling.
from sklearn.naive_bayes import GaussianNB
nbc = GaussianNB()


#predicting and testing.
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

#5 Cross validation
from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut

k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

l1_acc = cross_val_score(nbc, X, Y, cv=k_fold, n_jobs=1, scoring = 'accuracy')
l1_prec = cross_val_score(nbc, X, Y, cv = k_fold, n_jobs =1, scoring = 'precision_macro')
l1_rec = cross_val_score(nbc, X, Y, cv = k_fold, n_jobs = 1, scoring = 'recall_macro')




#accuracy:
print('NBC classifier : ')
print('List of Accuracies of 5-Cross-Validation :'+str(l1_acc))
print('Average of Accuracies of 5-Cross-Validation :'+str(np.average(np.array(l1_acc))))

# -- Precision Recall for each class seperately on each fold.
n = 5
i = 1
X, Y = pd.DataFrame(X), pd.DataFrame(Y)
avg_0_prec=0
avg_0_19_prec=0
avg_2_5_prec=0
avg_4_5_prec=0

avg_0_recall=0
avg_0_19_recall=0
avg_2_5_recall=0
avg_4_5_recall=0

avg_0_f1=0
avg_0_19_f1=0
avg_2_5_f1=0
avg_4_5_f1=0

for train_index, test_index in k_fold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    nbc.fit(X_train, Y_train)
    Y_pred = nbc.predict(X_test)
    print('For Fold '+str(i)+' Classwise Metrics : ([Precision], [Recall], [F1 score], [Support]):'+str(precision_recall_fscore_support(Y_test, Y_pred)))
    l1 = precision_recall_fscore_support(Y_test, Y_pred)
    avg_0_prec += l1[0][0]
    avg_0_19_prec += l1[0][1]
    avg_2_5_prec += l1[0][2]
    avg_4_5_prec += l1[0][3]
    avg_0_recall += l1[1][0]
    avg_0_19_recall += l1[1][1]
    avg_2_5_recall += l1[1][2]
    avg_4_5_recall += l1[1][3]
    avg_0_f1 += l1[2][0]
    avg_0_19_f1 += l1[2][1]
    avg_2_5_f1 += l1[2][2]
    avg_4_5_f1 += l1[2][3]
    
    
    i+=1

avg_0_prec/=5
avg_0_19_prec/=5
avg_2_5_prec/=5
avg_4_5_prec/=5
avg_0_recall/=5
avg_0_19_recall/=5
avg_2_5_recall/=5
avg_4_5_recall/=5

avg_0_f1/=5
avg_0_19_f1/=5
avg_2_5_f1/=5
avg_4_5_f1/=5


#others:
print("NBC Classifier            :       0, 0.19, 2.5, 4.5")
print("average precision for the 4 classes : ", avg_0_prec, avg_0_19_prec, avg_2_5_prec, avg_4_5_prec)
print("average recall for the 4 classes : ", avg_0_recall, avg_0_19_recall, avg_2_5_recall, avg_4_5_recall)
print("average f1 for the 4 classes : ", avg_0_f1, avg_0_19_f1, avg_2_5_f1, avg_4_5_f1)