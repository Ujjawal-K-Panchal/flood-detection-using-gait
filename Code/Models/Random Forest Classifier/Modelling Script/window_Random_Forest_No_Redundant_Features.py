# -*- coding: utf-8 -*-
"""
Creation data :  Thu Sep 13 12:10:39 2018
Latest Update : Mo Feb 26 11:39:40 2019
@author: Ujjawal.K.Panchal
Note <!> Please use windowed_new_data.csv with me.
"""

import pandas as pd
import numpy as np
import csv
import pickle
import os
import sys
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


os.chdir(r'..\..\..\..\data\windowed')
dataset  = pd.read_csv('windowed_new_data.csv')# Caution uses merged window. If you wish to use the same configuration, please set random_state to 1
os.chdir(r'..\..\Code\Models\Random Forest Classifier\Modelling Script')

#Removing some unwanted features.
dataset = dataset.drop([col for col in dataset.columns if  not col.find('MAGNETIC')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('ORIENTATION')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('ACCELEROMETER')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('PRT')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('std_dev')==-1], axis = 1)#input *std_dev for removing substring with std_dev
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]

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
'''
#PCA
from sklearn.decomposition import PCA

pca = PCA(n_components = 60)
X = pca.fit_transform(X)
'''

#train_test_dev
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=1)#Don't change random state for keeping standardized.

#X_test, X_dev, Y_test, Y_dev = train_test_split(X_test, Y_test, test_size=0.5, random_state=1)
"""
#Hyperparameter tuning for Random Forest

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 30, num = 30)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 15)]
max_depth.append(None)
max_depth = [10]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
min_samples_split = [8]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]
#Criterion
criterions = ['gini', 'entropy']

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': criterions
               }
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
"""
clf_random = RandomizedSearchCV(estimator = RandomForestClassifier(n_jobs  = -1), 
                                param_distributions = random_grid, 
                                n_iter = 100, cv = 3, verbose=10, random_state=42, n_jobs = 1) 
clf_random.fit(X_dev,Y_dev)
print(clf_random.best_params_)

"""
#model
#os.chdir(r'C:\Users\uchih\Documents\RESEARCH\GITLAB\flood-detection-using-gait\Code\Models\Random Forest Classifier\Post Jan-19')
#clf = pickle.load(open('rfc-92_47_100-trees.sav', 'rb'))
#os.chdir(r'C:\Users\uchih\Documents\RESEARCH\GITLAB\flood-detection-using-gait\Code\Models\Random Forest Classifier\Modelling Script')
'''
clf = RandomForestClassifier(n_estimators= 500, min_samples_split= 2, 
                             min_samples_leaf= 1, max_features= 'sqrt', max_depth= 15, 
                             criterion= 'entropy', bootstrap= False, random_state = 0)
'''

#Modelling.
'''
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators= 500, min_samples_split= 2, 
                             min_samples_leaf= 1, max_features= 'sqrt', max_depth= 15, 
                             criterion= 'entropy', bootstrap= False, random_state = 0)
'''

clf = RandomForestClassifier(n_estimators=30, min_samples_split = 8, max_depth = 10)
'''
#Pre loaded Model.
os.chdir('../Post Jan-19')
clf = pickle.load(open('rfc-96-68_500-trees-60-Features-NO-PCA(1).sav', 'rb'))
os.chdir('../Modelling Script')
'''


    

#5 Cross validation

from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

l1_acc = cross_val_score(clf, X, Y, cv=k_fold, n_jobs=1, scoring = 'accuracy')
l1_prec = cross_val_score(clf, X, Y, cv = k_fold, n_jobs =1, scoring = 'precision_macro')
l1_rec = cross_val_score(clf, X, Y, cv = k_fold, n_jobs = 1, scoring = 'recall_macro')
print('Random Forest classifier : ')
print('List of Accuracies of 5-Cross-Validation :'+str(l1_acc))
print('List of Precision of 5-Cross-Validation :'+str(l1_prec))
print('List of Recall of 5-Cross-Validation :'+str(l1_rec))

clf.fit(X_train, Y_train)
# -- Precision Recall for each class seperately on each fold.
i = 1
X, Y = pd.DataFrame(X), pd.DataFrame(Y)
for train_index, test_index in k_fold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print('For Fold '+str(i)+' Classwise Metrics : ([Precision], [Recall], [F1 score], [Support]):'+str(precision_recall_fscore_support(Y_test, Y_pred)))
    i+=1



##edit 29/4/19 class wise precision, recall and F1 score
y_pred = clf.predict(X_test)



y_true = pd.Series(Y_test)
y_pr = pd.Series(y_pred)
print(y_pred)
print(confusion_matrix(Y_test, y_pred))
print(precision_recall_fscore_support(Y_test, y_pred))
print(pd.crosstab(y_true, y_pr, rownames=['True'], colnames=['Predicted'], margins=True))


#Other evaluation metrics.

#confusion matrix.
print(pd.crosstab(np.array(Y_test),np.array(clf.predict(X_test)) ,  margins = True))

#Precision & Recall
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

print('Accuracy of the model : ', accuracy_score(Y_test, clf.predict(X_test)))
print('Precision of the model : ', precision_score(Y_test, clf.predict(X_test), average = 'micro'))
print('Recall of the model : ',recall_score(Y_test, clf.predict(X_test), average = 'micro') )
print('F1 Score of the model : ', f1_score(Y_test, clf.predict(X_test), average = 'micro'))
'''
Model Spec : 
#List of Accuracies of 10-Cross-Validation : (With PCA)
#[0.94835681 0.89201878 0.91549296 0.95305164 0.91509434 0.91509434 0.91037736 0.91037736 0.95283019 0.90566038]
#10-Cross-Validation-Accuracy mean : 0.9218

    
#List of Accuracies of 10-Cross-Validation : (Without PCA)
#[0.92957746 0.91079812 0.92488263 0.94835681 0.93396226 0.93867925 0.9245283  0.89622642 0.93396226 0.90566038]
#10-Cross-Validation-Accuracy mean : 0.9247

'''

'''
#Writing to files for train,test,dev

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
#Load Model pre Jan19
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
#Feature Importance
clf.fit(X,Y)
feature_list = dataset.columns
feature_list = feature_list[:-1]
importances = list(clf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 9)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

for (i,(name,imp)) in enumerate(feature_importances):
    print(i+1,name,"imp :",imp)
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
