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
import sys


#Importing dataset.

os.chdir(r'..\..\..\data\windowed')
dataset  = pd.read_csv('windowed_new_data.csv')# Caution uses merged window. If you wish to use the same configuration, please set random_state to 1
os.chdir(r'..\..\Code\Models\Support Vector Machines')
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
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.8, random_state = 0)
#X_cv, X_test, Y_cv, Y_test =train_test_split(X_test, Y_test, test_size = 0.5, random_state = 0)

#Modelling

#hyper parameter tuning.
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

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
'''
model = SVC(C = 3, gamma = 0.03, kernel = 'rbf')
pickle.dump(model, open('SVM-Model-No-PCA.sav' , 'wb'))
'''

#Load pre-trained model.
'''
model = pickle.load(open('SVM-Model-No-PCA(1).sav' , 'rb'))
'''

model = SVC(C = 1, gamma = 0.01, kernel = 'linear')
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
#num_classes = 4

# from lable to categorial
#y_prediction =  y_pred.argmax(1) 
#y_categorial = np_utils.to_categorical(y_prediction, num_classes)
y_true = pd.Series(Y_test)
y_pr = pd.Series(y_pred)
print(y_pred)
print(confusion_matrix(Y_test, y_pred))
print(precision_recall_fscore_support(Y_test, y_pred))
print(pd.crosstab(y_true, y_pr, rownames=['True'], colnames=['Predicted'], margins=True))
#sys.exit(0)
#print(y_prediction)
# from categorial to lable indexing
#y_pred = y_categorial.argmax(1)
'''
#cross val accuracy.
Y_cv_pred = model.predict(X_cv)
print('Cross validation set accuracy = ',accuracy_score(Y_cv,Y_cv_pred))
#testing accuracy.
Y_test_pred = model.predict(X_test)
print('Test set accuracy = ',accuracy_score(Y_test,Y_test_pred))
print('\n')
'''
#5 Cross validation

from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

l1_acc = cross_val_score(model, X, Y, cv=k_fold, n_jobs=1, scoring = 'accuracy')
l1_prec = cross_val_score(model, X, Y, cv = k_fold, n_jobs =1, scoring = 'precision_macro')
l1_rec = cross_val_score(model, X, Y, cv = k_fold, n_jobs = 1, scoring = 'recall_macro')
print('SVM classifier : ')
print('List of Accuracies of 5-Cross-Validation :'+str(l1_acc))
print('Average of Accuracies of 5-Cross-Validation :'+str(np.average(np.array(l1_acc))))

#print('List of Precision of 5-Cross-Validation :'+str(l1_prec))
#print('List of Recall of 5-Cross-Validation :'+str(l1_rec))

#Other evaluation metrics.

#confusion matrix.
print(pd.crosstab(np.array(Y_test),np.array(model.predict(X_test)) ,  margins = True))

#Precision & Recall
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

print('Accuracy of the model : ', accuracy_score(Y_test, model.predict(X_test)))
print('Precision of the model : ', precision_score(Y_test, model.predict(X_test), average = 'micro'))
print('Recall of the model : ',recall_score(Y_test, model.predict(X_test), average = 'micro') )
print('F1 Score of the model : ', f1_score(Y_test, model.predict(X_test), average = 'micro'))

# -- Precision Recall for each class seperately on each fold.
i = 1
X, Y = pd.DataFrame(X), pd.DataFrame(Y)
for train_index, test_index in k_fold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print('For Fold '+str(i)+' Classwise Metrics : ([Precision], [Recall], [F1 score], [Support]):'+str(precision_recall_fscore_support(Y_test, Y_pred)))
    i+=1


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
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
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


print("SVM Classifier            :       0, 0.19, 2.5, 4.5")
print("average precision for the 4 classes : ", avg_0_prec, avg_0_19_prec, avg_2_5_prec, avg_4_5_prec)
print("average recall for the 4 classes : ", avg_0_recall, avg_0_19_recall, avg_2_5_recall, avg_4_5_recall)
print("average f1 for the 4 classes : ", avg_0_f1, avg_0_19_f1, avg_2_5_f1, avg_4_5_f1)




#finding feature weights and sorting by mean and median.

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

sorted_by_mean = list()
for name in sorted_feature_names:
    sorted_by_mean.append([name, coeff_dict_mean[name]])
sorted_by_mean.reverse()

sorted_by_median = list()
sorted_feature_names = sorted(coeff_dict_median , key = coeff_dict_median.get)
for name in sorted_feature_names:
    sorted_by_median.append([name, coeff_dict_median[name]])
sorted_by_median.reverse()

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

#Top 10 important features.
for i in range(0,len(sorted_for_each_plane)):
    if(i == 0):
        print("0ft vs 0.19 ft")
    elif(i == 1):
        print("0ft vs 2.5 ft")
    elif(i==2):
        print("0ft vs 4.5 ft")
    elif(i==3):
        print("0.19ft vs 2.5 ft")
    elif(i ==4):
        print("0.19ft vs 4.5 ft")
    elif(i==5):
        print("2.5ft vs 4.5 ft")
    for j in range(0,10):
                print("\t",j ,":", sorted_for_each_plane[i][j][0], sorted_for_each_plane[i][j][1])
    

