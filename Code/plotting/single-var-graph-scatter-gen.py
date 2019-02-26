# -*- coding: utf-8 -*-
"""
Title : generating plot of windowed data single variables vs index (time).
Author : Ujjawal.K.Panchal

"""
#importing libraries.
import pandas as pd
import os
import matplotlib.pyplot as plt
#importing dataset.

os.chdir(r'..\..\data\windowed')
dataset  = pd.read_csv('window_50_stride_25_data.csv')# Caution uses merged window. If you wish to use the same configuration, please set random_state to 1

dataset = dataset.drop([col for col in dataset.columns if  not col.find('MAGNETIC')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('std_dev')], axis = 1)#input *std_dev for removing substring with std_dev
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]
feature_names = list(X.columns)

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
X = X.iloc[:,:].values
#plotting.
os.chdir(r'..\..\Plots\one feature scatter plots')
for feature in range(0,len(feature_names )):
    line_0 = [X[i,feature] for i in range(0,len(X)) if Y[i] == 'a']
    line_0_19 = [X[i,feature] for i in range(0,len(X)) if Y[i] == 'b']
    line_2_5 = [X[i,feature] for i in range(0,len(X)) if Y[i] == 'c']
    line_4_5 = [X[i,feature] for i in range(0,len(X)) if Y[i] == 'd']
    plt.scatter([i for i in range(100)], line_0[:100], color = 'yellow', label = 'land')
    plt.scatter([i for i in range(100)], line_0_19[:100], color = 'magenta', label = '0.19 feet')
    plt.scatter([i for i in range(100)], line_2_5[:100], color = 'blue', label = '2.5 feet')
    plt.scatter([i for i in range(100)], line_4_5[:100], color = 'red', label = '4.5 feet')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel(feature_names[feature])
    plt.savefig(feature_names[feature]+'vsTime.png')
    plt.show()



