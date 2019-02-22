# -*- coding: utf-8 -*-
"""
Title : generating plot of windowed data single variables vs index (time).
Author : Ujjawal.K.Panchal

"""
#importing libraries.
import pandas as pd
import os

#importing dataset.

os.chdir(r'..\..\..\data\windowed')
dataset  = pd.read_csv('window_50_stride_25.csv')# Caution uses merged window. If you wish to use the same configuration, please set random_state to 1
os.chdir(r'..\..\Code\Models\Support Vector Machines')

dataset = dataset.drop([col for col in dataset.columns if  not col.find('MAGNETIC')], axis = 1)
dataset = dataset.drop([col for col in dataset.columns if  not col.find('std_dev')], axis = 1)#input *std_dev for removing substring with std_dev
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]
feature_names = list(X.columns)

os.chdir(r'..\..\Plots\Single Attribute Plots')
for feature in range(0,len(feature_names )):
    line_0 = [X[i,feature] for i in range(0,len(X)) if Y[i] == 'a']
    line_0_19 = [X[i,feature] for i in range(0,len(X)) if Y[i] == 'b']
    line_2_5 = [X[i,feature] for i in range(0,len(X)) if Y[i] == 'c']
    line_4_5 = [X[i,feature] for i in range(0,len(X)) if Y[i] == 'd']

    plt.plot([i for i in range(len(line_0))], line_0, color = 'brown', label = 'land')

    plt.plot([i for i in range(len(line_0_19))], line_0_19, color = 'magenta', label = '0.19 feet')

    plt.plot([i for i in range(len(line_2_5))], line_2_5, color = 'blue', label = '2.5 feet')

    plt.plot([i for i in range(len(line_4_5))], line_4_5, color = 'red', label = '4.5 feet')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel(feature_names[feature])
    plt.savefig(feature_names[feature]+'vsTime.png')
    plt.show()



