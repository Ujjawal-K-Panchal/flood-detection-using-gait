# -*- coding: utf-8 -*-
"""
Title : generating plot of windowed data single variables vs index (time).
Author : Ujjawal.K.Panchal & Hardik Ajmani

"""
#importing libraries.
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import string
import os
import re
#importing dataset.

rcParams.update({'figure.autolayout': True})

def format_label(a):
    a = re.sub(r'\_', r' ', a)

    a = re.sub("LINEAR ACCELERATION Y", "$a_y$" ,a)

    a = re.sub("LINEAR ACCELERATION Z", "$a_z$" ,a)

    a = re.sub("GRAVITY Y", "$g_y$" ,a)

    a = re.sub("spec", "Spectral Energy", a)
    
    a = re.sub("fft", " Fast Fourier Transform", a)

    return a

#os.chdir(r'..\..\data\windowed')
#dataset  = pd.read_csv('window_50_stride_25_data.csv')# Caution uses merged window. If you wish to use the same configuration, please set random_state to 1
dataset  = pd.read_csv(os.path.join('data', 'transformed', 'transformed_new_cleaned.csv'))

cols = ['LINEAR_ACCELERATION_Y','LINEAR_ACCELERATION_Z', 'GRAVITY_Y']
Y = dataset.iloc[:,-1]

#dataset = dataset.drop([col for col in dataset.columns if  not col.find('MAGNETIC')], axis = 1)
#dataset = dataset.drop([col for col in dataset.columns if  not col.find('std_dev')], axis = 1)#input *std_dev for removing substring with std_dev
#X = dataset.iloc[:,:-1]

#feature_names = list(X.columns)

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
#X = X.iloc[:,:].values
#plotting.
#os.chdir(r'..\..\Plots\one feature scatter plots')
for c in cols:
    x = dataset[c].values
    plt.figure(figsize=(16,14.5))
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    line_0    = [x[i] for i in range(0,len(x)) if Y[i] == 'a']
    line_0_19 = [x[i] for i in range(0,len(x)) if Y[i] == 'b']
    line_2_5  = [x[i] for i in range(0,len(x)) if Y[i] == 'c']
    line_4_5  = [x[i] for i in range(0,len(x)) if Y[i] == 'd']
    plt.plot([i for i in range(200)],    line_0[100:300], color = 'red', label = '0')
    plt.plot([i for i in range(200)], line_0_19[100:300], color = 'blue', label = '0.19 feet')
    plt.plot([i for i in range(200)],  line_2_5[100:300], color = 'green', label = '2.5 feet')
    plt.plot([i for i in range(200)],  line_4_5[100:300], color = 'orange', label = '4.5 feet')
    plt.legend(prop={'size': 50})
    plt.xlabel('Reading Number', fontsize = 50)
    plt.ylabel(format_label(c), fontsize = 50)
    plot_path = os.path.join('plots', 'one')
    plt.savefig(os.path.join(plot_path, c + '.eps'), format='eps')
    plt.savefig(os.path.join(plot_path, c + '.png'))
    #plt.show()



