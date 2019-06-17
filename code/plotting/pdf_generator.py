# -*- coding: utf-8 -*-
"""
Created on 6th MAY 2019
@author: Hardik Ajmani
"""
#importing libraries.
import pandas as pd
import numpy as np
import os
from scipy.stats import norm
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

dataset  = pd.read_csv(os.path.join('data', 'windowed', 'window_50_stride_25_data_new.csv'))

cols = dataset.columns

Y = dataset.iloc[:,-1]
print(cols)
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
colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73']
lower, upper = -10, 10

for i in range(len(cols)):
    if re.match(r'PRT_Z_mean.*',cols[i]) or re.match(r'GYROSCOPE_Y_mean',cols[i]):
        x = dataset.iloc[:, i]
        #x = preprocessing.normalize(np.array(x).reshape(-1, 1))
        #x = [lower + (upper - lower) * v for v in x]
        #x = np.sort(x)
        line_0 = [x[j] for j in range(0,len(x)) if Y[j] == 'a']
        line_0 = np.sort(line_0)
        line_0_19 = [x[j] for j in range(0,len(x)) if Y[j] == 'b']
        line_0_19 = np.sort(line_0_19)
        line_2_5 = [x[j] for j in range(0,len(x)) if Y[j] == 'c']
        line_2_5 = np.sort(line_2_5)
        line_4_5 = [x[j] for j in range(0,len(x)) if Y[j] == 'd']
        line_4_5 = np.sort(line_4_5)
        
        path = os.path.join("plots", "PDFs", cols[i])
        #plt.plot(line_0, norm.pdf(line_0))
        #plt.plot(line_0_19, norm.pdf(line_0_19))
        #plt.plot(line_2_5, norm.pdf(line_2_5))
        #plt.plot(line_4_5, norm.pdf(line_4_5))
        #plt.hist([line_0,line_0_19,line_2_5,line_4_5], color=colors, normed=True)
        #sns.distplot(line_0, hist = False, kde = True, kde_kws = {'linewidth': 2}, label = "0 feet", color = 'blue')
        #sns.distplot(line_0_19, hist = False, kde = True, kde_kws = {'linewidth': 2}, label = "0.19 feet", color = 'orange')
        sns.distplot(line_2_5, hist = False, kde = True, kde_kws = {'linewidth': 2}, label = "2.5 feet", color = 'green')
        #sns.distplot(line_4_5  , hist = False, kde = True, kde_kws = {'linewidth': 2}, label = "4.5 feet", color = 'red')
        plt.legend()
        plt.xlabel(cols[i])
        plt.ylabel("density")
        #plt.suptitle(cols[i])
        #plt.savefig(path + ".png")
        plt.show()
