# -*- coding: utf-8 -*-
"""
Created on 6th MAY 2019
@author: Hardik Ajmani & Ujjawal Panchal.
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
os.chdir('..\..')
dataset  = pd.read_csv(os.path.join('data', 'transformed', 'transformed_data.csv'))

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
x = dataset['PRT_Z'].values
x.reshape(-1,1)
line_2_5 = [x[j] for j in range(0,len(x)) if Y[j] == 'c']

#Total Plot.
plt.plot(range(len(line_2_5)), line_2_5)
plt.ylabel(cols[i])
plt.xlabel("time")
plt.show()


#Plots for Volunteers.

#Volunteer 1.
plt.plot(range(len(line_2_5[0:50])), line_2_5[0:50], color = 'red', label = 'v1w1')
#plt.plot(range(len(line_2_5[50:100])), line_2_5[50:100], color = 'green',label='v1w2')
plt.plot(range(len(line_2_5[100:150])), line_2_5[100:150], color = 'blue', label='v1w3')
plt.ylabel('PRT_Z')
plt.xlabel("time")
plt.legend(loc= 'upper right')
plt.show()


#Volunteer 2.
plt.plot(range(len(line_2_5[10000:10050])), line_2_5[10000:10050], color = 'red', label = 'v2w1')
plt.plot(range(len(line_2_5[10100:10150])), line_2_5[10100:10150], color = 'blue', label='v2w3')
plt.ylabel('PRT_Z')
plt.xlabel("time")
plt.legend(loc= 'upper right')
plt.show()




"""
for i in range(len(cols)):
    if re.match(r'PRT_Z.*',cols[i]):
        x = dataset.iloc[:, i].values
        x.reshape(-1,1)
        line_2_5 = [x[j] for j in range(0,len(x)) if Y[j] == 'c']
        plt.plot(range(len(line_2_5)), line_2_5)
        plt.ylabel(cols[i])
        plt.xlabel("time")
        plt.show()
"""