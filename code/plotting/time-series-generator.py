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

for i in range(len(cols)):
    if re.match(r'PRT_Z.*',cols[i]):
        x = dataset.iloc[:, i]
        line_2_5 = [x[j] for j in range(0,len(x)) if Y[j] == 'c']
        plt.plot(range(len(line_2_5)), line_2_5)
        plt.ylabel(cols[i])
        plt.xlabel("time")
        plt.show()