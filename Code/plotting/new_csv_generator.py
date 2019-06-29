## -*- coding: utf-8 -*-
"""
Created on 5th JUNE 2019
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
import sys
import csv
#os.chdir('..\..')
dataset  = pd.read_csv(os.path.join('data', 'transformed', 'transformed_data.csv'))

cols = dataset.columns

Y = dataset.iloc[:,-1]
#print(cols) 

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

x = dataset[['PRT_Z', 'ORIENTATION_X','ORIENTATION_Y']].values



line_2_5 = np.asarray([x[j, :] for j in range(0,len(x)) if Y[j] == 'c'])

print(np.shape(line_2_5))


prtz_lessthan0 = []
prtz_morethan0 = []


for i in range(len(line_2_5)):
    if line_2_5[i,0] < 0:
        prtz_lessthan0.append(x[i,1:])
    else:
        prtz_morethan0.append(x[i,1:])

print(np.shape(prtz_lessthan0))
print(np.shape(prtz_morethan0))


headers = ['ORIENTATION_X','ORIENTATION_Y']

headers = np.array(headers).reshape(1,len(headers))
data = np.concatenate((headers, prtz_lessthan0), axis = 0)
print(data)

#writing it in a new file
out_csv = open(os.path.join("data", "transformed", "orientation_lessthan0.csv"), "w", newline="")
with out_csv:
    writer = csv.writer(out_csv)
    writer.writerows(data)


data = np.concatenate((headers, prtz_morethan0), axis = 0)
print(data)

#writing it in a new file
out_csv = open(os.path.join("data", "transformed", "orientation_morethan0.csv"), "w", newline="")
with out_csv:
    writer = csv.writer(out_csv)
    writer.writerows(data)




