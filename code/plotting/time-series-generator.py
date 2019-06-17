## -*- coding: utf-8 -*-
"""
Created on 6th MAY 2019
@author: Hardik Ajmani & Ujjawal Panchal.
"""
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
        prtz_lessthan0.append(line_2_5[i,:])
    else:
        prtz_morethan0.append(line_2_5[i,:])

prtz_lessthan0 = np.asarray(prtz_lessthan0)
prtz_morethan0 = np.asarray(prtz_morethan0)
print(prtz_lessthan0)

#sys.exit()
print(np.shape(prtz_morethan0))
 

cols = [ 'PRT_Z', 'ORIENTATION_X','ORIENTATION_Y']
 
for i in range(1):
    #Total Plot.
    
    plt.plot(range(3480, 3510), line_2_5[3480:3510, i], color = 'red', label = 'full')
    #plt.plot(range(len(prtz_lessthan0[:,i])), prtz_lessthan0[:, i], color = 'red', label = 'less_than 0')        
    #plt.plot(range(len(prtz_lessthan0[:,i]), len(prtz_lessthan0[:,i]) + len(prtz_morethan0[:,i])), prtz_morethan0[:, i], color = 'blue', label = 'more_than 0')
    #plt.plot(range(len(prtz_morethan0[:,i])), prtz_morethan0[:, i], color = 'blue', label = 'more_than 0')
    
    plt.ylabel(cols[i])
    plt.xlabel("row number")
    plt.legend(loc= 'upper right')
    #path = os.path.join("plots", "time series", cols[i] + "_for_all" )
    #plt.savefig(path + ".png")
    plt.show()



sys.exit()  
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