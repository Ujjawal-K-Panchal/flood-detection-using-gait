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
dataset  = pd.read_csv(os.path.join('data', 'raw', 'raw_data_reduced.csv'))

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

x = dataset.values

line_0 = np.asarray([x[j, :] for j in range(0,len(x)) if Y[j] == 'a'])
line_0_19 = np.asarray([x[j, :] for j in range(0,len(x)) if Y[j] == 'b'])
line_2_5 = np.asarray([x[j, :] for j in range(0,len(x)) if Y[j] == 'c' ])
line_2_5 = line_2_5[:3500, :]
line_4_5 = np.asarray([x[j, :] for j in range(0,len(x)) if Y[j] == 'd'])

#print(np.shape(line_0_19))
#print(np.shape(line_2_5))

#sys.exit(0)
cols = np.array(cols).reshape(1,len(cols))
data = np.concatenate((line_0, line_0_19, line_2_5, line_4_5), axis = 0)

data = np.concatenate((cols, data), axis = 0)


out_csv = open(os.path.join("data", "raw", "raw_data1.csv"), "w", newline="")
with out_csv:
    writer = csv.writer(out_csv)
    writer.writerows(data)

print("Done")