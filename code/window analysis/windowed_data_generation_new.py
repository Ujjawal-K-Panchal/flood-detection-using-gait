## -*- coding: utf-8 -*-
"""
Created on 26th JUNE 2019
@author: Hardik Ajmani & Ujjawal Panchal.
"""
import pandas as pd
import numpy as np
import os
from scipy.fftpack import rfft
import sys
import csv

#function to calculate fft
def sum_fft(l1):
    coeffs = rfft(l1)
    sum_fft = np.sqrt( np.sum( np.power( coeffs[0:5] , 2 ) ) )
    return sum_fft

def spectral_energy(l1, n = 16):
    #n = 16 Assuming Window size of 50
    Y = rfft(l1,n)
    P = abs(Y/n)
    P = P[1 : ( (n//2)+1 )] # double // to get integer equivalent.
    energy = np.sum(np.power(P,2))
    return energy
 


dataset  = pd.read_csv(os.path.join('data', 'transformed', 'transformed_new_cleaned.csv'))

cols = dataset.columns

Y = dataset.iloc[:,-1]
x = dataset.values


classes = [0, 0.19, 2.5, 4.5]
window = 50
stride = 25



for c in classes:
    class_raw_line = np.asarray([x[j, :-1] for j in range(0,len(x)) if Y[j] == c])

    #number of rows for one column of each class
    windowed_len = int(class_raw_line.shape[0] / stride)

    for col in range(class_raw_line.shape[1]):
        windowed_column_mean   = np.asarray([np.mean  (class_raw_line[i * stride : (i * stride) + window, col]) for i in range(windowed_len)]).reshape(windowed_len, 1)        
        windowed_column_std    = np.asarray([np.std   (class_raw_line[i * stride : (i * stride) + window, col]) for i in range(windowed_len)]).reshape(windowed_len, 1)
        windowed_column_median = np.asarray([np.median(class_raw_line[i * stride : (i * stride) + window, col]) for i in range(windowed_len)]).reshape(windowed_len, 1)
        windowed_column_var    = np.asarray([np.var   (class_raw_line[i * stride : (i * stride) + window, col]) for i in range(windowed_len)]).reshape(windowed_len, 1)
        windowed_column_fft    = np.asarray([sum_fft  (class_raw_line[i * stride : (i * stride) + window, col]) for i in range(windowed_len)]).reshape(windowed_len, 1)
        windowed_column_spec   = np.asarray([spectral_energy(class_raw_line[i * stride : (i * stride) + window, col]) for i in range(windowed_len)]).reshape(windowed_len, 1)
        
        #all calculated data for one column of one class
        windowed_column_data = np.concatenate((windowed_column_mean, windowed_column_std, windowed_column_median, windowed_column_var, windowed_column_fft, windowed_column_spec), axis = 1)
        
        if col == 0: windowed_class_data = windowed_column_data
        else: windowed_class_data = np.concatenate((windowed_class_data, windowed_column_data), axis = 1)

    #creating out for each class        
    out = np.asarray([c for j in range(windowed_len)]).reshape(windowed_len, 1)
    windowed_class_data = np.concatenate((windowed_class_data, out), axis = 1)

    #adding data of each class into one final file
    if c == 0:    windowed_total_data = windowed_class_data
    else: windowed_total_data = np.concatenate((windowed_total_data, windowed_class_data), axis = 0)


#creating headers
names = list(dataset)
math_functions = ["_mean", "_std_dev", "_median", "_variance", "_fft", "_spec"]
names.remove("OUT")
new_names = []

for n in names:
    for m in math_functions:
        new_names.append(n + m)
new_names.append("OUT")
new_names = np.array(new_names).reshape(1,len(new_names))


data = np.concatenate((new_names, windowed_total_data), axis = 0)





#writing it in a new file
out_csv = open(os.path.join("data", "windowed", "windowed_new_data.csv"), "w", newline="")
with out_csv:
    writer = csv.writer(out_csv)
    writer.writerows(data)



