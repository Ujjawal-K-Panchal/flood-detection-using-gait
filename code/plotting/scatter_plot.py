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
import random

dataset  = pd.read_csv(os.path.join('data', 'windowed', 'windowed_new_data.csv'))

#cols = ['GYROSCOPE_Y_spec','GYROSCOPE_Y_var']
cols = ['PRT_Y_variance', 'LINEAR_ACCELERATION_Z_variance']

Y = dataset.iloc[:,-1]

x = dataset[['PRT_Y_variance', 'LINEAR_ACCELERATION_Z_variance']].values

line_0    = np.asarray([x[j, :] for j in range(0,len(x)) if Y[j] == 0])
line_0_19 = np.asarray([x[j, :] for j in range(0,len(x)) if Y[j] == 0.19])
line_2_5  = np.asarray([x[j, :] for j in range(0,len(x)) if Y[j] == 2.5])
line_4_5  = np.asarray([x[j, :] for j in range(0,len(x)) if Y[j] == 4.5])

color = {0 : 'red', 0.19 : 'blue', 2.5 : 'green', 4.5 : 'orange'}

#random_points = random.sample(range(1,140), 100)

#plt.scatter(line_2_5[random_points, 0], line_2_5[random_points, 1], color = color[2.5], label = '2.5 ft')
plt.scatter(line_0_19[:, 0], line_0_19[:, 1], color = color[0.19], label = '0.19 ft')
#plt.scatter(line_0_19[random_points, 0], line_0_19[random_points, 1], color = color[0.19], label = '0.19 ft')
plt.scatter(line_4_5[:, 0], line_4_5[:, 1], color = color[4.5], label = '4.5 ft')
plt.legend()
plt.xlabel(cols[0])
plt.ylabel(cols[1])
plt.show()