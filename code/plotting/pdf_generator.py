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


dataset  = pd.read_csv(os.path.join('data', 'windowed', 'window_50_stride_25_data.csv'))

cols = dataset.columns

lower, upper = -7, 7

#print(cols)

for i in range(len(cols)):
    if re.match(r'PRT.*',cols[i]):
        x = dataset.iloc[:, i]
        x = np.sort(x)
        path = os.path.join("plots", "PDFs", cols[i])
        plt.plot(x, norm.pdf(x))
        plt.suptitle("PDF of " + cols[i])
        plt.savefig(path + ".png")
        plt.show()
