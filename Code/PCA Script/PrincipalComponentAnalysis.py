# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 08:59:21 2018

@author: Ujjawal.K.Panchal
Title : PCA on transformed data.
"""
import pandas as pd
import numpy as np
dataset = pd.read_csv('accel_transformed_data.csv') # file on which PCA is to be performed.

from sklearn.preprocessing import StandardScaler
dataset = StandardScaler().fit_transform(dataset)
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
X = None
for i in range(0,len(dataset),10):
   # print(i,' : ',i+20)
    principal = pca.fit_transform(dataset[i:i+20,:])
    if (i ==0):
        X = principal
    else:
        X = np.vstack([X,principal])
    
done_dataset = pd.DataFrame(data = X , columns = ['AccelPCA1' , 'AccelPCA2' , 'AccelPCA3' ])

np.savetxt('PCAd-raw0.csv' , done_dataset, delimiter = ',')

