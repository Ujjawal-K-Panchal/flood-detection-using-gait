#%%
# # -*- coding: utf-8 -*-
"""
    code for :- preprocessing on gait analysis data ( merged steps )
    created by - Ujjawal k Panchal and Hardik Ajmani
    project :- Nadai 
"""

#this file does a basic pre-processing and creates a new ouput file in the end
#just change the location of file in read_csv() after all the function definations
#UPDATE : 30.08.2018 by Ujjawal, down vec removed, rotator changed for partial rotational transformation.
#%%
#importing required libs
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import re

#%%
def cleaning_dataset(dataset):
    #this function will remove all the unwanted features from the dataset
    # accelerometer, gravity, Liner_aceeleration, magnetometer, Orientation X,Y and Z
    Y = dataset.iloc[:,-1].values
    X1 = dataset.iloc[:,:12].values
    X2 = dataset.iloc[:,13:19].values
    X = np.concatenate((X1, X2), axis = 1)
    return (X,Y)


#%% 
def making_headers(names):
    #to remove all the units
    unit = re.compile("\s\(.*\)")

    #creating a final list
    out = list()
    for i,n in zip(range(19), names):
        n = re.sub(unit, '', n)
        #subsitute spaces with _
        n = re.sub(r'\s', '_', n)
        if n != "LIGHT":
            out.append(n)
    out.append("PRT_X")
    out.append("PRT_Y")
    out.append("PRT_Z")
    out.append("OUT")
    return out 
#%%
'''
rotator = function for computing rotational matrix.
orientation = matrix containing azimuth, roll and pitch.
accel = matrix containing accel values in x y and z direction
tr_accel = rotationally transformed matrix
'''   
#a:    yaw,pitch,roll
def rotator(alpha,beta,gamma):
    # function to compute rotational matrix.
   alpha, beta, gamma = alpha * 0.0174533, -1 * (beta * 0.0174533) , gamma * 0.0174533 
   #Accounting for the sign negation according to the android convention.
   yaw_m_Z = [
           [np.cos(alpha) , -1 * np.sin(alpha) , 0],
             [np.sin(alpha) , np.cos(alpha) , 0] , 
             [0 , 0 , 1]
             ] # Not using Yaw for rot transform. 
   yaw_m_Z = np.matrix(yaw_m_Z)
   
   pitch_m_X =[ 
                [1 , 0 , 0],
                [0, np.cos(beta) , -1*np.sin(beta)],
                [0,np.sin(beta) , np.cos(beta)]
               ]
   
   pitch_m_X = np.matrix(pitch_m_X)
   
   roll_m_Y = [
               [np.cos(gamma) , 0 , np.sin(gamma)],
               [0 , 1 , 0],
               [-1*np.sin(gamma) , 0 , np.cos(gamma)]
              ]
   roll_m_Y = np.matrix(roll_m_Y)
   
   rot_matrix = np.dot(pitch_m_X , roll_m_Y) # notice not using the yaw matrix! the order of operations. R = yaw * pitch, then R * roll 
                                                #'''notice not using yaw for partial transformation!'''   
   return rot_matrix


#%%
def rotation_transform(dataset):   

    
    orientation = dataset.iloc[:,16:19].values

    tr_accel = np.zeros((len(orientation),3))

    for i in range(0,len(orientation)):
        rm = rotator(orientation[i,0],orientation[i,1],orientation[i,2]) #rotational matrix assignment
        accel = dataset.iloc[i,0:3].values
        accel = np.matrix(accel.reshape(3,1) , dtype = float)
        # 3 x 1 = 3 x 3 * 3 x 1
        tr_accel[i] = (np.dot(rm  ,accel)).reshape(3,)

    return tr_accel


#%%
df = pd.read_csv("C:\\Honey\\projects\\Research gait\\Flood Detection\\data\\Rec interval\\Rec interval _ 0.1 data\\RAW\\4 and half feet raw\\4_5_feet.csv")
#reading the csv file, enter the path here
#%%
#cleaning the dataset by removing uncessary columns
X, Y = cleaning_dataset(df)

#making sure that shape isn't (len,)
Y = np.array(Y).reshape(len(Y),1)

#%%
#calculating the rotatiom trasformed vector
rot = rotation_transform(df)
rot = np.array(rot).reshape(len(rot),3)

#%%
#concatinating all the columns, processed data and outputs
X = np.concatenate((X, rot, Y), axis = 1)

#%%
#cleaning headers to write it to a new file
headers = making_headers(list(df))
headers = np.array(headers).reshape(1,len(headers))


#%%
data = np.concatenate((headers, X), axis = 0)
print(data)


#%%
#writing it in a new file
out_csv = open("C:\\Honey\\projects\\Research gait\\Flood Detection\\data\\Rec interval\\Rec interval _ 0.1 data\\Transformed\\4_5_processed_new_interval.csv", "w", newline="")
with out_csv:
    writer = csv.writer(out_csv)
    writer.writerows(data)