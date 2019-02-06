# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 09:23:24 2018

@author: Ujjawal.K.Panchal
"""

import pandas as pd
import numpy as np

'''
rotator = function for computing rotational matrix.
orientation = matrix containing azimuth, roll and pitch.
accel = matrix containing accel values in x y and z direction
tr_accel = rotationally transformed matrix
'''   #a:    yaw,pitch,roll
def rotator(alpha,beta,gamma):
    # function to compute rotational matrix.
   alpha, beta, gamma = alpha * 0.0174533, -1 * (beta * 0.0174533) , gamma * 0.0174533 
   #Accounting for the sign negation according to the android convention.
   yaw_m_Z = [
           [np.cos(alpha) , -1 * np.sin(alpha) , 0],
             [np.sin(alpha) , np.cos(alpha) , 0] , 
             [0 , 0 , 1]
             ]
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
   
   rot_matrix = np.dot(np.dot(yaw_m_Z , pitch_m_X) , roll_m_Y) # notice the order of operations. R = yaw * pitch, then R * roll 
   
   return rot_matrix

dataset = pd.read_csv("raw0.csv")

orientation = dataset.iloc[:,3:5+1].values

tr_accel = np.zeros((len(orientation),3))

for i in range(0,len(orientation)):
    rm = rotator(orientation[i,0],orientation[i,1],orientation[i,2]) #rotational matrix assignment
    accel = dataset.iloc[i,0:3].values
    accel = np.matrix(accel.reshape(3,1) , dtype = float)
    # 3 x 1 = 3 x 3 * 3 x 1
    tr_accel[i] = (np.dot(rm  ,accel)).reshape(3,)

#tr_accel[0] = X , 1 = Y, 2 = Z. 
import matplotlib.pyplot as plt

print('x:')
length = 750
plt.figure(figsize=(15,5))
plt.plot(range(length), tr_accel[:length,0], color = 'red')
plt.show()
print('y:')
plt.figure(figsize=(15,5))
plt.plot(range(length), tr_accel[:length,1] , color = 'green')
plt.show()
print('z:')
plt.figure(figsize=(15,5))
plt.plot(range(length), tr_accel[:length,2] , color = 'blue')
plt.show()
np.savetxt("accel_transformed_6cm.csv", tr_accel, delimiter = ',')

# to count no. of times a value from x , y or z exceeds a magnitude of 9.8 i.e. acceleration due to gravity.
count_x, count_y, count_z = 0,0,0
for i in range(0,len(tr_accel)):
    if( np.abs(tr_accel[i,0]) >= 9.8): 
        count_x = count_x + 1
    if(np.abs(tr_accel[i,1]) >= 9.8):
        count_y = count_y + 1
    if(np.abs(tr_accel[i,2]) >= 9.8):
        count_z = count_z + 1
