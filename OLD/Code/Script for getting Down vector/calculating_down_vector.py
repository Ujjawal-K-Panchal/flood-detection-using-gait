import pandas as pd
import numpy as np
import csv

df = pd.read_csv("DOWNALL.csv")
#reading csv file

a = df[list(['ACCELEROMETER.X','ACCELEROMETER.Y','ACCELEROMETER.Z'])].values
v = df[list(['GRAVITY.X','GRAVITY.Y','GRAVITY.Z'])].values
#reading accelerometer and gravity readings from dataframe

A = np.array(a).reshape(len(a), 3)
V = np.array(v).reshape(len(v), 3)
#converting them to numpy arrays for help in mathematical calculations
'''

Now the formula for calculating the net acceleration in down direction irrespective of orientation is
d = a - v
p = ((d.v)/(v.v))v  

here '.' period represents vector dot product

'''
D = A - V

num = []
den = []

'''
this loop is to calculate numerator and denomiantor seperately
numpy dot product is matrix dot product which is different from vector dot product

this loop will give us one value for each row, that will be in the end multiplied
by the V vector to get a vector in down direction
'''

for i in range(len(a)):
    num.append(np.dot(D[i,:],np.transpose(V[i,:])))
    den.append(np.dot(V[i,:],np.transpose(V[i,:])))

NUM = np.array(num).reshape(len(a),1)
DEN = np.array(den).reshape(len(a),1)

P = (NUM / DEN) #* 
'''V Multiplied to V is commented out in order to get a single row of values. i.e. a time plo'''

H = D - P

''' 
H is similar to P but in horizontal direction
'''

file = open("down0.csv", "w", newline="")
with file:
    writer = csv.writer(file)
    writer.writerows([["DOWN",]])
    writer.writerows(P.tolist())
   # writer.writerows([["HORIZ"]])
   # writer.writerows(H.tolist())

'''a = list([[1,2,3],[2,3,4]])
b = list([[1,2,3],[2,3,4]])

print(a + b)'''

dataset = pd.read_csv('down0.csv')
import matplotlib.pyplot as plt
plt.plot(dataset.iloc[:1000,0].values, dataset.iloc[:1000,1].values)
plt.xlabel('Row')
plt.ylabel('Acceleration in Downward Direction')
plt.title('For Height0 (land) at 0.1s per reading')