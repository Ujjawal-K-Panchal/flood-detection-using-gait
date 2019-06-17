'''
    Version - 2.0 (added stride functionality)
    Objective - csv to csv WINDOW analysis
    created by - Hardik Ajmani
    project - Nadai

'''

#%%
import pandas as pd
import numpy as np
import os
from scipy.fftpack import rfft
import csv
import sys
#%%
df = pd.read_csv(os.path.join("data", "transformed", "transformed_new_cleaned.csv"))
#print(list(df))

#%%
#returning X and y
def return_X_Y(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return X,y
#print(return_X_Y(df))

#%%
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
 

#%%
#rolling fucntion for fft
def rolling_fft(data, window):
    size = data.shape
    result = np.full(size, np.nan)
    for r in range(size[1]):  
        for i in range(0, size[0] - window + 1):
            result[i + window - 1, r] = sum_fft(data.iloc[i : i + window, r])
    return result 
#print(rolling_fft(df.iloc[:, 1:5], 10))

#%%
#rolling fucntion for fft
def rolling(data, window, stride, operation = 'sum'):
    size = data.shape
    #print(np.mean(data.iloc[20:30,3].values))
    result = np.full(size, np.nan)
    i = 0
    for r in range(size[1]):
        i = 0  
        while i <= size[0] - window:           
            if operation == 'mean':
                #print(str((data.iloc[i : i + window, r].values)) + " : " + str(r))
                result[i + window - 1, r] = np.mean(data.iloc[i : i + window, r].values)
            elif operation == 'std':
                result[i + window - 1, r] = np.std(data.iloc[i : i + window, r].values)
            elif operation == 'median':
                result[i + window - 1, r] = np.median(data.iloc[i : i + window, r].values)
            elif operation == 'variance':
                result[i + window - 1, r] = np.var(data.iloc[i : i + window, r].values)
            elif operation == 'fft':
                result[i + window - 1, r] = sum_fft(data.iloc[i : i + window, r].values)
            elif operation == 'spec':
                result[i + window - 1, r] = spectral_energy(data.iloc[i : i + window, r].values)

            else:
                print("Please check the operation!")
            
            i += stride
    return result 
#print(rolling(df.iloc[:, 1:5], 10, 1, 'mean')[50:60,0:3])


#%%
#defining window function,
#choice is upto the user to use either individual functions or this function
def window_func(data, window):
    #here windowed data is being calculated with MEAN function and then coverted to numpy array
    #copy the last concatennate and add new functions IF required
    #mean, std deviation, median and variance calculated as of now
    final_data = rolling(data, 50, 25, 'mean')
    print(final_data.shape)
    final_data = np.concatenate((final_data,rolling(data, 50, 25, 'median')), axis = 1)
    final_data = np.concatenate((final_data,rolling(data, 50, 25, 'std')), axis = 1)
    final_data = np.concatenate((final_data,rolling(data, 50, 25, 'variance')), axis = 1)
    final_data = np.concatenate((final_data,rolling(data, 50, 25, 'fft')), axis = 1)
    final_data = np.concatenate((final_data,rolling(data, 50, 25, 'spec')), axis = 1)
    #print(rolling_fft(data, window).shape)
    return final_data

#%%
#slicing the data into classes and then using window_func to do sliding window analysis
def slice_by_classes(X,y, window = 10):
    #find all the unique classes
    classes = set(y)
    print("Length of dataset is - :" +  str(len(y)))

    print(classes)
    #sys.exit(0)

    sliced_data = np.array([])
    windowed_data = np.array([])
    i = 0

    #iterate to slice the data according to each class
    for c in classes:
        #finding all the location where classes occur
        loc = np.argwhere(y == c)

        #taking the last and first value of all the locations for range
        print("Class " + str(c) + " - " + str(loc[0,0]) + ":" + str(loc[-1,0]))

        #calling window_func to calculate for each class
        windowed_data = window_func(X.iloc[loc[0,0] : loc[-1,0] + 1, :], window)

        #this is a flag, as empty np array can't be reshaped :P
        if i == 0:
            sliced_data = windowed_data
            i = 1
        else:
            sliced_data =  np.concatenate((sliced_data,windowed_data), axis = 0)
    return sliced_data

#print(np.shape(slice_by_classes(X,y)))


#%%
#this will create headers for the csv file, 20 *4 = 80 headers
def creating_headers(names):
    math_functions = ["_mean", "_std_dev", "_median", "_variance", "_fft", "_spec"]
    names.remove("OUT")
    new_names = []
    for m in math_functions:
        for n in names:
            new_names.append(n + m)
    new_names.append("OUT")
    return new_names
#print((creating_headers(list(df))))

#now calls to the functions are made and concatenations are done

#%%
X,y = return_X_Y(df)

#%%
X = slice_by_classes(X,y)
print(X)

#%%
y = np.array(y).reshape(len(y),1)


#%%
headers = creating_headers(list(df))
headers = np.array(headers).reshape(1,len(headers))

#%%
#print(np.shape(X))
X = np.concatenate((X,y), axis = 1)
X = X[~np.isnan(X).any(axis=1)]
#print(np.shape(X))
#%%
data = np.concatenate((headers, X), axis = 0)
print(data)


#%%
#writing it in a new file
out_csv = open(os.path.join("data", "windowed", "window_50_stride_25_data_new.csv"), "w", newline="")
with out_csv:
    writer = csv.writer(out_csv)
    writer.writerows(data)


