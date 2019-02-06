'''
    Objective - csv to csv WINDOW analysis
    created by - Hardik Ajmani
    project - Nadai

'''

#%%
import pandas as pd
import numpy as np
from scipy.fftpack import rfft
import csv

#%%
df = pd.read_csv("C:\Honey\projects\Research gait\Flood Detection\data\Rec interval\Rec interval _ 0.1 data\Transformed\\merged_processed_new_interval.csv")
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
#defining window function,
#choice is upto the user to use either individual functions or this function
def window_func(data, window):
    #here windowed data is being calculated with MEAN function and then coverted to numpy array
    #copy the last concatennate and add new functions IF required
    #mean, std deviation, median and variance calculated as of now
    final_data = data.rolling(window).mean().values
    print(final_data.shape)
    final_data = np.concatenate((final_data,data.rolling(window).std()), axis = 1)
    final_data = np.concatenate((final_data,data.rolling(window).median()), axis = 1)
    final_data = np.concatenate((final_data,data.rolling(window).var()), axis = 1)
    final_data = np.concatenate((final_data,rolling_fft(data, window)), axis = 1)
    #print(rolling_fft(data, window).shape)
    return final_data

#%%
#slicing the data into classes and then using window_func to do sliding window analysis
def slice_by_classes(X,y, window = 10):
    #find all the unique classes
    classes = set(y)
    print("Length of dataset is - :" +  str(len(y)))

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
    math_functions = ["_mean", "_std_dev", "_median", "_variance", "_fft"]
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
out_csv = open("C:\Honey\projects\Research gait\Flood Detection\data\Rec interval\Rec interval _ 0.1 data\windowed\\merged_windowed_with_fft.csv", "w", newline="")
with out_csv:
    writer = csv.writer(out_csv)
    writer.writerows(data)


