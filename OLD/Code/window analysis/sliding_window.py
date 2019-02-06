#%%
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
df = pd.read_csv("your_file.csv")

df = df.dropna(axis=1)

#%%
#taking the X and the y
X = df.iloc[:,2]
y = df.iloc[:,-1]

#%%
print(X.values)

#%%
df["sum"] = X.rolling(10).sum()
df["mean"] = X.rolling(10).mean()
df["std_dev"] = X.rolling(10).std()
df["median"] = X.rolling(10).median()
df["var"] = X.rolling(10).var()
df["min"] = X.rolling(10).min()
df["max"] = X.rolling(10).max()
df["corr"] = X.rolling(10).corr()
df["cov"] = X.rolling(10).cov()



df["X"] = X

#%%

df[1100:1200].plot(y=["X","var","std_dev","median","corr","cov"], figsize=(16,12), title="Land")
#df[2800:2900].plot(y=["X","var","std_dev","median","corr","cov"], figsize=(16,12), title="0.19")
#df[3400:3500].plot(y=["X","var","std_dev","median","corr","cov"], figsize=(16,12), title="2.5")
#df[5500:5600].plot(y=["X","var","std_dev","median","corr","cov"], figsize=(16,12), title="4.5")

plt.show()

