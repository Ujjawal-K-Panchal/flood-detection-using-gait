#%%
"""
Topic :- Model Testing
Project :- Flood Detection
Creator :- Hardik Ajmani

"""
#%%
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,Normalizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
from keras.utils import np_utils
import re
import os
import sys
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

#%%
#reading the dataset
df = pd.read_csv(os.path.join("data", "windowed", "windowed_new_data.csv"))
df = df.dropna(axis=1)
#print(df.columns.values)

#%%
#taking the X and the y
X = df.iloc[:,:-1]
y = df.iloc[:,-1]


#%%

X = X.drop(columns = [x for x in list(X) if re.search(r'MAG*', x)])
#print(len(list(X)))

#%%
#using scaler to scale the values
X_scaled = StandardScaler().fit_transform(X)
#print(X_scaled)

#l2 normalization
X_norm = Normalizer().fit_transform(X_scaled)
#pd.plotting.scatter_matrix(pd.DataFrame(data=X_norm))
#plt.show()
#TODO:
#plot each feature after normalizing using python

#%%
#one hot encoding the outputs
# this will convert each output class to binary vectors
encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)
dummy_y = np_utils.to_categorical(y_encoded)

#%%


#%%
#train test split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_encoded, test_size=0.33, shuffle=True)


#%%
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn import model_selection, linear_model
from sklearn.svm import SVR
seed = 7
"""
lr = LinearRegression()
kfold = KFold(n_splits=20, random_state=seed)
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(lr, X_scaled, y, cv=kfold, scoring=scoring)
print(results.mean(), results.std())
#print("MAE: %.3f (%.3f)") % (results.mean(), results.std())


#%%
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
clf = OneVsOneClassifier(LinearSVC(random_state=seed))
kfold = KFold(n_splits=20, random_state=seed)
scoring = 'r2'
results = model_selection.cross_val_score(clf, X_norm, y_encoded, cv=kfold, scoring=scoring)
print(results.mean(), results.std())
lf = clf.fit(X_train, y_train)
out = lf.predict(X_test)
score = 0
for i in range(len(out)):
	if out[i] == y_test[i]:
		score += 1
print(score/len(X_test) * 100)"""
#%%
"""#making a neural network regressor
def baseline_model():
    #create model
    model = Sequential()
    model.add(Dense(20, input_dim=126, activation='relu' ))
    model.add(Dense(4, activation='softmax'))

    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#%%
#creating the classifier
estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=25,verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = model_selection.cross_val_score(estimator, X_norm , dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
"""


#%%
from keras import backend as K
precision_all = []
recall_all = []
f1_all = []
	



def f1(y_true, y_pred):
	def recall(y_true, y_pred):
		"""Recall metric.

		Only computes a batch-wise average of recall.

		Computes the recall, a metric for multi-label classification of
		how many relevant items are selected.
		"""
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		recall_all.append(recall)
		return recall

	def precision(y_true, y_pred):
		"""Precision metric.

		Only computes a batch-wise average of precision.

		Computes the precision, a metric for multi-label classification of
		how many selected items are relevant.
		"""
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		precision_all.append(precision)
		return precision
	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	f1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
	f1_all.append(f1)
	return f1


# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=108, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(4, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',f1])
	print(model.summary())
	return model


#%%
"""
np.random.seed(seed)
#estimators = []
##estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=1)))
#pipeline = Pipeline(estimators)
estimator = KerasClassifier(build_fn=larger_model, epochs=100, batch_size=25,verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
#kfold = KFold(n_splits=10, random_state=seed)
results = model_selection.cross_val_score(estimator, X_scaled, dummy_y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
"""

estimator = KerasClassifier(build_fn=larger_model, epochs=50, batch_size=16,verbose=1)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)


y_true = pd.Series(y_test)
y_pr = pd.Series(y_pred)
print(y_pred)
print(confusion_matrix(y_test, y_pred))
print(precision_recall_fscore_support(y_test, y_pred))
print(pd.crosstab(y_true, y_pr, rownames=['True'], colnames=['Predicted'], margins=True))
sys.exit(0)


kfold = KFold(n_splits=15, shuffle=True, random_state=seed)
results = model_selection.cross_val_score(estimator, X_norm , dummy_y, cv=kfold)
#print(precision_all)
print(precision_all)
print(sum(precision_all)/len(precision_all)*100)
print(recall_all)
print(sum(recall_all)/len(recall_all)*100)
print(f1_all)
print(sum(f1_all)/len(f1_all)*100)
print("Precision: %.2f%% (%.2f%%)" % ((sum(precision_all)/len(precision_all))*100))
print("Recall: %.2f%% (%.2f%%)" % ((sum(recall_all)/len(recall_all))*100))
print("F1: %.2f%% (%.2f%%)" % ((sum(recall_all)/len(recall_all))*100))