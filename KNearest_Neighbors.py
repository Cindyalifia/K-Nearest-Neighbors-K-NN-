# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 23:03:50 2019

@author: PERSONALISE NOTEBOOK
"""

# K Nearest Neighbors

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('DataTrain_Tugas3_AI.csv')
datatest = pd.read_csv('DataTest_Tugas3_Ai.csv')
dataindeks = datatest.iloc[:, [0,1]].values
datatest = datatest.iloc[: , [1,2,3,4,5]].values
X = dataset.iloc[:, [1,2,3,4,5]].values
y = dataset.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting classifier to the Training Set
from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Predicting the dataTest results
pred = classifier.predict(datatest)

# Making the coonfusion Matrix to know how much data is wrong classification 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

h = 1  # step size in the mesh
 
# Calculate min, max and limits
x_min, x_max = dataindeks[:, 0].min() - 1, dataindeks[:, 0].max() + 1
y_min, y_max = min(pred) - 1, max(pred) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Put the result into a color plot
plt.figure()
plt.scatter(dataindeks[:, 0], pred)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Data points")
plt.show()

# Count how much data with 1,2 or 3 class
from collections import Counter
cnth = Counter(pred)
class0 = cnth[0]
class1 = cnth[1]
class2 = cnth[2]
class3 = cnth[3]

x = np.arange(4)
plt.bar(x, height= [class0, class1, class2, class3])
plt.xticks(x, ['Class 0','Class 1','Class 2','Class 3'])
