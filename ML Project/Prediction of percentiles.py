# Project for predicting percentile of JEE Mains

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Percentile.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0) 

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, Y_train)

from sklearn.tree import DecisionTreeRegressor
regressor1 = DecisionTreeRegressor(random_state =0)
regressor1.fit(X_train, Y_train)   # Decision tree algorithm is more suitable for multiple linear regressor models

Y_pred = regressor1.predict(X_test)

YLR_pred = regressor.predict(X_test)