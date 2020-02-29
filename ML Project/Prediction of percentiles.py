# Project for predicting percentile of JEE Mains
# With Multiple Linear Regression model
# Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Percentile.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

# Splitting of dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0) 

# Fitting of linear regression model into dataset 
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the values in the training set
Y_pred = regressor.predict(X_test)
