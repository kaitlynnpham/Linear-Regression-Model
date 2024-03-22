#Name: Kaitlynn Pham 
#Course : CS 422 1001
#Assignment : HW 3

import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import math



data = pd.read_csv('auto-mpg.data', header=None, delimiter= '\s+')
data = data[~data.isin(['?']).any(axis=1)]


X = np.array(data.iloc[:, 1:-1])
X= X.astype(float)

y = np.array(data.iloc[:, 0])



bias = np.array(np.c_[np.ones(X.shape[0]), X])

#print(bias.ctypes)



#add bias intercept(1) to input vector 



# b = ((X^T(X))^-1)X^T(y) - use this formula to get coefficients
#\def findCoefficients(addBias_X, y):
  
 

def linearRegression(addBias_X, y):
 CV = KFold(n_splits=10, random_state=42, shuffle=True)
 for i, (train_index, test_index) in enumerate(CV.split(addBias_X)):
     X_train, X_test, y_train, y_test = addBias_X[train_index], addBias_X[test_index], y[train_index], y[test_index]
    
     coeff= np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(), X_train)), X_train.transpose()), y_train)
     #print(coeff)
     #print('\n')
     prediction = np.dot(X_test, coeff)
     print("test:")
     print(y_test)
     print ('\n')
     
     print("predictions")
     print(prediction)
     print('\n')
     #RMSE = math.sqrt(np.square(np.subtract(y_test, prediction)))
    
     #print(RMSE) 
     #print('\n')
    

#cross validation for 10 folds
linearRegression(bias, y)
