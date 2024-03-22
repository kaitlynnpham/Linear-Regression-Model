#Name: Kaitlynn Pham 
#Course : CS 422 1001
#Assignment : HW 3

import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold

data = pd.read_csv('auto-mpg.data.csv')

X = data.drop('mpg', axis=1)
X = X.drop('carname', axis=1)
y = data.iloc[:, 0] 


#add bias intercept(1) to input vector 
bias = np.ones(X.shape[0], 1) 
addBias_X = np.c_[bias, X]   

# b = ((X^T(X))^-1)X^T(y) - use this formula to get coefficients

coefficient = np.linalg.inv(addBias_X.T.dot(addBias_X)).dot(addBias_X).dot(y)
#cross validation for 10 folds
KFold(n_splits=10, random_state=None, shuffle=False)
