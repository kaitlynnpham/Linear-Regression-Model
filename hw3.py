#Name: Kaitlynn Pham 
#Course : CS 422 1001
#Assignment : HW 3

#libraries
import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
import math

#read data from auto-mpg.data file, file has varying whitespaces separating column
#used delim_whitespace boolean to separate columns in data file (same as sep='\s+')
data = pd.read_csv('auto-mpg.data', header=None, delim_whitespace= True)


data = data[~data.isin(['?']).any(axis=1)]


X = np.array(data.iloc[:, 1:-1])
X= X.astype(float)
Xmean= np.mean(X, axis =0)
Xstd= np.std(X, axis =0)
X = (X-Xmean)/Xstd
y = np.array(data.iloc[:, 0])



bias = np.array(np.c_[np.ones(X.shape[0]), X])
#print(bias)
#print(bias.ctypes)



#add bias intercept(1) to input vector 



# b = ((X^T(X))^-1)X^T(y) - use this formula to get coefficients
#\def findCoefficients(addBias_X, y):
  


def linearRegression(addBias_X, y):
 CV = KFold(n_splits=10, random_state=20, shuffle=True)
 lrData = []
 rsme = []
 for i, (train_index, test_index) in enumerate(CV.split(addBias_X)):
     X_train, X_test, y_train, y_test = addBias_X[train_index], addBias_X[test_index], y[train_index], y[test_index]
    
     coeff= np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(), X_train)), X_train.transpose()), y_train)
    # print(coeff)
     intercept, *c = coeff
     prediction = np.dot(X_test, coeff)
     #print("test:")
     #print(y_test.shape)
     #print ('\n')
     
     #print("predictions")
     #print(prediction.shape)
     #print('\n')
     #RMSE = math.sqrt(np.square(np.subtract(y_test, prediction)))
     mse= np.sum(np.square(np.subtract(y_test, prediction)))
     #mse= mean_squared_error(y_test, prediction)
     # sqrt∑N i=1, (𝑦i − 𝐗𝐢𝐛)^2
     r=math.sqrt(mse)
     #print(r) 
     #print('\n')
     lrData.append(c)
     rsme.append(r)
    
 CVtable= pd.DataFrame(lrData)
 CVtable.columns=  ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration',
                    'Model Year', 'Origin Year']
 CVtable2= CVtable.assign(RSME = rsme)
 CVtable2.index = ['fold {}:'.format(i) for i in range(len(CVtable2))]

 #lrData['RMSE'] = r
 print(CVtable2)

 
#cross validation for 10 folds
linearRegression(bias, y)
