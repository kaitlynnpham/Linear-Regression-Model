#Name: Kaitlynn Pham 
#Course : CS 422 1001
#Assignment : HW 3

#libraries
import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold #CV fold
import math #use for rsme equation

#read data from auto-mpg.data file, file has varying whitespaces separating column
#used sep='\s+' to separate all whitespaces 
data = pd.read_csv('auto-mpg.data', header=None, sep= '\s+')

#preprocess data
#used isin to check for missing values , set data to not include missing data with ~, 
#axis =1 (rows)
data = data[~data.isin(['?']).any(axis=1)]
#set X(features), drop first and last column of data
X = np.array(data.iloc[:, 1:-1])
#change all values to float to calcuate 
X= X.astype(float)
#standardize X values using z score ; z = (X -mean)/standard deviation 
#mean
mean= np.mean(X, axis =0)
#sd
sd= np.std(X, axis =0)
#normalized  X
X = (X-mean)/sd
#set y values to first column (mpg)
y = np.array(data.iloc[:, 0])


#linear regression function 
def linearRegression(X, y):
 #add bias constant to X (set to 1)
#.c_ used to add first column , np.ones set an array filled with ones and with shape of X
 addBias_X = np.array(np.c_[np.ones(X.shape[0]), X])
 #10 fold cross validation, shuffle data using random_state 20 
 crossValidation = KFold(n_splits=10, random_state=20, shuffle=True)
#arrays to output data 
 LRData = []
 rsme = []
 #loop through each fold 
 for i, (train_i, test_i) in enumerate(crossValidation.split(addBias_X)):
     #set train and test data 
     trainX, testX, trainY, testY = addBias_X[train_i], addBias_X[test_i], y[train_i], y[test_i]
     #get intercept and coefficents
     #equation : b =(X'X)^-1(X'y)
     coeffVector= np.dot(np.dot(np.linalg.inv(np.dot(trainX.transpose(), trainX)), trainX.transpose()), trainY)
     #separate the intercept(first index) from the feature coefficients
     intercept, *coeff = coeffVector
     #make a prediction from the coefficients and test data 
     prediction = np.dot(testX, coeffVector)
     #Find root mean square error 
     # sqrt∑N i=1, (𝑦i − 𝐗𝐢𝐛)^2 ; find sum of the squares of testY[i] - prediction(testX * coeff)
     mse= np.sum(np.square(np.subtract(testY, prediction)))
     #calculate square root of the mean square error
     r=math.sqrt(mse)
     #add coeff array at fold[i] to LRData array 
     LRData.append(coeff)
     #add rsme value to the rsme array 
     rsme.append(r)

#create data frame with the LRData(coeff arrays)  
 CVtable= pd.DataFrame(LRData)
 #add column name for each index of coeff array 
 CVtable.columns=  ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration',
                    'Model Year', 'Origin Year']
 #use assign to add another column for 'RSME" with the rsme array 
 CVtable= CVtable.assign(RSME = rsme)
 #add row names using .index ; fold starting at 0 until end of rows in table
 CVtable.index = ['fold {}:'.format(i) for i in range(len(CVtable))]

 #output the table ; includes coefficients for each feature and RSME in each CV fold 
 print(CVtable)

 
#call linear regression function using X and y data 
linearRegression(X, y)
