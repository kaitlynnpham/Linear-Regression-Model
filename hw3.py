#Name: Kaitlynn Pham 
#Course : CS 422 1001
#Assignment : HW 3

import pandas as pd
import numpy as np 

data = pd.read_csv('auto-mpg.data.csv')


y = data.iloc[:, 0] 
X = data.drop('mpg', axis=1)
X = X.drop('carname', axis=1)


#add bias intercept(1) to input vector 
