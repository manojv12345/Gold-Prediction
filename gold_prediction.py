#importing the Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics 


#loading  the csv data to a pandas dataframe
import pandas as pd 
path="C:\\Users\\Dell\\Desktop\\Batch_04\\Data\\gld_price_data.csv" 
data=pd.read_csv(path) 
print(data)


#print first 5 rows in the dataframe 
data.head()

#print last 5 rows in the dataframe
data.tail()

#number of coloums and rows 
data.shape

#getting some basic information about the data 
data.info()

#checking number of missing values 
data.isnull().sum()

#getting the statistical measures of the data 
data.describe()

#constructing a heatmap to understand the correlation 
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize =(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8},cmap='Blues')

#correlation values of GLD
print(correlation ['GLD'])

#checing the distribution of the GLD Price 
sns.distplot(data['GLD'],color='green')

#splitting the features and targets 
X = data.drop(['Date','GLD'], axis=1)
Y = data['GLD'] 
print(X)

#spliting testing and training data 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2) 

#training the model  
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,Y_train)

#model evaluation 
#predication on test data 
test_data_prediction = regressor.predict(X_test)
print(test_data_prediction)

# R squared error 
error_score=metrics.r2_score(Y_test,test_data_prediction)
print("R squared error :", error_score)

#compare the actual values and predicted values in plot 
Y_test=list(Y_test)

plt.plot(Y_test, color='blue', label='actual value') 
plt.plot(test_data_prediction, color='green',label='predicated value')
plt.title('actual price vs predicted price')
plt.xlabel('Number of values')
plt.ylabel('GLD price')
plt.legend()
plt.show()