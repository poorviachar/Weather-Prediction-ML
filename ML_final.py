#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split  
import seaborn as sns # for plot visualization
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.svm import SVC  
import os
weather_df = pd.read_csv('weather.csv')
print(weather_df.head())

#'Humidity9am', 'Humidity3pm','Pressure3pm','WindSpeed9am',
rain_df = weather_df.loc[:,['RainToday','Humidity9am', 'Humidity3pm','WindSpeed9am','RainTomorrow']]
#removing infinte data and NAN value
rain_df = rain_df[np.isfinite(rain_df['WindSpeed9am'])]
rain_df = rain_df[np.isfinite(rain_df['Humidity3pm'])]
rain_df = rain_df[np.isfinite(rain_df['Humidity9am'])]


evaporation_df = weather_df.loc[:,['Humidity9am', 'Humidity3pm','Pressure3pm','WindSpeed9am','RainToday','RainTomorrow', 'Temp9am' ,'Temp3pm','Sunshine','Evaporation']]
sunshine_df = weather_df.loc[:,['Humidity9am', 'Humidity3pm','Pressure3pm','WindSpeed9am','RainToday','RainTomorrow', 'Temp9am' ,'Temp3pm','Evaporation','Sunshine']]

rain_df.RainToday.replace(('Yes', 'No'), (1, 0), inplace=True)
rain_df.RainTomorrow.replace(('Yes', 'No'), (1, 0), inplace=True)


print('dataset shape (rows, columns) - {rain_df.shape}')
print(rain_df.head())


X = rain_df.iloc[:, :-1].values  
y = rain_df.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  #splits the arrays and the metrics to the random train and test subsets 
#it determines the proportion of the data that has to be present on the rest split
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

#for only 1 prediction optimal is calculated as 4
classifier = KNeighborsClassifier(n_neighbors=4)  
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)

print("Starting KNN \n")

print("confusion matrix:\n")
print(confusion_matrix(y_test, y_pred))# do for k=4 and down paramters
print("REPORT:\n")
print(classification_report(y_test, y_pred)) 
#print(classification_report(X_train, X_test))



print("Starting SVM")

X = rain_df.iloc[:, :-1].values  
y = rain_df.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  

scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train) 
print("weights:") 
print(svclassifier.coef_)
print("intercept:")
print(svclassifier.intercept_)

y_pred = svclassifier.predict(X_test)  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 



print("Starting ARIMA")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import sqrt
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
weather_df = pd.read_csv('weather.csv')
print(weather_df.head())

#'Humidity9am', 'Humidity3pm','Pressure3pm','WindSpeed9am',
rain_df = weather_df.loc[:,['RainToday','Humidity9am', 'Humidity3pm','WindSpeed9am','RainTomorrow']]
#removing infinte data and NAN value
rain_df = rain_df[np.isfinite(rain_df['WindSpeed9am'])]
rain_df = rain_df[np.isfinite(rain_df['Humidity3pm'])]
rain_df = rain_df[np.isfinite(rain_df['Humidity9am'])]


evaporation_df = weather_df.loc[:,['Humidity9am', 'Humidity3pm','Pressure3pm','WindSpeed9am','RainToday','RainTomorrow', 'Temp9am' ,'Temp3pm','Sunshine','Evaporation']]
sunshine_df = weather_df.loc[:,['Humidity9am', 'Humidity3pm','Pressure3pm','WindSpeed9am','RainToday','RainTomorrow', 'Temp9am' ,'Temp3pm','Evaporation','Sunshine']]

rain_df.RainToday.replace(('Yes', 'No'), (1, 0), inplace=True)
rain_df.RainTomorrow.replace(('Yes', 'No'), (1, 0), inplace=True)


print('dataset shape (rows, columns) - {rain_df.shape}')
print(rain_df.head())

X = rain_df.iloc[:, :-1].values  
y = rain_df.iloc[:, 4].values
print(y)
print(X)
size = int(len(y) * 0.66)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  
y_train, y_test = y[0:size], y[size:len(X)]

history = y_train
predictions = list()

for t in range(len(y_test)):
	model = ARIMA(history, order=(1,0,1))
	#print(model)
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	#print(output)
	yhat = output[0]
	predictions.append(yhat)
	obs = y_test[t]

cnt=0
error = mean_squared_error(y_test, predictions)
print('test:',y_test,'pred:', predictions)
for i in range(0,len(y_test)):
	if(y_test[i]==predictions[i]):
		cnt=cnt+1
print(cnt,len(y_test))
#print(predictions)
print('Test MSE: %.3f' % error)
rmse = sqrt(error)
print('RMSE: %.3f' % rmse)
#plot
pyplot.plot(y_test)
pyplot.plot(predictions, color='red')
pyplot.show()



# In[5]:
print ("starting Decision Tress \n")

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

col_names = ['RainToday','Humidity9am', 'Humidity3pm','WindSpeed9am','RainTomorrow']
# load dataset
pima = weather_df[col_names]
print(pima.head())
feature_cols = ['RainToday','Humidity9am', 'Humidity3pm','WindSpeed9am']
pima.RainToday.replace(('Yes', 'No'), (1, 0), inplace=True)
pima.RainTomorrow.replace(('Yes', 'No'), (1, 0), inplace=True)
pima= pima[np.isfinite(pima['WindSpeed9am'])]
pima = pima[np.isfinite(pima['Humidity3pm'])]
pima = pima[np.isfinite(pima['Humidity9am'])]
X = pima[feature_cols] # Features
y = pima.RainTomorrow
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[6]:


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('weather.png')
Image(graph.create_png())


# In[ ]:




