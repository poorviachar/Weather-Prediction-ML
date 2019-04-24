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




