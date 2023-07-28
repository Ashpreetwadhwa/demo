#!/usr/bin/env python
# coding: utf-8

# # Iris Classification
# Iris classification is a typical machine learning classification project where we use a dataset to train the module to identify/recognise the target. There are three species of Iris flower. When a new flower is given, we need to predict it belongs to which type. Following figure shows the samples of all three species
# ![Iris_Type.jpg](attachment:Iris_Type.jpg)
# we don't have to use image processing. Some numeric measurements are given in the dataset that will help the module to classify.
# 

# ![Iris_Measure.png](attachment:Iris_Measure.png)

# # Import modules

# In[83]:


import sklearn
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from tensorflow.python.keras.models import load_model

warnings.filterwarnings('ignore')


# In[84]:


# # Loading the Dataset

# In[85]:


# load the csv data
df = pd.read_csv('IRIS.csv')
df.head()


# In[86]:


#deleting id column
df = df.drop(columns = ['ID'])
df.head()


# In[87]:


#display basic stats of data
df.describe()


# In[88]:


df.info()


# In[89]:


#disolay no. of samples on each class
df['Species'].value_counts()


# # Preprocessing dataset

# In[90]:


#check null values
df.isnull().sum()


# # Data analysis

# In[91]:


df['Sepal.Length'].hist()


# In[92]:


df['Sepal.Width'].hist()


# In[93]:


df['Petal.Length'].hist()


# In[94]:


df['Petal.Width'].hist()


# In[95]:


# create list of colors and class labels
colors = ['red', 'orange', 'blue']
species = ['virginica', 'versicolor', 'setosa']


# In[96]:


for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['Sepal.Length'], x['Sepal.Width'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[97]:


for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['Petal.Length'], x['Petal.Width'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[98]:


for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['Sepal.Length'], x['Petal.Length'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[99]:


for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['Sepal.Width'], x['Petal.Width'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# # Correlation Matrix

# In[100]:


# display the correlation matrix
df.corr()


# In[101]:


corr = df.corr()
# plot the heat map
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# # Label Encoder

# In[102]:


#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
# transform the string labels to integer
#df['Species'] = le.fit_transform(df['Species'])
#df.head()


# # Model Training and Testing

# In[103]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

# input data
X = df.drop(columns=['Species'])
# output data
Y = df['Species']
# split the data for train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[104]:


# Logistic Regression
model = LogisticRegression()
model.fit(x_train, y_train)
print("Logistic Regression Accuracy: ", model.score(x_test, y_test) * 100)


# In[105]:


# model training
model.fit(x_train.values, y_train.values)


# In[106]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[107]:


# K-nearest neighbors
model = KNeighborsClassifier()
model.fit(x_train.values, y_train.values)
print("K-nearest neighbors Accuracy: ", model.score(x_test, y_test) * 100)


# In[108]:


model.fit(x_train.values, y_train.values)


# In[109]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[110]:


# Decision Tree
model = DecisionTreeClassifier()
model.fit(x_train.values, y_train.values)
print("Decision Tree Accuracy: ", model.score(x_test, y_test) * 100)


# In[111]:


model.fit(x_train.values, y_train.values)


# In[112]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[113]:


# save the model
#import pickle
#filename = 'saved_model.sav'
#pickle.dump(model, open(filename, 'wb'))


# In[114]:


#import pickle


# In[115]:


# Save the updated model
#filename = 'saved_model.sav'
#try:
  #  with open(filename, 'wb') as file:
   #     pickle.dump(model, file)
    #print("Model saved successfully.")
#except Exception as e:
 #   print(f"Error saving the model: {e}")


# In[116]:


#load_model = pickle.load(open(filename, 'rb'))


# In[117]:


#load_model.predict([[6.0, 2.2, 4.0, 1.0]])


# In[118]:


import sklearn
print(sklearn.__version__)


# In[119]:


#x_test.head()


# In[120]:


#load_model.predict([[4,3,1,5]])


# In[121]:


import joblib

# Save the updated model using joblib
filename_joblib = 'saved_model1.joblib'
try:
    joblib.dump(model, filename_joblib)
    print("Model saved successfully using joblib.")
except Exception as e:
    print(f"Error saving the model using joblib: {e}")


# In[122]:


import joblib

# Load the updated model using joblib
filename_joblib = 'saved_model1.joblib'
loaded_model = joblib.load(filename_joblib)

# Now you can use the "loaded_model" for inference or any other operations


# In[123]:


loaded_model.predict([[5,3,1,0.2]])


# In[ ]:




