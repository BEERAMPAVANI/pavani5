#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[2]:


iris = pd.read_csv("iris.csv")


# In[3]:


iris


# In[4]:


import seaborn as sns
counts = iris["variety"].value_counts()
sns.barplot(data=counts)


# In[5]:


iris.info()


# In[6]:


iris[iris.duplicated(keep=False)]


# # Observations
# 
# 
# There are 150 rows and 5 columns
# 
# There are no Null values
# 
# There is one duplicated row
# 
# The x-colomns are sepal.length,sepal.width,petal.length and petal.width
# 
# All the x-columns are continuous
# 
# The y-column is "veriety" which is categorical
# 
# There are three flower categories(classes)

# In[7]:


iris=iris.drop_duplicates(keep='first')


# In[8]:


iris[iris.duplicated]


# In[9]:


iris=iris.reset_index(drop=True)
iris


# In[10]:


labelencoder=LabelEncoder()
iris.iloc[:,-1]=labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[11]:


iris.info()


# # Observations
# 
# - the target column ('variety') is still object type . it needs to be converted to numeric(int)

# In[12]:


iris['variety']=pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[13]:


X=iris.iloc[:,0:4]
Y=iris['variety']


# In[14]:


Y


# In[15]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
x_train


# # Building Decision Tree Classifier using Entropy Criteria

# In[16]:


model=DecisionTreeClassifier(criterion = 'entropy',max_depth=None)
model.fit(x_train,y_train)


# In[17]:


plt.figure(dpi=1200)
tree.plot_tree(model);


# In[ ]:




