#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


data = pd.read_csv("Titanic.csv")
data


# In[2]:


get_ipython().system('pip install mlxtend')


# In[3]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt


# In[4]:


titanic = pd.read_csv("Titanic.csv")
titanic


# In[5]:


titanic.info()


# In[7]:


counts = titanic["Class"].value_counts()
plt.bar(counts.index,counts.values)


# In[8]:


counts = titanic["Age"].value_counts()
plt.bar(counts.index,counts.values)


# In[10]:


counts = titanic["Gender"].value_counts()
plt.bar(counts.index,counts.values)


# In[11]:


df = pd.get_dummies(titanic,dtype=int)
df.head()


# In[12]:


df.info()


# In[16]:


frequent_itemsets = apriori(df, min_support = 0.05, use_colnames=True, max_len=None)
frequent_itemsets


# In[17]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules


# In[18]:


rules.sort_values(by='lift',ascending = False)


# In[22]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[23]:


plt.scatter(rules['support'],rules['confidence'])
plt.show()


# In[ ]:


int binarySearchIterative(int arr[], int n, int target) { 
int low = 0, high = n - 1; 
while (low <= high) { 
int mid = (low + high) / 2; 
if (arr[mid] == target) { 
return mid;    
} 
if (arr[mid] > target) { 
high = mid - 1;    
} else { 
low = mid + 1;   
} 
} 
return -1;   
} 

