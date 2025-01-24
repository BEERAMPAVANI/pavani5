#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
df=pd.read_csv("universities.csv")
df


# In[8]:


df[(df["GradRate"]>=95)]


# In[9]:


df[(df["GradRate"]>=80) &(df["SFRatio"]<=12)]


# In[10]:


#sorting table values
df.sort_values(ascending=False,by="SFRatio")


# In[11]:


df.sort_values(by="SFRatio")


# In[16]:


df.sort_values(by="SFRatio",ascending=False)


# In[19]:


#use groupby() to find aggregated values
sal=pd.read_csv("salaries.csv")
sal


# In[21]:


sal["salary"].groupby(sal["rank"]).mean()


# In[22]:


sal["salary"].groupby(sal["rank"]).sum()


# In[23]:


sal["salary"].groupby(sal["rank"]).median()


# In[25]:


sal[["salary","phd","service"]].groupby(sal["rank"]).mean()


# In[26]:


sal[["salary","phd","service"]].groupby(sal["rank"]).sum()


# In[29]:


sal[["salary","phd","service"]].groupby(sal["rank"]).median()


# In[32]:


#reshaping the dataframe with pivot
import pandas as pd
data={'User ID':[1,1,2,2,3,3,4,3,7],
      'MOVIE NAME':['Inception','Titanic','Inception','Avatar','Puspha','july','Happy','Puspha2','Avp'],
      'Rating':[6,8,4,9,10,9,8,10,10]
}
df=pd.DataFrame(data)
pivot_table=df.pivot(index='User ID',columns='MOVIE NAME',values='Rating')
print(pivot_table)


# In[ ]:




