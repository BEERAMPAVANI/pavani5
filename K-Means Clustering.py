#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[3]:


Univ = pd.read_csv("Universities.csv")


# In[4]:


Univ


# In[6]:


Univ.info()


# In[7]:


Univ.describe()


# In[9]:


Univ.isna().sum()


# # Standardization of the data 

# In[10]:


Univ1=Univ.iloc[:,1:]


# In[12]:


Univ1


# In[15]:


cols=Univ1.columns


# In[16]:


# Standardisation function 
from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler() 
scaled_Univ_df= pd.DataFrame(scaler.fit_transform(Univ1), columns=cols) 
scaled_Univ_df


# In[19]:


from sklearn.cluster import KMeans
clusters_new=KMeans(3,random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[20]:


clusters_new.labels_


# In[24]:


set(clusters_new.labels_)


# In[26]:


Univ['clusterid_new']=clusters_new.labels_


# In[27]:


Univ


# In[28]:


Univ[Univ['clusterid_new']==1]


# In[29]:


Univ[Univ['clusterid_new']==2]


# In[30]:


Univ[Univ['clusterid_new']==0]


# In[32]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# # Observations:
# 
# - cluster 2 appears to be the top rated universities cluster as the cut off score,Top10,SFRatio parameter means values ar high
# 
# - cluster 1 appears to occupy the middle level rayed universities
# 
# - cluster 0 comes as the lower level rated universities

# In[ ]:


wcss=[]
for i in range(1,20):
    kmeans= KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_Univ_df) 
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20),wcss)
pit.title('Elbow Mwthod')
plt.xlabel('Number of clusters') 
plt.ylabel('WCSS')
plt.show()


# In[ ]:




