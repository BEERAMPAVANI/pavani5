#!/usr/bin/env python
# coding: utf-8

# Multilinear regression
# 
# **Assumptions in Multilinear Regression 
# 1. Linearity: The relationship between the predictors and the resporse a linear 
# 2. Independence Deservations are ndependent of sach other 
# 3. Nomoscedasticity. The residuals (differences between observed and predicted values) exhibit constant variance at all levels of the predictor. 
# 4. Normal Distribution of Emons. The residuals of the model are normally distributed. 
# 5. No murticolineinty. The independent variables should not be too highly comslated with each other 
# 
# Violations of these assumptions may lead to inefficiency in the regression parameters and unreliable predictions 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars=pd.read_csv("Cars.csv")
cars.head()


# In[3]:


cars= pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# ### Desription of columns
# - MPG : Milege of the car(Mile per Gallon)
# - HP : Horse Power of the car
# - VOL : Volume of the car(size)
# - SP : Top speed of the car(Miles per Hour)
# - WT : Weight of the car(Pounds)

# In[4]:


cars.isna().sum()


# Observations
# 
# - There are no missing values
# - There are 81 observations
# - The data typesof the columns are relevant and valid

# In[5]:


fig, (ax_box, ax_hist)=plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, 85)}) 
#Creating a bokpint 
sns.boxplot(data=cars, x='HP', ax=ax_box, orient="h") 
ax_box.set(xlabel=' ')
##Creating a histogram in the same art 
sns.histplot(data=cars,x='HP', ax=ax_hist, bins=38, kde=True, stat="density") 
ax_hist.set(ylabel="Density") 
plt.tight_layout() 
plt.show() 


# Observations from boxplot and histograms
# 
# 1.There are some extreme values loutiersi observed in towards the right tail of SP and HP distributions 
# 
# 2.In VOL and WT columns, a few others are observed in both tais of their distributions 
# 
# 3.The extreme values of cars data may have come from the specially designed nature of cars 
# 
# 4.As this is multi-dimensional data the outliers with respect to spatial dimansions may have to be considered while building the regression madel 
# 
# Checking for duplicated rows 

# In[6]:


cars[cars.duplicated()]


# Pair plot Correlation Cofficients

# In[7]:


sns.set_style(style="darkgrid")
sns.pairplot(cars)


# In[8]:


cars.corr()


# Observations
# 
# 1.Trong correlation between HP and SP.
# 
# 2.VOL and WT show a very strong positive correlation.
# 
# 3.A negative relationship between HP and MPG, VOL and MPG, SP and MPG, and WT and MPG indicates that cars with higher performance characteristics.

# Observations from correlation plots and Coeffcients 
# 
# 1.Betwop the x variables showing moderate to high con slution strengths,highest being between HP and MPG 
# 
# 2.Therefore the dataset qualities for bulliding a multiple lincear regression model to predict MPG 
# 
# 3.Among x columns(x1,x2 x3 and x4) some very high correlation strengths are observed between SP vs HP, VOL vs WT 
# 
# 4.The high comelation among columns is not desirable as it might lead to multi collinearity problem 
# 

# Preparing a preliminary model considering all X columns 

# In[9]:


model1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[10]:


model1.summary()


# In[ ]:





# In[11]:


df1 = pd.DataFrame()
df1["actual_y1"]=cars["MPG"]
df1.head()


# In[12]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"]=pred_y1
df1.head()


# In[13]:


cars


# In[14]:


cars.head()


# In[15]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# Observations:
#     
# 1. The ideal range of VIF values shall be between 0 to 10 .However slightly higher values can be tolerated.
# 2. As seen from the very high VIF values for VOL and WT, it is clear that they are prone to multicolinearity prone.
# 3. Hence it is decided to drop one of the columns (either VOL or WT)to overcome the multicolinearity.
# 4. It is decided to drop WT and retain VOL column in further models.

# In[16]:


cars1 = cars.drop("WT",axis=1)
cars1.head()


# In[17]:


import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()


# In[18]:


model2.summary()


# In[19]:


df2 = pd.DataFrame()
df2["actual_y2"]=cars["MPG"]
df2.head()


# In[20]:


pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"]=pred_y2
df2.head()


# In[21]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"],df2["pred_y2"])
print("MSE: ",mse)
print("RMSE: ",np.sqrt(mse))


# Observations
# 
# 1.The adjusted R-Squared value improved slighty to 0.76
# 
# 2.All the p-values for model parameters are less than 5% they are significant
# 
# 3.Therefore the HP, VOL, SP columns are finalized as the significant predictor for the MPG
# 
# 4.There is no improvement in MSE value

# In[22]:


cars1.shape


# In[23]:


k= 3
p= 81
leverage_cutoff = 3*((k+1)/p)
leverage_cutoff


# In[24]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model1,alpha=0.5)

y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')

plt.show()


# . from the above plot it is evident that data points 65,70,76,78,79,80 are the influencers.
# 
# . as their H Leverage values are higher and size is higher

# In[25]:


cars[cars1.index.isin([65,70,76,78,79,80])]


# In[26]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)
cars2


# In[27]:


model3 = smf.ols('MPG~VOL+SP+HP',data=cars2).fit()


# In[28]:


model3.summary()


# In[35]:


df3 = pd.DataFrame()
df3["actual_y3"]=cars2["MPG"]
df3.head()


# In[37]:


pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"]=pred_y3
df3.head()


# In[38]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"],df3["pred_y3"])
print("MSE: ",mse)
print("RMSE: ",np.sqrt(mse))


# ### Comparison of models
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# | ---------------|---------| ------- | --------|
# | R-squared      | 0.871   | 0.770   | 0.885   |
# | Adj. R-squared | 0.858   | 0.751   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 

# In[39]:


model3.resid


# In[40]:


model3.fittedvalues


# In[41]:


import statsmodels.api as sm
qqplot=sm.qqplot(model3.resid,line='q',loc=45)
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[42]:


sns.displot(model3.resid,kde=True)


# In[43]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[44]:


plt.figure(figsize=(6,4))
plt.scatter(get_standardized_values(model3.fittedvalues),
            get_standardized_values(model3.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[ ]:




