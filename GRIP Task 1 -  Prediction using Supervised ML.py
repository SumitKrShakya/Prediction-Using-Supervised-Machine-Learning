#!/usr/bin/env python
# coding: utf-8

# # GRIP - DATA SCIENCE AND BUSINESS ANALYTICS #TASK 1
# ## Prediction using Supervised ML
# ### Predict the percentage of an student based on the no. of study hours.
# ### SUMIT KUMAR SHAKYA

# In[1]:


# Importing required libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('dark_background')

# HERE,READING DATA FROM THE REMOTE LINK
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully From the remote link")
data.head(10)


# In[2]:


# Split the data into Training and Test sets
X = data.iloc[:, :-1].values   
y = data.iloc[:, 1].values  
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)


# In[3]:


X_test.shape


# In[4]:


X_train.shape


# In[5]:


y_test.shape


# In[6]:


y_train.shape


# In[7]:


# Training the Algorithm
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training Successful...")


# In[8]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.figure(dpi = 140)
plt.scatter(X, y)
plt.plot(X, line,'y');
plt.title('Hours vs Percentage')  
plt.xlabel("Hours Studied")  
plt.ylabel('Percentage Score') 
plt.show()


# In[9]:


# Testing data - In Hours
print(X_test) 

# Predicting the scores
y_pred = regressor.predict(X_test)

# Comparing Actual vs Predicted
data_frame = pd.DataFrame({'Actual Value': y_test, 'Predicted Value': y_pred})  
data_frame


# In[10]:


# Predicting score for 6 hours
hours = 6
own_pred = regressor.predict([[hours]])
print("Number of Hours are = {}".format(hours))
print("Predicted Score is  = {}".format(own_pred[0]))


# In[11]:


# Evaluating the model
from sklearn import metrics  
print('Mean Absolute Error is :', metrics.mean_absolute_error(y_test, y_pred))


# ### Hence, sucessfully completed task to Predict the percentage of an student based on the no. of study hours.
