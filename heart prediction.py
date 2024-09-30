#!/usr/bin/env python
# coding: utf-8

# ### importing dependencies

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ### data collection and processing

# In[4]:


df= pd.read_csv(r"C:\Users\Jayashri\Downloads\dataset (1).csv")


# In[5]:


df.head()


# In[6]:


df.tail()


# ### total no of rows and columns

# In[8]:


df.shape


# ### info about the data

# In[9]:


df.info()


# ### missing values
# 

# In[10]:


df.isnull().sum()


# In[11]:


df.describe()


# ### distribution of target values

# In[12]:


df['target'].value_counts()


# ### splitting features and target

# In[13]:


x= df.drop(columns='target', axis=1)
y= df['target']


# In[14]:


print(x)


# In[15]:


print(y)


# ### splitting training and testing data

# In[16]:


x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[17]:


print(x.shape, x_train.shape, x_test.shape)


# ### model training

# In[18]:


# logistic regression 


# In[20]:


model = LogisticRegression()


# In[21]:


model.fit(x_train,y_train)


# ### model evaluation

# In[22]:


# accuracy score on training data


# In[23]:


x_train_prediction= model.predict(x_train)
training_data_accuracy= accuracy_score(x_train_prediction, y_train)


# In[24]:


print('Accuracy on Training data:', training_data_accuracy)


# In[25]:


# accuracy on test data


# In[26]:


x_test_prediction=model.predict(x_test)
test_data_accuracy= accuracy_score(x_test_prediction, y_test)


# In[27]:


print('Accuracy on Training data:', test_data_accuracy)


# ### building a predicitve system

# In[31]:


input_data =(44,1,1,120,263,0,1,173,0,0,2,0,3)

#change the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the numpy array as we are predicting for only one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)

if(prediction [0]==0 ):
    print('person does not have a heart disease')
else:
          print('person has heart disease')


# In[32]:


#giving other input

input_data =(53,1,0,140,203,1,0,155,1,3.1,0,0,3)

#change the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the numpy array as we are predicting for only one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)

if(prediction [0]==0 ):
    print('person does not have a heart disease')
else:
          print('person has heart disease')


# In[ ]:




