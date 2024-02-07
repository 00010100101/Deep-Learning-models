#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
train_data = pd.read_csv('C:\\Users\\Dell\\Documents\\dissertation_train$.csv',encoding = 'latin=1')
train_data


# In[2]:




train_data['Date']=pd.to_datetime(train_data['Date'])

train_data['Time']=pd.to_datetime(train_data['Time'])
train_data['Year'] = train_data['Date'].dt.year
train_data['Hour'] = train_data['Time'].dt.hour

train2=train_data.drop(['Date','Time','Unnamed: 6'],axis=1)
train2


# In[3]:


train_data.drop(['Date','Time'],axis=1)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
train_data.drop(['Date','Time'],axis=1)


# In[4]:


X = train2.drop(['Magnitude'], axis =1)
y = train2['Magnitude']


# In[5]:


X_train , X_test ,y_train ,y_test = train_test_split(X,y, test_size = 0.2 , random_state =20)
X_train


# In[6]:



model = Sequential()
model.add(Dense(128,input_dim=5,activation = 'relu'))




model.add(Dense(64,activation = 'relu'))
model.add(Dense(1, activation ='linear'))
model.compile(loss = 'mean_squared_error',optimizer ='adam',metrics =['mae'])
model.summary()


# In[7]:


history = model.fit(X_train,y_train,validation_split = 0.2,epochs=10)
from matplotlib import pyplot as plt
loss = history.history['loss']
val_loss =history.history['val_loss']
epochs = range(1,len(loss)+1)
plt.plot(epochs,val_loss,'r',label ='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['mae']
val_acc = history.history['val_mae']
plt.plot(epochs,acc,'y',label = 'Training MAE')
plt.plot(epochs,val_acc ,'r',label ='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[8]:


test_data = pd.read_csv('C:\\Users\\Dell\\Documents\\dissertation_test$.csv',encoding = 'latin=1')



test_data['Date']=pd.to_datetime(test_data['Date'])

test_data['Time']=pd.to_datetime(test_data['Time'])
test_data['Year'] = test_data['Date'].dt.year
test_data['Hour'] = test_data['Time'].dt.hour
test_data2=test_data.drop(['Date','Time','Magnitude'],axis=1)
test_data2


# In[11]:


predictions = model.predict(X_test[:15])


# In[12]:


test_data


# In[13]:


print("Prediction values are:",predictions)
print("Real values are:",y_test[:15])


# In[14]:


test_predict = model.predict(X_test)


# In[15]:


from sklearn.metrics import mean_squared_error
test_score =mean_squared_error(y_test,test_predict)


# In[16]:


print('Score on test set : %.2f MSE'  % test_score)


# In[17]:


X_test


# In[25]:


y_test[:15]


# In[20]:





# In[ ]:




