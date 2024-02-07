#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy 
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np


# cost-function
def f(x, y):
    return x * x + y * y


# derivative of the cost-function
def df(x, y):
    return np.asarray([2.0 * x, 2.0 * y])


def adam(bounds, n, alpha, beta1, beta2, epsilon=1e-8):
    # generate an initial point (random usually)
    x = np.asarray([0.8, 0.9])
    # initialize first moment and second moment
    m = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]

    for t in range(1, n+1):
        # gradient g(t) so the partial derivatives
        g = df(x[0], x[1])
        # update every feature independently
        for i in range(x.shape[0]):
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] ** 2
            m_corrected = m[i] / (1.0 - beta1 ** t)
            v_corrected = v[i] / (1.0 - beta2 ** t)
            x[i] = x[i] - alpha * m_corrected / (np.sqrt(v_corrected) + epsilon)

        print('(%s) - function value: %s' % (x, f(x[0], x[1])))


if __name__ == '__main__':
    adam(np.asarray([[-1.0, 1.0], [-1.0, 1.0]]), 100, 0.05, 0.9, 0.999)


# In[2]:


NUM_OF_PREV_ITEMS = 1000
def reconstruct_data(data_set,n=1):
    x,y=[] ,[]
    for i in range(len(data_set)-n-1):
        a= data_set[i:(i+n),0]
        x.append(a)
        y.append(data_set[i+n,0])
    return numpy.array(x) ,numpy.array(y)


# In[3]:


df = pd.read_csv('C:\\Users\\Dell\\Documents\\dissertation_train$2.csv',usecols=[1])
print(df)
print(df.values)


# In[37]:


data = df.values
data = df.astype('float32')


# In[38]:


scaler = MinMaxScaler(feature_range=(0,1))
data= scaler.fit_transform(data)
data


# In[39]:


train,test =data[0:int(len(data)*0.7),:],data[int(len(data)*0.7):len(data),:]
train
train_x,train_y = reconstruct_data(train,NUM_OF_PREV_ITEMS)
test_x,test_y  = reconstruct_data(test,NUM_OF_PREV_ITEMS)
train_x.shape[0],1,train_x.shape[1]
train_x.shape[0]


# In[40]:


train_x.shape[0],1,train_x.shape[1]


# In[41]:


train_y


# In[42]:


train_x.shape[0],1,train_x.shape[1]


# In[43]:


train_x.shape[0],1,train_x.shape[1]


# In[44]:


train_x =numpy.reshape(train_x,(train_x.shape[0],1,train_x.shape[1]))
test_x =numpy.reshape(test_x,(test_x.shape[0],1,test_x.shape[1]))


# In[52]:


model= Sequential()
model.add(LSTM(units=100,return_sequences=True,input_shape=(1,NUM_OF_PREV_ITEMS)))
model.add(Dropout(0.5))
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=50))
model.add(Dropout(0.3))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error',optimizer ='adam')
model.fit(train_x,train_y,epochs=10000,batch_size=32,verbose=2)


# In[70]:


test_predict = model.predict(test_x)
test_predict


# In[72]:


test_predict2 =scaler.inverse_transform(test_predict)
test_labels  =scaler.inverse_transform([test_y])
test_predict2[:15]


# In[55]:


test_labels


# In[56]:


test_score =mean_squared_error(test_labels[0],test_predict[:,0])


# In[57]:


print('Score on test set : %.2f MSE'  % test_score)


# In[58]:


test_predict_plot =numpy.empty_like(data)
test_predict_plot[:,:] = numpy.nan
test_predict_plot[len(train_x)+2*NUM_OF_PREV_ITEMS+1:len(data)-1,:] = test_predict
plt.plot(scaler.inverse_transform(data))
plt.plot(test_predict_plot,color ='red')


# In[66]:


print("Prediction values are:",predictions)
print("Real values are:",test_labels[:15])


# In[ ]:




