#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow import convert_to_tensor
from sklearn.preprocessing import MinMaxScaler


# In[6]:


#import data into a pandas dataframe 
file = 'Downloads/ETH-USD 5y.csv'
df = pd.read_csv(file)
df = df.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'PerChangeRough', 'PofMaxVol', 'PoVolRough'], 1)


# In[7]:


#I create a nested list that will be the inputs/outputs for the model. Each list has the percent change of ETH each day for the
#past week and the final element is 1 if ETH increased in price the final day or 0 if it decreased.
list1 = []
for x in range(10,len(df)):
    entry = []
    for i in range(-7,0):
        pc = df.iloc[x+i]['PerChange']
        entry.append(pc)
    entry.append(df.iloc[x]['Increased'])
    list1.append(entry)


# In[10]:


#Scaling the data with MinMaxScaler changes every data to a number between 0 and 1. This process helps speed up learning.
dataset = np.asarray(list1)
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)
dataset


# In[12]:


#Convert the data to tensors because its more efficient for keras models.
X1 = convert_to_tensor(dataset[:1600,0:7])
y1 = convert_to_tensor(dataset[:1600,7])

testx1 = convert_to_tensor(dataset[1600:,0:7])
testy1 = convert_to_tensor(dataset[1600:,7])


# In[14]:


#reshape the data because LSTM models expect 3-D data
X1 = np.reshape(X1, (X1.shape[0], X1.shape[1], 1))


# In[2]:


#Create and train the data.
#I chose LSTM as it's typically good for data over time. It also performed better than only dense layers.
#Adding any data pertaining to volume didn't increase accuracy very much and made the model much more complicated so I removed it.
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X1.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X1, y1, epochs=100, batch_size=128, validation_split = .2, shuffle=True)


# In[1]:


#test the model
model.predict(testx1, testy1)


# In[1]:




