# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:01:59 2020

@author: Iacopo
"""

#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import data
df = pd.read_csv('AB_NYC_2019.csv', sep = ',')

#info
df.info()
df.isnull().sum()
df['host_id'].value_counts()
df['neighbourhood_group'].value_counts()
df['neighbourhood'].value_counts()

###DATA VISUALIZATION

#Geographical plot of the b&b distribution aroun NY
import urllib
plt.figure(figsize=(10,8))
i=urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG')
geo=plt.imread(i)
plt.imshow(geo,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])
ax=plt.gca()
df.plot(kind='scatter', x='longitude', y='latitude', ax=ax, alpha=0.4, figsize = (15,10),zorder=5)
plt.show()

#Identifying 10 most listings hosts
host10 = df['host_id'].value_counts().head(10)
plt1=host10.plot(kind='bar')
plt1.set_ylabel('Listings count')
plt1.set_xlabel('Host ID')
plt1.set_title('Listings count of NY hosts')

#Identifying 10 most reviews hosts
subdf=df[df['host_id'].isin(['219517861','107434423','30283594','137358866'
             '12243051','16098958','61391963','22541573','200380610','7503643'])]
plt2 = sns.barplot(x='host_id', y = 'number_of_reviews',data=subdf)
plt2.set_xticklabels(plt2.get_xticklabels(), rotation=45, horizontalalignment='right')

#Identifying 10 most expensive hosts
plt3 = sns.barplot(x='host_id', y = 'price',data=subdf)
plt3.set_xticklabels(plt3.get_xticklabels(), rotation=45, horizontalalignment='right')

#Identifying areas with more listings
sns.countplot(x='neighbourhood_group',data=df)

#Identifying most reviewed areas
sns.barplot(x='neighbourhood_group', y = 'number_of_reviews',data=df)

#Identifying most expensive areas
sns.barplot(x='neighbourhood_group', y = 'price',data=subdf)

#Identifying areas of the most listing hosts
plt4 = sns.countplot(x='host_id',hue = 'neighbourhood_group',data=subdf)
plt4.set_xticklabels(plt4.get_xticklabels(), rotation=45, horizontalalignment='right')

#Price as a function of area and host crossed
plt5 = sns.barplot(x='host_id', y = 'price',hue = 'neighbourhood_group',data=subdf)
plt5.set_xticklabels(plt5.get_xticklabels(), rotation=45, horizontalalignment='right')

#Price as a function of the type room
sns.pointplot(x='room_type', y = 'price',data=df,capsize=.2)

#Room type count
sns.countplot(x='room_type',data=df,capsize=.2)

#Identifying most reviewed rooms
sns.pointplot(x='room_type', y = 'number_of_reviews',data=df,capsize=.2)

####ARTIFICIAL NEURAL NETWORK 

#Splitting Features set and outcome 
X = subdf.loc[:,['host_id','neighbourhood_group','reviews_per_month']]
y = subdf['price']

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X.loc[:, ['reviews_per_month']])
X.loc[:,['reviews_per_month']] = imputer.transform(X.loc[:, ['reviews_per_month']])

#Encoding categorical data
area = pd.get_dummies(X['neighbourhood_group'],drop_first = True)
host = pd.get_dummies(X['host_id'],drop_first = True)
X = pd.concat([X,area,host],axis = 1)
X = X.drop('neighbourhood_group', axis = 1)
X = X.drop('host_id', axis = 1)

#Splitting training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20,random_state=42)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))

# Adding the second hidden layer
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse','mae'])

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, epochs = 100)

#Making the predictions and evaluating the model
y_pred = model.predict(X_test)








