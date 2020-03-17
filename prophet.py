# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:42:52 2020

@author: shrav
"""

import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import math


df = pd.read_csv('/Users/egreddy/Desktop/proj_vam/more_than_200/data_59.csv',parse_dates=True)
data_59_FBP = df.copy()
del data_59_FBP['Unnamed: 0']
del data_59_FBP['item_id']

#Renaming the columns 
data_59_FBP.columns=['ds','y']
data_59_FBP.head()
data_59_FBP.plot(x='ds',y='y',figsize=(10,7))


len(data_59_FBP)
data_59_FBP.tail()


num_pred=14

#Splitting the dataset into train and test set
train = data_59_FBP.iloc[:len(data_59_FBP)-num_pred]
test = data_59_FBP.iloc[len(data_59_FBP)-num_pred:]


#Instantiating a new Prophet object
m = Prophet()

#Training the model
m.fit(train)

#num_pred=180

#Predicting the values for training examples using the instantiated object
future = m.make_future_dataframe(periods=num_pred,freq='d')
forecast = m.predict(future)
forecast.tail()
forecast_yhat=forecast.yhat[len(forecast)-num_pred-1:]

#Now testing the trained model

test.tail()
test['yhat']=forecast.yhat[len(forecast)-num_pred-1:]
test.set_index('ds',inplace=True)
test.plot(figsize=(10,7))


fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)


#writing the code to find out the smape
import math
def smape(y_true, y_pred):
    out = 0
    for i in range(y_true.shape[0]):
        a = y_true[i]
        b = y_pred[i]
        c = a+b
        if c == 0:
            continue
        out += math.fabs(a - b) / c
    out *= (200.0 / y_true.shape[0])
    return out

metric_smape = smape(test['y'], test['yhat'])#34.87



"""
data_59 = pd.read_csv('/Users/egreddy/Desktop/proj_vam/more_than_200/data_59.csv',index_col='timestamp',parse_dates=True)
data_59=data_59.asfreq(freq='d')
#
del data_59['Unnamed: 0']
del data_59['item_id']
#
data_59.head()
data_59.demand=data_59.demand.fillna(method='ffill')
data_59.demand.plot()
data_59['demand'].plot(figsize=(6,3))
plt.show()
#
from statsmodels.tsa.seasonal import seasonal_decompose
#
results = seasonal_decompose(data_59.demand)
results.observed.plot(figsize=(6,2))
results.trend.plot(figsize=(6,2))
results.seasonal.plot(figsize=(10,2))
results.resid.plot(figsize=(6,2))
"""