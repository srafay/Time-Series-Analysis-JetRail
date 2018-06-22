import pandas as pd
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.metrics import mean_squared_error as MSE
from math import sqrt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
import statsmodels.api as sm
import numpy as np

# Load Dataset
train=pd.read_csv("data/Train_SU63ISt.csv")
test=pd.read_csv("data/Test_0qrQsBZ.csv")

train = train.drop(['ID'],axis=1)

# make a copy of original dataset
train['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
test['Datetime'] = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 

valid = train.iloc[15287:18287, :]
train = train.iloc[0:15286, :]

# Visualize trainin-validation data split
plt.figure(figsize=(40,20))
plt.plot(train.Datetime, train['Count'], label='train')
plt.plot(valid.Datetime, valid['Count'], label='validation')
plt.xlabel("Datetime")
plt.ylabel("Passenger count")
plt.legend(loc='best')
plt.show()


# Naive method to predict time series
y_hat = valid.copy()
# Assume that all next values will be the same as last observed value
y_hat['Count'] = train['Count'][len(train)-1]

# Visualize Naive method predictions
plt.figure(figsize=(40,20))
plt.plot(train.Datetime, train['Count'], label='train')
plt.plot(valid.Datetime, valid['Count'], label='validation')
plt.plot(y_hat.Datetime, y_hat['Count'], label='Naive Forecast')
plt.xlabel('Datetime')
plt.ylabel('Passenger count')
plt.legend(loc='best')
plt.show()

rmse = pd.DataFrame(columns=['Method', 'RMSE'])

# Calculate RMSE for Naive method
rmse.loc[len(rmse)]="Naive", sqrt(MSE(valid.Count, y_hat.Count))


# Moving Average Method to predict time series

# last 10 days
y_hat['Count'] = train['Count'].rolling(10).mean().iloc[-1]
# Calculate RMSE for Moving average 10 days
rmse.loc[len(rmse)]="Moving Average 10D", sqrt(MSE(valid.Count, y_hat.Count))

# last 20 days
y_hat['Count'] = train['Count'].rolling(20).mean().iloc[-1]
# Calculate RMSE for Moving average 20 days
rmse.loc[len(rmse)]="Moving Average 20D", sqrt(MSE(valid.Count, y_hat.Count))

# last 50 days
y_hat['Count'] = train['Count'].rolling(50).mean().iloc[-1]
# Calculate RMSE for Moving average 50 days
rmse.loc[len(rmse)]="Moving Average 50D", sqrt(MSE(valid.Count, y_hat.Count))

# RMSE of 10 days is better than 20 and 50 days
# Thus predictions are getting weaker as we increase number of observations

# Visualize Moving Average predictions with window size of 10 days
plt.figure(figsize=(40,20))
plt.plot(train.Datetime, train['Count'], label='train')
plt.plot(valid.Datetime, valid['Count'], label='validation')
plt.plot(y_hat.Datetime, y_hat['Count'], label='Moving average 10 days forecast')
plt.xlabel('Datetime')
plt.ylabel('Passenger count')
plt.legend(loc='best')
plt.show()


# Simple Exponential Smoothing to predict time series

y_hat = valid.copy()
fit1 = SimpleExpSmoothing(train['Count']).fit(smoothing_level=0.1, optimized=False)
y_hat['Count'] = fit1.forecast(len(valid)+1)
# Calculate RMSE for SES 0.1
rmse.loc[len(rmse)]="Simple Exp Smoothing 0.1", sqrt(MSE(valid.Count, y_hat.Count))

fit1 = SimpleExpSmoothing(train['Count']).fit(smoothing_level=0.2, optimized=False)
y_hat['Count'] = fit1.forecast(len(valid)+1)
# Calculate RMSE for SES 0.2
rmse.loc[len(rmse)]="Simple Exp Smoothing 0.2", sqrt(MSE(valid.Count, y_hat.Count))

fit1 = SimpleExpSmoothing(train['Count']).fit(smoothing_level=0.6, optimized=False)
y_hat['Count'] = fit1.forecast(len(valid)+1)
# Calculate RMSE for SES 0.6
rmse.loc[len(rmse)]="Simple Exp Smoothing 0.6", sqrt(MSE(valid.Count, y_hat.Count))

# Visualize Simple Exp Smoothing predictions with smoothing const of 0.2
plt.figure(figsize=(40,20))
plt.plot(train.Datetime, train['Count'], label='train')
plt.plot(valid.Datetime, valid['Count'], label='validation')
plt.plot(y_hat.Datetime, y_hat['Count'], label='Simple Exp Smoothing forecast')
plt.xlabel('Datetime')
plt.ylabel('Passenger count')
plt.legend(loc='best')
plt.show()



# Holt's Linear Trend Model to predcit time series

# Similar to SES but also takes trend into account

# Visualize the trend in data
sm.tsa.seasonal_decompose(np.asarray(train['Count']), freq=3).plot()
result = sm.tsa.stattools.adfuller(train['Count'])
plt.show()


# We can see that the trend is increasing
# Thus Holt's linear trend model will perform better than above methods

fit1 = Holt(train['Count']).fit(smoothing_level = 0.1,smoothing_slope = 0.0001)
y_hat['Count'] = fit1.forecast(len(valid))

# Calculate RMSE for Holt's Linear Trending Model
rmse.loc[len(rmse)]="Holt's Linear Trend 0.0001", sqrt(MSE(valid.Count, y_hat.Count))

# Visualize Holt's predictions
plt.figure(figsize=(40,20))
plt.plot(train.Datetime, train['Count'], label='train')
plt.plot(valid.Datetime, valid['Count'], label='validation')
plt.plot(y_hat.Datetime, y_hat['Count'], label='Holts Linear Trending Forecast')
plt.xlabel('Datetime')
plt.ylabel('Passenger count')
plt.legend(loc='best')
plt.show()


# Submission using Holts Linear Trending model
submission=pd.read_csv("data/Sample_Submission_QChS6c3.csv")
fit1 = Holt(np.asarray(train['Count'])).fit(smoothing_level = 0.1,smoothing_slope = 0.0001)
predict=fit1.forecast(len(test))
submission['ID'] = test['ID']
submission['Count'] = predict

# Converting the final submission to csv format
submission.to_csv("submissions/1.csv", index=False)




















