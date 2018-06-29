import pandas as pd
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.metrics import mean_squared_error as MSE
from math import sqrt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller # for Dickey Fuller test
from statsmodels.tsa.stattools import acf, pacf # for p,q in Arima Model
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import numpy as np

# Load Dataset
train=pd.read_csv("data/Train_SU63ISt.csv")
test=pd.read_csv("data/Test_0qrQsBZ.csv")

train = train.drop(['ID'],axis=1)

# make a copy of original dataset
train['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
test['Datetime'] = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 

valid = train.iloc[16056:18287, :]
train = train.iloc[0:16055, :]

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
sm.tsa.seasonal_decompose(np.asarray(train['Count']), freq=24).plot()
result = sm.tsa.stattools.adfuller(train['Count'])
plt.show()


# We can see that the trend is increasing
# Thus Holt's linear trend model will perform better than above methods

fit1 = Holt(train['Count']).fit(smoothing_level = 0.1,smoothing_slope = 0.0001)
y_hat['Count'] = fit1.forecast(len(valid) + 1)

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



# Holt's Winter Model to predict time series

# Takes into account both Seasonality and Trend

fit1 = ExponentialSmoothing(np.asarray(train['Count']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
y_hat['Count'] = fit1.forecast(len(valid))

rmse.loc[len(rmse)]="Holt's Winter Model @@7", sqrt(MSE(valid.Count, y_hat.Count))

# Visualize Holt Winter model predictions
plt.figure(figsize=(40,20))
plt.plot(train.Datetime, train['Count'], label='train')
plt.plot(valid.Datetime, valid['Count'], label='validation')
plt.plot(y_hat.Datetime, y_hat['Count'], label='Holts Winter Model Forecast')
plt.xlabel('Datetime')
plt.ylabel('Passenger count')
plt.legend(loc='best')
plt.show()


# Submission using Holts Winter model
submission=pd.read_csv("data/Sample_Submission_QChS6c3.csv")
fit1 = ExponentialSmoothing(np.asarray(train['Count']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
predict=fit1.forecast(len(test))
submission['ID'] = test['ID']
submission['Count'] = predict

# Converting the final submission to csv format
submission.to_csv("submissions/2.csv", index=False)


# Use Dickey Fuller test to check stationarity of the series

# Null Hypothesis: Time series is not stationary
# If Test statistics < Critical value, reject Null Hypothesis

def test_stationarity(timeseries, title='Rolling Mean & Standard Deviation'):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(center=False,window=24).mean() # 24 hours on each day
    rolstd = timeseries.rolling(center=False,window=24).std()
    
    #Plot rolling statistics:
    plt.figure(figsize=(20,10))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title(title)
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


test_stationarity(train['Count'])


# Since Test stats < Critical value, Series is stationary
# But we can see the increasing trend, let's remove it first

Train_log = np.log(train['Count'])
valid_log = np.log(valid['Count'])

moving_avg = pd.rolling_mean(Train_log, 24)

plt.figure(figsize=(20,10))
plt.plot(train.Datetime, Train_log, label='log(train)')
plt.plot(train.Datetime, moving_avg, color = 'red', label='Rolling_Mean(log(train))')
plt.legend(loc='best')
plt.title('Transforming series by taking the log')
plt.show()

# Remove the increasing trend

train_log_moving_avg_diff = Train_log - moving_avg

# The first 23 values in difference are Nan, drop them

train_log_moving_avg_diff.dropna(inplace = True)

# Since value of Test Statistics is very small compared to Critical Value
# Thus it means that the trend has almost been removed

test_stationarity(train_log_moving_avg_diff)
# And from plot we can see that std is stable

# Mean of the series is still fluctuating
# Stabilize the mean by Shifting and subtracting the train values

train_log_diff = Train_log - Train_log.shift(1)

# remove the first nan value and plot the series
test_stationarity(train_log_diff.dropna())



# Removing Seasonality

decomposition = seasonal_decompose(pd.DataFrame(Train_log).Count.values, freq = 24)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(50,10))
plt.subplot(411)
plt.plot(train.Datetime, Train_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(train.Datetime, trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(train.Datetime, seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(train.Datetime, residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# Check the stationarity of residuals

train_log_decompose = pd.DataFrame(residual)
train_log_decompose['date'] = Train_log.index
train_log_decompose.set_index('date', inplace = True)
train_log_decompose.dropna(inplace=True)
test_stationarity(train_log_decompose[0], title='Stationarity of Residuals')


# Forecasting the time series using Arima

lag_acf = acf(train_log_diff.dropna(), nlags=25)
lag_pacf = pacf(train_log_diff.dropna(), nlags=25, method='ols')

# Finding the values of p and q for Arima Model
plt.figure(figsize=(20,10))
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.show()

plt.figure(figsize=(20,10))
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.show()

# Make AR Model
model = ARIMA(train.Count, order=(1, 1, 0))  # here the q value is zero since it is just the AR model
results_AR = model.fit(disp=-1)  
plt.figure(figsize=(50,10))
plt.plot(train.Count, label='original')
plt.plot(results_AR.fittedvalues, color='red', label='predictions')
plt.title('Predictions using Auto Regression Model')
plt.legend(loc='best')
plt.show()


# Convert the values to original scale and plot the validation curve for AR Model

AR_predict=results_AR.predict(start=16055, end=18285)
AR_predict=AR_predict.cumsum().shift().fillna(0)
AR_predict1=pd.Series(np.ones(valid.shape[0]) * np.log(valid['Count']), index = valid.index)
AR_predict1=AR_predict1.add(AR_predict,fill_value=0)
AR_predict = np.exp(AR_predict1)


plt.figure(figsize=(50,20))
plt.plot(valid['Count'], label = "Valid")
plt.plot(AR_predict, color = 'red', label = "Predict")
plt.legend(loc= 'best')
plt.title("Validation Curve Using Auto Regression Model")
plt.show()

def check_prediction_diff(predict_diff, given_set):
    predict_diff= predict_diff.cumsum().shift().fillna(0)
    predict_base = pd.Series(np.ones(given_set.shape[0]) * np.log(given_set['Count']), index = given_set.index)
    predict_log = predict_base.add(predict_diff,fill_value=0)
    predict = np.exp(predict_log)
    
    plt.figure(figsize=(20,10))
    plt.plot(given_set['Count'], label = "Given set")
    plt.plot(predict, color = 'red', label = "Predict")
    plt.legend(loc= 'best')
    temp = np.sqrt(np.dot(predict, given_set['Count']))/given_set.shape[0]
    plt.title('RMSE: ' + temp)
    plt.show()
    
def check_prediction_log(predict_log, given_set):
    predict = np.exp(predict_log)
    plt.figure(figsize=(20,10))
    plt.plot(given_set['Count'], label = "Given set")
    plt.plot(predict, color = 'red', label = "Predict")
    plt.legend(loc= 'best')
    #plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['Count']))/given_set.shape[0]))
    plt.show()
    
check_prediction_diff(AR_predict, valid)

plt.figure(figsize=(20,10))
plt.plot(valid['Count'], label = "Given set")
plt.plot(AR_predict, color = 'red', label = "Predict")
plt.legend(loc= 'best')
#plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set['Count']))/given_set.shape[0]))
plt.show()

def gridSearchSARIMAX():
    for p in range(12):
        for d in range(12):
            for q in range(12):
                try:
                    fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(p, d, q),seasonal_order=(0,1,1,7)).fit()
                    y_hat = fit1.predict(start=16055, end=18285, dynamic=True)
                    plt.figure(figsize=(50,20))
                    plt.plot( train['Count'], label='Train')
                    plt.plot(valid['Count'], label='Valid')
                    plt.plot(y_hat, label='SARIMA')
                    plt.title("p={}, d={}, q={}" .format(p,d,q))
                    plt.legend(loc='best')
                    plt.savefig("results/p={}, d={}, q={}" .format(p,d,q))
                except:
                    continue

# Calculate RMSE
fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(3, 1, 2),seasonal_order=(0,1,1,7), enforce_stationarity=False, enforce_invertibility=False).fit()
y_hat = fit1.predict(start=16055, end=18285, dynamic=True)
rmse.loc[len(rmse)]="SARIMAX 312_", sqrt(MSE(valid.Count, y_hat))


# Submission using SARIMAX Model 312
submission=pd.read_csv("data/Sample_Submission_QChS6c3.csv")
fit1 = sm.tsa.statespace.SARIMAX(train.Count, enforce_stationarity=False, enforce_invertibility=False, order=(3, 1, 2),seasonal_order=(0,1,1,7)).fit()
predict = fit1.predict(start=18286, end=23397, dynamic=True)
submission.Count = predict
submission['ID'] = test['ID']

# Converting the final submission to csv format
submission.to_csv("submissions/5.csv", index=False)





