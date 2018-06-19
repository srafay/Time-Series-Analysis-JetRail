import pandas as pd
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.metrics import mean_squared_error as MSE
from math import sqrt

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

# Calculate RMSE for Naive method
rmse_Naive = sqrt(MSE(valid.Count, y_hat.Count))
print ("RMSE for Naive method is {}" .format(rmse_Naive))


# Moving Average Method to predict time series

# last 10 days
y_hat['Count'] = train['Count'].rolling(10).mean().iloc[-1]
# Calculate RMSE for Moving average 10 days
rmse_MV_10 = sqrt(MSE(valid.Count, y_hat.Count))
print ("RMSE for 10 days moving avg is {}" .format(rmse_MV_10))

# last 20 days
y_hat['Count'] = train['Count'].rolling(20).mean().iloc[-1]
# Calculate RMSE for Moving average 20 days
rmse_MV_20 = sqrt(MSE(valid.Count, y_hat.Count))
print ("RMSE for 20 days moving avg is {}" .format(rmse_MV_20))

# last 50 days
y_hat['Count'] = train['Count'].rolling(50).mean().iloc[-1]
# Calculate RMSE for Moving average 50 days
rmse_MV_50 = sqrt(MSE(valid.Count, y_hat.Count))
print ("RMSE for 50 days moving avg is {}" .format(rmse_MV_50))

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



train_X = train.iloc[:, 0]
train_y = train.iloc[:, 1]

valid_X = valid.iloc[:, 0]
valid_y = valid.iloc[:, 1]