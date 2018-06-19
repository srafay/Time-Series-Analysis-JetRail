import pandas as pd
import numpy as np          # For mathematical calculations
import matplotlib.pyplot as plt  # For plotting graphs
from datetime import datetime    # To access datetime
from pandas import Series        # To work on series
from sklearn.metrics import mean_squared_error

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
plt.figure(figsize=(20,8))
plt.plot(train.Datetime, train['Count'], label='train')
plt.plot(valid.Datetime, valid['Count'], label='validation')
plt.xlabel("Datetime")
plt.ylabel("Passenger count")
plt.legend(loc='best')
plt.show()


train_X = train.iloc[:, 0]
train_y = train.iloc[:, 1]

valid_X = valid.iloc[:, 0]
valid_y = valid.iloc[:, 1]