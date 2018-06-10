import pandas as pd
import numpy as np          # For mathematical calculations
import matplotlib.pyplot as plt  # For plotting graphs
from datetime import datetime    # To access datetime
from pandas import Series        # To work on series

# Load Dataset
train=pd.read_csv("data/Train_SU63ISt.csv")
test=pd.read_csv("data/Test_0qrQsBZ.csv")

# make a copy of original dataset
train['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
test['Datetime'] = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 

