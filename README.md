# Time Series Forecasting and Analysis


## Time Series
* Some of the definitions of time series could be:
    * Time Series is generally data which is collected over time and is dependent on it
    * A series of data points collected (or indexed) in time order is known as a time series
	* Examples
		* Stock prices
		* Count of traffic
		* Temperature over time
	* Not every data collected with respect to time represents a time series


### Components of Time Series

1. **Trend**
	* Trend is a general direction in which something is developing or changing
	* Example
		* Number of internet users over time
	* <p align="center"><img src="https://i.imgur.com/8wkV8YV.png"></p>


2. **Seasonality**
	* Any predictable change or pattern in a time series that recurs or repeats over a specific time period
	* Example
		* Consumption of Gas in summer vs winter
	* <p align="center"><img src="https://i.imgur.com/3jFCsZg.jpg"></p>
	

### Difference between a Time Series and Regression Problem
* Before studying time series components and properties, we might think that the target variable could be predicted using Regression techniques (since it's just numerical value), but these problems differ in following ways:
	* Time series is time dependent. So the basic assumption of a linear regression model that the observations are independent doesn’t hold in this case
	* Along with an increasing or decreasing trend, most Time Series have some form of seasonality trends,i.e. variations specific to a particular time frame
* So, predicting a time series using regression techniques is not a good approach

	
### Splitting the dataset
* We divide the training data given into training and validation parts
* Validation is used to check the accuracy of our predictions
	* before submitting our forecasts on testing data to the server
* We mostly divide the dataset into training and validation sets randomly
* But if we are dealing with timeseries, splitting data randomly is the worst approach
	* Here we use time-based split
	* Next predictions are based on the previous predicted values. If some values are missing (randomly transferred to validation dataset), then the predictions would be drastically effected.
* We will predict the traffic for the validation part and then visualize how accurate our predictions were
* Then we will make predictions for test dataset if we are satisfied with the performance of our model with validation dataset
* <p align="center"><img src="https://i.imgur.com/peqoK05.png"></p>


### Modeling Techniques
* There can be different modeling techniques to forecast the time series

1. **Naive Approach**
	* We assume that the next expected point is equal to the last observed point
	* So we can expect a straight horizontal line as the prediction
	* <p align="center"><img src="https://i.imgur.com/tdoSDCq.png"></p>
	* The blue line is the prediction here. All the predictions are equal to the last observed point
	* Let’s make predictions using naive approach for the validation set
	* <p align="center"><img src="https://i.imgur.com/I0aaHTo.png"></p>

2. **Moving Average**
	* In this technique we will take the average of the passenger counts for last few time periods
	* Here the predictions are made on the basis of the average of last few points instead of taking all the previously known values
	* We can take last 10, 20, 50, or any number of observations (depends on the data and intuition)
	* <p align="center"><img src="https://i.imgur.com/hHEwDK7.png"></p>
	
3. **Simple Exponential Smoothing**
	* In this technique, we assign larger weights to more recent observations than to observations from the distant past
	* The weights decrease exponentially as observations come from further in the past
	* The smallest weights are associated with the oldest observations
		* If we give the entire weight to the last observed value only, this method will be similar to the naive approach
		* So, we can say that naive approach is also a simple exponential smoothing technique where the entire weight is given to the last observed value
	* <p align="center"><img src="https://i.imgur.com/oeBY2vI.png"></p>
	
4. **Holt's Linear Trend Model**
	* It is an extension of simple exponential smoothing
		* allows forecasting of data with a trend
	* The forecast function in this method is a function of level and trend
	* We can see an inclined line as the model has taken into consideration the trend of the time series
	* <p align="center"><img src="https://i.imgur.com/6XfnWnu.png"></p>

5. **Holt's Winter Model**
	* Datasets which show a similar set of pattern after fixed intervals of a time period suffer from seasonality
	* The above models don't take seasonality of dataset into account while forecasting
	* Holt's Winter model takes seasonality into account while forecasting
	* The idea behind Holt’s Winter is to apply exponential smoothing to the seasonal components in addition to level and trend
	* <p align="center"><img src="https://i.imgur.com/fHHqjX1.png"></p>







