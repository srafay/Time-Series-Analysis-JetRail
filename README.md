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
