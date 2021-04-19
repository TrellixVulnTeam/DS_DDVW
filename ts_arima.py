#
# K. Rasku RN BSN & Programmer / Analyst
# My python data science library
# Initiated 4/19/2021
#
# TS_ARIMA.py
#
# ARIMA Time Series Forecasting
#
import ts_utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import statsmodels.tsa.seasonal as seasonal
import statsmodels.formula.api as api
import statsmodels.tools.eval_measures as eva

# Define GLOBAL variables
WINDOW_SIZE = 12
TRAIN_START = WINDOW_SIZE
TRAIN_END = 0
TEST_START = 0
TEST_END = 0

# Read in Hourly Admissions Data for 2014-2017: 35,064 hours
ts_data = pd.read_csv('data/patient_ts.csv')
ts_data_tidx = ts_data.set_index('Time')
ts_data_tidx.index = pd.to_datetime(ts_data_tidx.index, format='%Y-%m-%d %H:%M:%S')
ts = ts_data_tidx["Admits"]

# I cut this down several times.
# Particularly the ARIMA model does not work well with more than a few
# data points.  I can see this type of data ultimately requiring a
# slightly different forecasting model for each month and special
# models for special days of the year such as Christmas, the 4th of July,
# and others.
ts_2014 = ts['2015-05-01 00:00:00':'2015-05-09 23:59:59']
ts_2014.index = ts_2014.index.to_period('H')
ts_train, ts_test = ts_utils.split_train_test_ts(ts_2014, 0.2)

# Format ts for forecasting
ts_2014_copy = ts['2015-05-01 00:00:00':'2015-05-09 23:59:59']
admits_df = ts_2014_copy.reset_index()
admits_df.columns = ["Time", "Admits"]
admits_df["month_name"] = pd.DatetimeIndex(admits_df['Time']).month_name()
admits_df["month_name"] = admits_df.month_name.astype("category")
admits_df["month"] = pd.DatetimeIndex(admits_df['Time']).month
admits_df["day"] = pd.DatetimeIndex(admits_df["Time"]).day
admits_df["hour"] = pd.DatetimeIndex(admits_df["Time"]).hour

# Allocate 20% of the data for testing
admits_train, admits_test = ts_utils.split_train_test_df(admits_df, 0.2)

##################################
#  ARIMA MODEL
##################################

# Perform Augmented Dickey-Fuller test to determine 'stationarity'.
# We want this p-value to be significant.  If not, we must apply 1 or more diffs.
from statsmodels.tsa.stattools import adfuller

print("Augmented Dickey-Fuller p-value:", adfuller(admits_train["Admits"])[1])
# This interval of the data is stationary.  p-value 0.0016


# Trying ARIMA
#
# Reference:
# Brownlee, J. (2017, January 9). How to Create an ARIMA Model for Time Series Forecasting in Python.
# Machine Learning Mastery [blog].  Retrieved from https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

# First, look for a trend in the data.
ts_train.plot()
plt.show()

# Next, look at auto-correlation to see if there is a
# significant positive correlation between data lags.
# This helps to set the starting point for the AR parameter.
# I had to keep cutting the data down to reduce this value.
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(ts_train)
plt.show()

# fit a preliminary ARIMA model and plot residual errors
from statsmodels.tsa.arima.model import ARIMA

# fit model
model = ARIMA(ts_train, order=(40,1,0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())

# attempt rolling forecast
history = [x for x in ts_train]
l_predictions = list()
# walk-forward validation
for t in range(len(ts_test)):
    model = ARIMA(history, order=(40,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    l_predictions.append(yhat)
    obs = ts_test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# evaluate rolling forecasts
# mse = eval.mse(ts_test["Admits"], l_predictions)
# rmse = np.sqrt(mse)
# print("Predicted (Test) Metrics - ROLLING ARIMA \n MSE: ", mse, " RMSE: ", rmse, "\n\n")

# plot the forecasts against actual outcomes
# plt.plot(ts_test["Admits"])
# plt.plot(l_predictions, color='red')
# plt.show()