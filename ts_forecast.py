#
# K. Rasku RN BSN & Programmer / Analyst
# My python data science library
# Initiated 4/18/2021
#
# TS_FORECAST.py
#
# Basic Time-Series Data Analysis
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import statsmodels.tsa.seasonal as seasonal
import statsmodels.formula.api as api

# Define GLOBAL variables
WINDOW_SIZE = 12
TRAIN_START = WINDOW_SIZE
TRAIN_END = 0
TEST_START = 0
TEST_END = 0

# Define FUNCTIONS for splitting training and testing data
def split_train_test_df(data, test_ratio):
    '''
    split a time series dataframe into a test set and training set
    '''
    global WINDOW_SIZE, TRAIN_START, TRAIN_END, TEST_START, TEST_END
    num_rows = data.shape[0]
    num_to_split = num_rows - (WINDOW_SIZE * 2)
    train_num = int((1 - test_ratio) * num_to_split)
    test_num = num_to_split - train_num
    TRAIN_END = train_num + WINDOW_SIZE
    TEST_START = TRAIN_END
    TEST_END = TRAIN_END + test_num
    train_set = data.iloc[TRAIN_START:TRAIN_END, :]
    test_set = data.iloc[TEST_START:TEST_END, :]
    return train_set, test_set


def split_train_test_ts(data, test_ratio):
    '''
    split a series into a test set and training set
    '''
    global WINDOW_SIZE, TRAIN_START, TRAIN_END, TEST_START, TEST_END
    num_rows = data.shape[0]
    num_to_split = num_rows - (WINDOW_SIZE * 2)
    train_num = int((1 - test_ratio) * num_to_split)
    test_num = num_to_split - train_num
    TRAIN_END = train_num + WINDOW_SIZE
    TEST_START = TRAIN_END
    TEST_END = TRAIN_END + test_num
    train_set = data.iloc[TRAIN_START:TRAIN_END]
    test_set = data.iloc[TEST_START:TEST_END]
    return train_set, test_set


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

# Decompose & visualize this time series
# It looks like this may or may not have captured the seasonality
# We will have to see how the models turn out.
# Residuals plot looks pretty good, however.
ts_2014_copy = ts['2015-05-01 00:00:00':'2015-05-09 23:59:59']
plt.rc("figure", figsize=(32, 30))
result = seasonal.seasonal_decompose(ts_2014_copy, model='additive')
result.plot()
plt.show()

# Format ts for forecasting
admits_df = ts_2014_copy.reset_index()
admits_df.columns = ["Time", "Admits"]
admits_df["trend"] = admits_df['Time'].map(result.trend)
admits_df["month_name"] = pd.DatetimeIndex(admits_df['Time']).month_name()
admits_df["month_name"] = admits_df.month_name.astype("category")
admits_df["month"] = pd.DatetimeIndex(admits_df['Time']).month
admits_df["day"] = pd.DatetimeIndex(admits_df["Time"]).day
admits_df["hour"] = pd.DatetimeIndex(admits_df["Time"]).hour

# Allocate 20% of the data for testing
admits_train, admits_test = split_train_test_df(admits_df, 0.2)
ts_train, ts_test = split_train_test_ts(ts_2014, 0.2)

##################################
#  TREND MODEL
##################################
trend_model = api.ols('Admits ~ trend', data=admits_train).fit()
p = trend_model.params
print(trend_model.summary())

predicted = trend_model.predict(admits_test)
fitted = trend_model.predict(admits_train)

# for plotting the area where train and test data meet
admits_subset = admits_df.iloc[TRAIN_END-50:TEST_START+50,:]
test_subset = admits_test.head(50)
train_subset = admits_train.tail(50)
predicted_sub = predicted.head(50)
fitted_sub = fitted.tail(50)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(admits_subset['Time'], admits_subset['Admits'])
ax.plot(test_subset['Time'], predicted_sub, 'r')
ax.plot(train_subset['Time'], fitted_sub, 'purple')
ax.plot()
plt.show()

# TRAIN METRICS & RESIDUALS
y_true = admits_df.iloc[TRAIN_START:TRAIN_END, 1]
mse = metrics.mean_squared_error(y_true, fitted)
rmse = np.sqrt(mse)
print("Fitted (Train) Metrics - TREND \n MSE: ", mse, " RMSE: ", rmse, "\n\n")

# TRAIN RESIDUAL PLOT FOR TREND MODEL
residual = y_true - fitted
plt.scatter(fitted, residual)
plt.show()

# TEST METRICS & RESIDUALS
y_true2 = admits_df.iloc[TEST_START:TEST_END, 1]
mse = metrics.mean_squared_error(y_true2, predicted)
rmse = np.sqrt(mse)
print("Predicted (Test) Metrics - TREND \n MSE: ", mse, " RMSE: ", rmse, "\n\n")

# TEST RESIDUAL PLOT FOR TREND MODEL
residual2 = y_true2 - predicted
plt.scatter(predicted, residual2)
plt.show()

##################################
#  TREND & SEASONAL MODEL
##################################
seasonal_model = api.ols('Admits ~ trend + month + day + hour', data=admits_train).fit()
print(seasonal_model.summary())

predicted_m = seasonal_model.predict(admits_test)
fitted_m = seasonal_model.predict(admits_train)

# for plotting
predicted_m_sub = predicted_m.head(50)
fitted_m_sub = fitted_m.tail(50)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(admits_subset['Time'], admits_subset['Admits'])
ax.plot(test_subset['Time'], predicted_m_sub, 'r')
ax.plot(train_subset['Time'], fitted_m_sub, 'purple')
ax.plot()
plt.show()

# TRAIN METRICS & RESIDUALS - SEASONAL MODEL
mse = metrics.mean_squared_error(y_true, fitted_m)
rmse = np.sqrt(mse)
print("Fitted (Train) Metrics - SEASONAL \n MSE: ", mse, " RMSE: ", rmse, "\n\n")

# TRAIN RESIDUAL PLOT FOR SEASONAL MODEL
residual_m1 = y_true - fitted_m
plt.scatter(fitted_m, residual_m1)
plt.show()

# TEST METRICS & RESIDUALS
mse = metrics.mean_squared_error(y_true2, predicted_m)
rmse = np.sqrt(mse)
print("Predicted (Test) Metrics - SEASONAL \n MSE: ", mse, " RMSE: ", rmse, "\n\n")

# TEST RESIDUAL PLOT FOR SEASONAL MODEL
residual_m2 = y_true2 - predicted_m
plt.scatter(predicted_m, residual_m2)
plt.show()


