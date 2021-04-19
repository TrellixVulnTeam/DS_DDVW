#
# K. Rasku RN BSN & Programmer / Analyst
# My python data science library
# Initiated 4/18/2021
#
# TS_UTILS.py
#
# Functions related to Time Series Analysis
#

import pandas as pd


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