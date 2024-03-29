#
# K. Rasku RN BSN & Programmer / Analyst
# My python data science library
# Initiated 1/4/2020
#
# ENRICH.py
#
#### train-test preparation
#### distributions
#### statistics
#### matrix math
#### probability
#

import numpy as np
import pandas as pd
import zlib as zl

import sklearn.preprocessing as skp
import sklearn.model_selection as ms


# split a dataframe into a test set and training set
# using a random method
# supply seed / random_state for reproducibility
#
def split_train_test_rand(data, test_ratio, seed):
    rs = np.random.RandomState(seed)
    idx_shuffle = rs.permutation(len(data))
    td_size = int(len(data) * test_ratio)
    ts_idxs = idx_shuffle[:td_size]
    tr_idxs = idx_shuffle[td_size:]
    return data.iloc[tr_idxs], data.iloc[ts_idxs]


# split a dataframe into a test set and training set
# using an index
def test_set_check(identifier, test_ratio):
    return zl.crc32(np.int64(identifier)) & Oxffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column="index"):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# split a dataframe into a test set and training set
# using randomization on strata for proportional representation
#
def split_train_test_strat(data, split_category, n_splits, test_ratio, seed):
    strat_train_set = pd.DataFrame(data=None, columns=data.columns, index=data.index)
    strat_test_set = pd.DataFrame(data=None, columns=data.columns, index=data.index)
    sd = ms.StratifiedShuffleSplit(n_splits=n_splits, test_size=test_ratio, random_state=seed)
    for train_idx, test_idx in sd.split(data, data[split_category]):
        strat_train_set = data.loc[train_idx]
        strat_test_set = data.loc[test_idx]
    return strat_train_set, strat_test_set


# split a dataframe into a test set and training set
# using the shuffle method
#
def split_train_test_shuffle(data, n_splits, test_ratio, seed=None):
    train_set = pd.DataFrame(data=None, columns=data.columns, index=data.index)
    test_set = pd.DataFrame(data=None, columns=data.columns, index=data.index)
    rs = ms.ShuffleSplit(n_splits=n_splits, test_size=test_ratio, random_state=seed)
    for train_idx, test_idx in rs.split(data):
        train_set = data.loc[train_idx]
        test_set = data.loc[test_idx]
    return train_set, test_set


# split a time series dataframe into a test set and training set
# assuming a size 6 rolling window
#
def split_train_test_ts(data, test_ratio, window_size=6):
    num_rows = data.shape[0]
    num_to_split = num_rows - (window_size * 2)
    train_num = (1 - test_ratio) * num_to_split / 100
    test_num = test_ratio * num_to_split / 100
    train_start = window_size
    train_end = window_size + train_num
    test_start = train_end
    test_end = train_end + test_num
    train_set = data.iloc[train_start:train_end,:]
    test_set = data.iloc[test_start:test_end,:]
    return train_set, test_set


# prepare Xs and Ys from test and training data sets
#
def prepareXYSets(train, test, y_col):
    train_x = train.drop(y_col, axis=1)
    train_y = train[y_col].copy()
    test_x = test.drop(y_col, axis=1)
    test_y = test[y_col].copy()
    return train_x, train_y, test_x, test_y


# prepare just x and y
#
def prepareXY(df, y_col):
    x = df.drop(y_col, axis=1)
    y = df[y_col].copy()
    return x, y


# create a categorical value from a continuous value
# may be used to create strata
#
def add_categorical_from_continuous(df, cont_col, cat_col_name, bins, labels):
    df[cat_col_name] = pd.cut(df[cont_col], bins=bins, labels=labels)


# standardize dataframe of CONTINUOUS variables to a new dataframe
# to index, join to original dataframe
#
def standardize(df, cols=None):
    scale = skp.StandardScaler()
    if cols is None:
        # standardize the whole DataFrame
        return pd.DataFrame(scale.fit_transform(df))
    else:
        for c in cols:
            df[c] = pd.DataFrame(scale.fit_transform(pd.DataFrame(df[c])), columns=[c])
        return df


# normalize dataframe of CONTINUOUS variables to a new dataframe
# to index, join to original dataframe
#
def normalize(df, cols=None):
    scale = skp.MinMaxScaler()
    if cols is None:
        # standardize the whole DataFrame
        return pd.DataFrame(scale.fit_transform(df))
    else:
        for c in cols:
            df[c] = pd.DataFrame(scale.fit_transform(pd.DataFrame(df[c])), columns=[c])
        return df


# normalize dataframe of CONTINUOUS variables to a new dataframe
# to index, join to original dataframe
#
def normalizeRobust(df, cols=None):
    scale = skp.RobustScaler()
    if cols is None:
        # standardize the whole DataFrame
        return pd.DataFrame(scale.fit_transform(df))
    else:
        for c in cols:
            df[c] = pd.DataFrame(scale.fit_transform(pd.DataFrame(df[c])), columns=[c])
        return df


# summarize dataframe of CONTINUOUS variables to a new dataframe
# IMPORTANT: 'describe' method ignores NULL and NaN values!!
#
def summarize(df):
    return pd.DataFrame(df.describe())


# One-Hot encode a CATEGORICAL variable using pd.get_dummies
# When drop=True, drop the categorical variable you used to
# create the one-hot columns
def one_hot_encode(ds, cat_attribute_name, usePrefix=True, drop=True):
    if usePrefix:
        one_hot = ds[cat_attribute_name].str.get_dummies().add_prefix(cat_attribute_name + "_")
    else:
        one_hot = pd.get_dummies(ds[cat_attribute_name])

    if drop:
        ds = ds.drop(cat_attribute_name, axis=1)
    ds = ds.join(one_hot)
    return ds


def one_hot_all(self, cat_attrs, usePrefix=True, drop=True):
    for c in cat_attrs:
        self = one_hot_encode(self, c, usePrefix=usePrefix, drop=drop)
    return self

# replace values in values_list with replacement value
# for all columns in cols_list in dataframe df.
# NOTE: use 'np.nan' in list if NaN is one of the values to replace
#       also use 'np.nan' if it is the replacement value you desire
#
def replaceCVs(df, cols_list, values_list, replacement):
    for c in cols_list:
        for v in values_list:
            df[c] = df[c].replace(v, replacement)

    return df



