#
# K. Rasku RN BSN & Programmer / Analyst
# My python data science library
# Initiated 1/4/2021
# Last Updated 2/16/2021
#
# MANIP.py
#
#### regex
#### substitutions / stripping
#### sub-setting
#### date aggregation
#### dataframe aggregation and manipulation
#### data set balancing
#### categorical encoding

# This library requires the use of the following python libraries.
# Before running this code, please make sure they are installed.
# NOTE: doCleanupEncode also uses the library category_encoders

import re
import datetime
import numpy as np
import pandas as pd
import sklearn.utils as sku


# count_matches
# counts all the matches in a file, f for any regex, using findall
# provide the regular expression with r'[match]' syntax
# returns: the number of matches, an integer
# requires: re
#
def count_matches(f, regex):
    fh = open(f, "r")
    file_data = fh.read()
    matches = re.findall(regex, file_data)
    return len(matches)


# sub_and_write
# substitutes all instances of a matched regex with replacement text
# writes the changes to a new file called [f]_new.[same suffix]
# provide the regular expression with r'[match]' syntax
# requires: re
#
def sub_and_write(f, regex, replacement):
    fh = open(f, "r")
    file_data = fh.read()
    new_data = re.sub(regex, replacement, file_data)
    print(new_data)
    print("\n")

    filename_list = f.split(".")
    nfnp = filename_list[0]
    nfn = nfnp + "_new." + filename_list[1]

    nfnh = open(nfn, "w")
    nfnh.write(new_data)

    # check (debug; small files only!)
    # block_operator(nhn, do_func(print))


# sub-setting
# set_names, and a dictionary of what belongs in what set
# returns: a dataframe
# requires: pandas
#
# def sub_set(set_names, set_descriptions):


# agg_dates
# bucketize dates by week, month, quarter, year, or fiscal year
# returns: a dictionary
# requires: datetime
#
# def agg_dates(agg_type, date_list, date_format_string):
#
#    date_dictionary = dict.fromkeys(['week', 'month', 'quarter', 'year', 'fiscal year'])
#    if agg_type == "week":

#    elif agg_type == "month":

#    elif agg_type == "quarter":

#    elif agg_type == "year":

#    else:
# fiscal year

#    return date_dictionary


# stack_dataframes
#     files_list: a list of csv files to convert to dataframes and stack
#     diff_column_name: the name of a column to create to differentiate the three data-sets in the new stacked dataframe.  Example: YEAR, SETNO
#     diff_value_expr: a regular expression that, when applied to the file name, finds the value to be applied to the new diff_column_name.
#                      Example: [0-9]+ to find a number in the name of a file ("mydata2017.csv")
#     orientation: If 0 (default), stack data on to of each other (height-wise) or zero axis, or if 1, stack data side by side (length-wise) or 1 axis.
#     add_index: If 0, add no index; if 1 (default) add an index and a unique ID combining the index and the value of diff_column_name
#     returns:  a big or wide dataframe
#     requires: pandas
#
def stack_dataframes(files_list, diff_column_name, diff_value_expr, orientation=0, add_index=1):
    dframes = {}
    for f in files_list:
        dframes[f] = pd.read_csv(f)
        dframes[f][diff_column_name] = re.findall(diff_value_expr, f)[0]
        if add_index:
            dframes[f]["Index"] = dframes[f].index
            dframes[f]["ID"] = dframes[f][diff_column_name].map(str) + dframes[f]["Index"].map(str)

    return pd.concat(dframes, orientation)


# combine_columns; transform a dataframe
#   combine_list: two-element list of names of cols that will be combined
#   new_col: name of the new column that will be added to the dataframe
#            containing the combined values of combine_list
#   requires: pandas
#
def combine_columns(df, combine_list, new_col):
    if len(combine_list) == 2:
        df[new_col] = df[combine_list[0]].combine_first(df[combine_list[1]])


# combine_by_re; transform a dataframe
#   col_expr: a regex that matches the name(s) of the columns to combine
#   new_col: name of the new column that will be added to the dataframe
#            containing the combined values of m_cols
#   requires: pandas, re
#
def combine_by_re(df, col_expr, verbose=True):
    # faster execution time than list(df)
    all_cols = df.columns.values.tolist()
    m_cols = re.findall(col_expr, " ".join(all_cols))

    col_root = col_expr.split('[')[0]
    new_col = ""
    if len(re.findall(r'^_', col_root)) > 0:
        new_col = "N" + col_root
    else:
        new_col = "N_" + col_root

    if len(m_cols) == 2:
        if verbose:
            print("Combining")
            print(m_cols)
            print("Into")
            print(new_col)
        combine_columns(df, m_cols, new_col)
    elif len(m_cols) == 1:
        # don't do anything, nothing to combine
        return
    else:
        # too many things to combine
        print("Unable to combine ")
        print(m_cols)


# create_binary; add a new binary column to a dataframe based on the values in another column
#   cols: one or more existing columns to be mapped to the new binary column
#   yes_values: a list of the values of the existing column(s) to be associated with the value 1, or True
#   no_values: a list of the values of the existing column(s) to be associated with the value 0, or False
#   new_col: the name you want to give to the new binary column example) B_DIABETES
#   requires: numpy
#
def create_binary(df, cols, yes_values, no_values, new_col, override_new_values=None):
    for c in cols:
        cur_values = [df[c].isin(yes_values), df[c].isin(no_values)]
        if override_new_values is not None:
            new_values = override_new_values
        else:
            new_values = [1, 0]
        df[new_col] = np.select(cur_values, new_values)


# recode_col: create a new_col in dataframe (df), translating the values of existing_col
# (keys of mapping_dict), to new values (values of mapping_dict)
# example mapping_dict: { 1 : 'Alabama', 2 : 'Alaska', 4 : 'Arizona', 5 : 'Arkansas', 6 : 'California', 8 : 'Colorado', 9: 'Connecticut'}
# requires: pandas
#
def recode_col(df, existing_col, mapping_dict, new_col):
    df[new_col] = df[existing_col].map(mapping_dict)


# numeric_only: return a new dataframe that is a subset of the dataframe argument (df)
# containing only continuous variables.
# This only works if categorical variables and binary variables are NOT coded.
# requires: pandas
#
def numeric_only(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    return df.select_dtypes(include=numerics)


# For categorical data
# Supply y from the training data, if using WOE or LOO encoding
#
def doCleanupEncode(X, y=None, cat=None, oh=None, binary=None, loo=None, woe=None, lp_cols=None, NoData=True):
    from enrich import replaceCVs
    from enrich import one_hot_encode
    from category_encoders import BinaryEncoder
    from category_encoders import OneHotEncoder
    from category_encoders import WOEEncoder
    from category_encoders import LeaveOneOutEncoder

    if NoData is False:
        if cat is not None | oh is not None:
            # translate associated columns' null, NaN, blank and 9 values to zero
            X = replaceCVs(X, cat + oh, [np.nan, 9, "", " "], 0)

    if oh is not None:
        if NoData:
            ec = OneHotEncoder(cols=oh, use_cat_names=True, return_df=True, handle_unknown='indicator', handle_missing='indicator').fit(X)
            X = ec.fit_transform(X)
            # dropping these columns did not help performance
            # for o in oh:
            #    stem = o.split("_")[1]
            #    d1 = "L_" + stem + "_-1"
            #    d2 = "L_" + stem + "_nan"
            #    print("DROPPING ", d1, " ", d2, "\n")
            #    X.drop(d1, axis=1, errors='ignore', inplace=True)
            #    X.drop(d2, axis=1, errors='ignore', inplace=True)
        else:
            # one-hot encode, then drop 0 if created
            for oh_c in oh:
                X = one_hot_encode(X, oh_c, False)
                X.drop(0, axis=1, errors='ignore', inplace=True)

    if binary is not None:
        # binary encode binary columns
        if NoData:
            enc = BinaryEncoder(cols=binary, drop_invariant=True, return_df=True, handle_unknown='indicator').fit(X)
            X = enc.transform(X)
        else:
            enc = BinaryEncoder(cols=binary, drop_invariant=True, return_df=True).fit(X)
            X = enc.transform(X)

    if woe is not None:
        # use weight of evidence on woe columns
        for w in woe:
            X[w] = X[w].fillna('NoData')

        wenc = WOEEncoder(cols=woe).fit(X, y)
        X = wenc.transform(X).round(2)

    if loo is not None:
        # use leave one out on loo columns
        for l in loo:
            X[l] = X[l].fillna('NoData')

        lenc = LeaveOneOutEncoder(cols=loo, return_df=True).fit(X, y)
        X = lenc.transform(X).round(2)


    # Cast all to int64
    # X = X.astype("int64")

    if lp_cols is not None:
        # drop least predictive
        X.drop(lp_cols, axis=1, errors="ignore", inplace=True)

    X.reset_index(drop=True, inplace=True)
    return X


# Binary classification balancer
#
def rebalanceSample(df, col, majority_val, minority_val, minority_percent, seed):

    # Separate majority and minority classes
    df_majority = df[df[col] == minority_val]
    df_minority = df[df[col] == majority_val]

    # Re-sample minority class
    total = df[col].count()
    n = int(total * minority_percent)
    df_minority_resampled = sku.resample(df_minority,
                                         replace=True,  # sample with replacement
                                         n_samples=n,
                                         random_state=seed)

    # Re-sample the majority class in a complementary manner
    n2 = total - n
    df_majority_resampled = sku.resample(df_majority,
                                         replace=True,  # sample with replacement
                                         n_samples=n2,
                                         random_state=seed)

    df_resampled = pd.concat([df_majority_resampled, df_minority_resampled])

    # Display new class counts
    df_resampled.reset_index(drop=True, inplace=True)
    return df_resampled


# multiclass balancer
#
def rebalanceMulticlass(df, col, num_classes, class_values, seed):
    # separate class values
    c = {}
    for i in range(num_classes):
        c[i] = df[df[col] == class_values[i]]

    # print(c)
    # Re-sample classes to be equal in size
    total = df[col].count()
    ppc = 1 / num_classes
    n = int(total * ppc)
    dfs = []
    for i in range(num_classes):
        df = sku.resample(c[i],
                          replace=True,  # sample with replacement
                          n_samples=n,
                          random_state=seed)
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)
    final_df.reset_index(drop=True, inplace=True)

    # Display new class counts
    # print(final_df[col].value_counts())
    return final_df
