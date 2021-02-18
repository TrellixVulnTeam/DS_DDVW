#
# K. Rasku RN BSN & Programmer / Analyst
# My python data science library
# Initiated 1/4/2021
# Last Updated 1/29/2021
#
# UTILS.py
#
#### dummy decorator & block operator
#### large file reader
#### loading data via url, tarfile
#### scraping your own data
#### count words, lines or characters
#### timer for functions

import os
import time
import tarfile
import pandas as pd
import urllib.request as ur
import sklearn.datasets as data


# dummy decorator
# allows execution of any function passed to it, func
#
def do_func(func):
    def wrapper(a):
        return func(a)

    return wrapper


# read_any
# generator - read any-sized file, fhandle in blocks
# default size block = 10,000
#
def read_any(fhandle, size=10000):
    block = []
    for i in fhandle:
        block.append(i)
        if len(block) == size:
            yield block
            block = []

    if block:
        yield block


# block_operator
# open and read any-sized file located at path
# and calls an operational decorator to execute block_op_function
# on the file block
# default block size = 10,000
#
def block_operator(path, block_op_function, size=10000):
    with open(path) as lfile:
        for b in read_any(lfile, size):
            # do something, for example print
            block_op_function(b)


def fetch_data(url, path):
    os.makedirs(path, exist_ok=True)
    url_list = url.split("/")
    get_file_idx = len(url_list)-1
    file_root_name = url_list[get_file_idx].split(".")[0]
    csv_file_name = file_root_name + ".csv"
    tgz_path = os.path.join(path, url_list[get_file_idx])
    if not os.path.isfile(os.path.join(path, csv_file_name)):
        ur.urlretrieve(url, tgz_path)
        with tarfile.open(tgz_path) as tf:
            tf.extractall(path)


def load_data(path, csv_file):
    return pd.read_csv(os.path.join(path, csv_file))


def fetch_sklearn(dataset, version):
    return data.fetch_openml(dataset, version=version)

# scrape a list of fields and their values off a web page
# or a series of web pages
# creates: a csv file
# returns: the name of the csv file written, a string
#
# def scrape_a_set(url_list, field_list):


# count
# counts file-related things: all, words or lines for a file f
# returns: an integer
#
def count(f, count_type=all):
    fh = open(f, "r")
    file_data = fh.read()
    if count_type == "word":
        return len(file_data.split(" "))

    elif count_type == "line":
        return len(file_data.split("\n"))

    else:
        # entire document / all characters
        return len(file_data)


# count_missing
# counts missing values in a dataframe (df), or counts only values of the columns requested by (cols)
# if orientation is 0 (default) compute counts col-wise, otherwise compute them row-wise (orientation=1)
# returns: Series
# requires: pandas
#
def count_missing(df, cols=[], orientation=0):
    if len(cols) > 0:
        # subset first
        df = pd.DataFrame(df, cols)

    # count everything
    if orientation:
        return df.isna().sum(axis=1)
    else:
        return df.isna().sum()


# percent_missing
# produces a dataframe of the percentage of missing values for each column of a dataframe
# returns: Series
# requires: pandas
#
def percent_missing(df):
    return df.isna().mean().round(4) * 100


# Author: Bex Tuychiev
def timer(func):
    """
    A decorator to calculate how long a function runs.

    Parameters
    ----------
    func: callable
      The function being decorated.

    Returns
    -------
    func: callable
      The decorated function.
    """

    def wrapper(*args, **kwargs):
        # Start the timer
        start = time.time()
        # Call the `func`
        result = func(*args, **kwargs)
        # End the timer
        end = time.time()

        print(f"{func.__name__} took {round(end - start, 4)} "
              "seconds to run!")
        return result

    return wrapper

