#
# Testing calls to my libraries
# K. Rasku RN BSN  Last Modified: 1/24/2021
#

import utils
import manip
import enrich

import numpy as np
import pandas as pd


utils.block_operator("data/SampleTextFile_1000kb.txt",utils.do_func(print))

print("\n")

print("Counting words in file:")
print("Words: ")
print(str(utils.count("data/SampleTextFile_1000kb.txt", "word")))
print("Lines: ")
print(str(utils.count("data/SampleTextFile_1000kb.txt", "line")))
print("All Characters in document: ")
print(str(utils.count("data/SampleTextFile_1000kb.txt")))

print("\n")
print("Use a regex to count the number of white spaces in the file:")
print(str(manip.count_matches("data/SampleTextFile_1000kb.txt", r'\s')))
print("Count the number of times 'er' appears at the end of a word:")
print(str(manip.count_matches("data/SampleTextFile_1000kb.txt", r'er\b')))

print("\n")
print("Substitute all instances of 'er' at a word-boundary with 'as'")
manip.sub_and_write("data/SampleTextFile_1000kb.txt", r'er\b', "as")

# NOTE: This is commented out because this implementation is not yet complete.
#
# scrape_fields_list = ['Best Bid/Ask', '1 Year Target', 'Share Volume', '50 Day Avg. Daily Volume', 'Previous Close', '52 Week High/Low', 'Market Cap', 'P/E Ratio', 'Forward P/E (1y)', 'Earnings Per Share (EPS)', 'Annualized Dividend', 'Ex-Dividend Date', 'Dividend Payment Date', 'Current Yield', 'Beta Open Price', 'Open Date', 'Close Price', 'Close Date']
# scrape_urls_list = ['http://www.nasdaq.com/symbol/aapl']
# scraped_file = utils.scrape_a_set(scrape_urls_list, scrape_fields_list)
# print("The data I scraped is in " + str(scraped_file))
# print("\n")

print("\n")


