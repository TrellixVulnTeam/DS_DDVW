import utils
import manip
import numpy as np
import pandas as pd

# Stack DataFrames
big_df = manip.stack_dataframes(["data/LLCP2017.csv", "data/LLCP2018.csv", "data/LLCP2019.csv"], "YEAR", "[0-9]+")


# Combine columns using Regular Expressions
# Great regex debugger: https://regex101.com/
print("Combine Columns:")

ccol_exprs = ['ALCDAY[0-9]', 'AVEDRNK[0-9]', 'DIABAGE[0-9]', 'DRNKDRI[0-9]', 'FALLINJ[0-9]', 'SLEPTIM[0-9]',
              'HEIGHT[0-9]', 'NUMPHON[0-9]', 'WEIGHT[0-9]', 'CHKHEMO[0-9]', 'MARIJAN[0-9]', 'NUMBURN[0-9]',
              'HTIN[0-9]', 'HTM[0-9]', 'WTKG[0-9]', '_BMI[0-9]', 'ADDEPEV[0-9]', 'ASTHMA[0-9]', 'CHCCOPD[0-9]',
              'CHCKDNY[0-9]', 'CSTATE[0-9]', 'CTELENM[0-9]', 'CTELENUM[0-9]', 'CVDCRHD[0-9]', 'CVDINFR[0-9]',
              'DIABETE[0-9]', 'EXERANY[0-9]', 'FLUSHOT[0-9]', 'HADHYST[0-9]', 'HADPAP[0-9]', 'HAVARTH[0-9]',
              'HLTHPLN[0-9]', 'NUMHHOL[0-9]', 'PCPSAAD[0-9]', 'PCPSADI[0-9]', 'PCPSARE[0-9]', 'PNEUVAC[0-9]',
              'PSATEST[0-9]', 'PVTRESD[0-9]', 'STATERE[0-9]', 'STOPSMK[0-9]', 'VETERAN[0-9]', 'CAREGIV[0-9]',
              'CHECKUP[0-9]', 'RSPSTAT[0-9]', 'EMPLOY[0-9]', 'IMFVPLA[0-9|A-Z]', 'LASTDEN[0-9]', 'LASTPAP[0-9]',
              'LASTSMK[0-9]', 'PERSDOC[0-9]', 'RENTHOM[0-9]', 'RMVTETH[0-9]', 'SEX[0-9]', 'SMOKDAY[0-9]',
              'USENOW[0-9]', 'PREDIAB[0-9]', 'EYEEXAM[0-9]', 'HLTHCVR[0-9]', 'DELAYME[0-9]', 'CRGVREL[0-9]',
              'CRGVLNG[0-9]', 'CRGVHRS[0-9]', 'CRGVPRB[0-9]', 'USEMRJN[0-9]', 'RSNMRJN[0-9]', 'ADPLEAS[0-9]',
              'ADDOWN[0-9]', 'CNCRTYP[0-9]', 'CSRVTRT[0-9]', 'CSRVDOC[0-9]', '_PRACE[0-9]', '_MRACE[0-9]',
              '_SMOKER[0-9]', ' _RFSMOK[0-9]']

for c in ccol_exprs:
    manip.combine_by_re(big_df, c)

# create binary columns from non-binary
#
binary_cols = ["N_DIABETE", "N_ADDEPEV", "N_CHCCOPD", "N_CHCKDNY", "N_FLUSHOT", "N_HAVARTH", "N_NUMHHOL", "N_PNEUVAC",
               "N_PVTRESD"]
for c in binary_cols:
    nc = "B_" + c.split("_")[1]
    manip.create_binary(big_df, [c], [1], [2, 3, 4, 7, 9], nc)

manip.create_binary(big_df, ["_CASTHM1"], [2], [1, 7, 9], "B_ASTHMA")

# combine multiple non-binary columns into one binary column
#
cols_to_1 = ["CHCSCNCR", "CHCOCNCR"]
manip.create_binary(big_df, cols_to_1, [1], [2, 3, 4, 7, 9], "B_CANCER")
cols_to_2 = ["CVDCRHD4", "CVDINFR4", "CVDSTRK3"]
manip.create_binary(big_df, cols_to_2, [1], [2, 3, 4, 7, 9], "B_HEART")


# Create summary variable, TOTCHRONIC, containing the number of chronic conditions for each participant
#
big_df["TOTCHRONIC"] = big_df["B_ASTHMA"] + big_df["B_CANCER"] + big_df["B_CHCCOPD"] + big_df["B_ADDEPEV"] + big_df[
    "B_DIABETE"] + big_df["B_HEART"]


# Create summary variable, CHRONICGRP, with a coded categorical variable based on TOTCHRONIC
# 0 = No Chronic Conditions, 1 = One Chronic Condition, 2 = Two Chronic Conditions, 3 = 3 OR MORE Chronic Conditions
#
big_df["CHRONICGRP"] = big_df["TOTCHRONIC"]
if_cc = [big_df["TOTCHRONIC"] == 0, big_df["TOTCHRONIC"] == 1, big_df["TOTCHRONIC"] == 2, big_df["TOTCHRONIC"] >= 3]
then_cc = [0, 1, 2, 3]
big_df["CHRONICGRP"] = np.select(if_cc, then_cc)

# Create binary variable from summary variable
manip.create_binary(big_df, ["CHRONICGRP"], [1, 2, 3], [0], "COMORB_1")

# Combine "SEX" columns together
manip.combine_columns(big_df, ["SEX", "SEX1"], "_SEX1")
manip.combine_columns(big_df, ["_SEX", "_SEX1"], "_SEX2")
# Recode "SEX" to 1 = Male, 2 = Female or NaN
if_sex = [big_df["_SEX2"] == 1, big_df["_SEX2"] == 2, big_df["_SEX2"] > 2]
then_sex = [1, 2, np.nan]
big_df["SEX"] = np.select(if_sex, then_sex)

# Create binary summary variable, B_COUPLED, based on MARITAL
manip.create_binary(big_df, ["MARITAL"], [1, 6], [2, 3, 4, 5, 9], "B_COUPLED")

# Create binary summary variable, B_RACE, based on _IMPRACE
manip.create_binary(big_df, ["_IMPRACE"], [1], [2, 3, 4, 5, 6], "B_RACE", [1, 2])

# Lifestyle Variables
manip.create_binary(big_df, ["_FRTLT1A"], [1], [2, 3], "B_FRUIT")
manip.create_binary(big_df, ["_VEGLT1A"], [1], [2, 9], "B_VEGGIE")
manip.create_binary(big_df, ["_RFBING5"], [2], [1, 9], "B_BINGER")
manip.create_binary(big_df, ["CHECKUP1"], [1], [2, 3, 4, 7, 8, 9, np.nan], "B_CHECKUP")
manip.create_binary(big_df, ["_RFHLTH"], [1], [2, 9], "B_GOODHLTH")
manip.create_binary(big_df, ["HIVRISK5"], [1], [2, 7, 9, np.nan], "B_HIVRISK")
manip.create_binary(big_df, ["_SMOKER3"], [1, 2], [3, 4, 9], "B_SMOKER")
manip.create_binary(big_df, ["HLTHPLN1"], [1], [2, 3, 4, 7, 9], "B_HLTHPLN")

# Chose not to use EXERANY2, even though it was available for all 3 years
# Using EXEROFT1 and EXERHMM1 instead
if_ex = [big_df["EXEROFT1"] < 200, big_df["EXEROFT1"] >= 200]
then_ex = [1, 0]
big_df["B_EXERCISE"] = np.select(if_ex, then_ex)
manip.create_binary(big_df, ["B_EXERCISE"], [1], [0, np.nan], "B_EXERCISE")

big_df["B_EXER30"] = big_df["EXERHMM1"].copy()
if_ex2 = [big_df["B_EXER30"] >= 30, big_df["B_EXER30"] < 30]
then_ex2 = [1, 0]
big_df["B_EXER30"] = np.select(if_ex2, then_ex2)
manip.create_binary(big_df, ["B_EXER30"], [1], [0, 77, 99, np.nan], "B_EXER30")

big_df["B_SLEEPOK"] = big_df["SLEPTIM1"].copy()
if_sl = [big_df["SLEPTIM1"] > 6]
then_sl = [1]
big_df["B_SLEEPOK"] = np.select(if_sl, then_sl)
if_sl2 = [big_df["SLEPTIM1"] < 10]
then_sl2 = [1]
big_df["B_SLEEPOK"] = np.select(if_sl2, then_sl2)
manip.create_binary(big_df, ["B_SLEEPOK"], [1], [77, 99, np.nan], "B_SLEEPOK")

if_ph = [big_df["POORHLTH"] < 31]
then_ph = [1]
big_df["B_POORHLTH"] = np.select(if_ph, then_ph)
manip.create_binary(big_df, ["B_POORHLTH"], [1], [77, 88, 99, np.nan], "B_POORHLTH")

if_sb = [big_df["SEATBELT"] < 3, big_df["SEATBELT"] >= 3]
then_sb = [1, 0]
big_df["B_SEATBLT"] = np.select(if_sb, then_sb)

# TRIM the DataFrame to a size that will allow for faster analysis on desired columns
# Don't bother to include binary condition variables, their definition is too similar to the two outcome variables,
# but keep TOTCHRONIC for visualizations (be sure to drop it before running models)

med_df = pd.DataFrame(big_df, columns=["ID", "YEAR", "IMONTH", "_STATE", "_AGE_G", "SEX", "_IMPRACE", "_INCOMG", "_EDUCAG",
                                       "EMPLOY1", "_BMI5CAT", "_LLCPWT2", "COMORB_1", "B_COUPLED", "B_SMOKER", "B_HLTHPLN",
                                       "B_FRUIT", "B_VEGGIE", "B_EXERCISE", "B_EXER30", "B_SLEEPOK", "B_BINGER", "B_CHECKUP",
                                       "B_GOODHLTH", "B_POORHLTH", "B_SEATBLT", "B_HIVRISK", "CHRONICGRP", "TOTCHRONIC"])

# Add "WEIGHT" as a copy of "_LLCPTW2"
med_df["WEIGHT"] = med_df["_LLCPWT2"].copy()

# Check Data Set for NULLs
print(med_df.info())
print("\n")

# count missing columns
#
print("Count of Missing Column Values:")
missing_cols_df = utils.count_missing(med_df)
print(missing_cols_df.head())
print("\n")

# count missing rows
#
print("Count of Missing Row Values:")
missing_rows_df = utils.count_missing(med_df, orientation=1)
print(missing_rows_df.head())
print("\n")

# percent missing for all columns
#
print("Percentage Missing for All Columns:")
percent_missing_df = utils.percent_missing(med_df)
print(percent_missing_df.head())
print("\n")

# Last: Create TEXT/ String LABEL "L_" columns for all columns in the new DataFrame

# Create new column with State Names
# states_dict = {1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California', 8: 'Colorado', 9: 'Connecticut',
#               10: 'Delaware', 11: 'District of Columbia', 12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho',
#               17: 'Illinois', 18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana', 23: 'Maine',
#               24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota', 28: 'Mississippi', 29: 'Missouri',
#               30: 'Montana', 31: 'Nebraska', 32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico',
#               36: 'New York', 37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon',
#               42: 'Pennsylvania', 44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota', 47: 'Tennessee',
#               48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia', 53: 'Washington', 54: 'West Virginia',
#              55: 'Wisconsin', 56: 'Wyoming', 66: 'Guam', 72: 'Puerto Rico', 78: 'Virgin Islands'}

# manip.recode_col(med_df, "_STATE", states_dict, "L_STATENM")


states_abbr_dict = {1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT', 10: 'DE', 11: 'DC', 12: 'FL',
                    13: 'GA', 15: 'HI', 16: 'ID', 17: 'IL', 18: 'IN', 19: 'IA', 20: 'KS', 21: 'KY', 22: 'LA', 23: 'ME',
                    24: 'MD', 25: 'MA', 26: 'MI', 27: 'MN', 28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV', 33: 'NH',
                    34: 'NJ', 35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH', 40: 'OK', 41: 'OR', 42: 'PA', 44: 'RI',
                    45: 'SC', 46: 'SD', 47: 'TN', 48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV',
                    55: 'WI', 56: 'WY', 66: 'GU', 72: 'PR', 78: 'VI'}

manip.recode_col(med_df, "_STATE", states_abbr_dict, "L_STATEAB")

# REGIONS - Created to increase the signal of states with high levels of chronic disease
# NNE (Northern New England): ME, NH & VT
# NE (Northeast): MA, CT, RI, NY & NJ
# MA (Mid-Atlantic): PA, DE, MD, DC & VA
# SE (Southeast): NC, SC, GA & FL
# EC (East-Central): WV, OH, IN, MI, KY, TN, AL, MS, LA, AR, MO & OK
# CC (Central-Central): WI, IL, MN, IA, ND, SD, NE, KS & TX
# WC (West-Central): WY, CO, NM, AZ & UT
# WW (West-West): NV, CA, AK & HI
# TT (Territories): GU, VI & PR
regions_abbr_dict = {1: 'EC', 2: 'WW', 4: 'WC', 5: 'EC', 6: 'WW', 8: 'WC', 9: 'NE', 10: 'MA', 11: 'MA', 12: 'SE',
                    13: 'SE', 15: 'WW', 16: 'PNW', 17: 'CC', 18: 'EC', 19: 'CC', 20: 'CC', 21: 'EC', 22: 'EC', 23: 'NNE',
                    24: 'MA', 25: 'NE', 26: 'EC', 27: 'CC', 28: 'EC', 29: 'EC', 30: 'PNW', 31: 'CC', 32: 'WW', 33: 'NNE',
                    34: 'NE', 35: 'WC', 36: 'NE', 37: 'SE', 38: 'CC', 39: 'EC', 40: 'EC', 41: 'PNW', 42: 'MA', 44: 'NE',
                    45: 'SE', 46: 'CC', 47: 'EC', 48: 'CC', 49: 'WC', 50: 'NNE', 51: 'MA', 53: 'PNW', 54: 'EC',
                    55: 'CC', 56: 'WC', 66: 'TT', 72: 'TT', 78: 'TT'}

manip.recode_col(med_df, "_STATE", regions_abbr_dict, "L_REGION")

# binary_dict = {0: 'No', 1: 'Yes'}
# binary_labels = ["B_ASTHMA", "B_CANCER", "B_CHCCOPD", "B_ADDEPEV", "B_DIABETE", "B_HEART", "B_SMOKER", "B_HLTHPLN", "B_COUPLED", "COMORB_1"]
# for b in binary_labels:
#    if b != "COMORB_1":
#        newcol_sfx = b.split("_")[1]
#    else:
#        newcol_sfx = "COMORB"
#    new_col = "L_" + newcol_sfx
#   manip.recode_col(med_df, b, binary_dict, new_col)

# Choose one - you cannot one-hot encode both of these variables because of column name conflicts.
#
# age_dict = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64",
#           10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"}
# manip.recode_col(med_df, "_AGEG5YR", age_dict, "L_AGEG5YR")

age_dict2 = {1: "18-24", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55-64", 6: "65+"}
manip.recode_col(med_df, "_AGE_G", age_dict2, "L_AGE_G")

sex_dict = {1: "MALE", 2: "FEMALE"}
manip.recode_col(med_df, "SEX", sex_dict, "L_SEX")

race_dict = {1: "WHITE", 2: "BLACK", 3: "ASIAN", 4: "NA/PI/AN", 5: "HISPANIC", 6: "OTHRACE"}
manip.recode_col(med_df, "_IMPRACE", race_dict, "L_IMPRACE")

employ_dict = {1: "EMPLOYED", 2: "SELF-EMPLOYED", 3: "OOW 1 yr+", 4: "OOW lt 1 yr", 5: "HOMEMAKER",
               6: "STUDENT", 7: "RETIRED", 8: "UNABLE"}
manip.recode_col(med_df, "EMPLOY1", employ_dict, "L_EMPLOY1")

income_dict = {1: "$15K", 2: "$15K-25K", 3: "$25K-35K", 4: "$35K-50K", 5: "$50K+"}
manip.recode_col(med_df, "_INCOMG", income_dict, "L_INCOMG")

educa_dict = {1: "lt HS", 2: "HS GRAD", 3: "SOME COLLEGE", 4: "COLLEGE GRAD"}
manip.recode_col(med_df, "_EDUCAG", educa_dict, "L_EDUCAG")

bmi_dict = {1: "UNDER WEIGHT", 2: "NORMAL WEIGHT", 3: "OVER WEIGHT", 4: "OBESE"}
manip.recode_col(med_df, "_BMI5CAT", bmi_dict, "L_BMI5CAT")

# create report CSV(s)
#
missing_cols_df.to_csv("data/Missing_Columns_Count.csv")
percent_missing_df.to_csv("data/Missing_Columns_Percent.csv")

# create CSV of clean DataFrame
#
med_df.to_csv("data/BRFSS_Clean_Combo.csv")

# create a SAMPLE of 50000 random rows
#
smaller = med_df.sample(n=50000)
smaller.to_csv("data/BRFSS_SAMPLE.csv")

# create a SAMPLE of 300000 random rows
#
medium = med_df.sample(n=300000)
medium.to_csv("data/BRFSS_Medium.csv")

# create a WEIGHTED SAMPLE, also
#
sm_weighted = med_df.sample(n=50000, weights=med_df['WEIGHT'])
sm_weighted.to_csv("data/BRFSS_SAMPLE_WEIGHTED.csv")

