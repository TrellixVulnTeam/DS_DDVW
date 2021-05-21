import utils
import pandas as pd

SSA2FIPS = utils.load_data("data", "ssa_fips_state_county2011_augmented.csv")
BRFSS = utils.load_data("data", "BRFSSCountyAggregates2010.csv")
PUF = utils.load_data("data/PUF", "PUF_COUNTY_SUMMARY.csv")

PUF_FIPS = pd.merge(PUF, SSA2FIPS, on=["FIPS_STATE_CODE", "FIPS_COUNTY_CODE"], how='left')
PUF_BRFSS = pd.merge(PUF_FIPS, BRFSS, left_on=["FIPS_STATE_CODE", "FIPS_COUNTY_CODE"], right_on=["StateCode", "CountyFIPS"], how='left')
PUF_BRFSS.to_csv("data/PUF/PUF_BRFSS_MERGED.csv")