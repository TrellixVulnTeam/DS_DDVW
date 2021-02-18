# Hands-On Machine Learning
# Chapter 2
# housing2 - Transforming the Housing Dataset Using Pipeline Transformers

import os
import enrich
import learn
import utils
import numpy as np
import pandas as pd
import sklearn.compose as sk_compose
import sklearn.pipeline as sk_pipe
import sklearn.impute as sk_impute
import sklearn.preprocessing as skp


HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_CSV = "housing.csv"

# Assuming this is already present
housing_df = utils.load_data(HOUSING_PATH, HOUSING_CSV)

# Create categorical variable based on median income for Stratification Sampling
enrich.add_categorical_from_continuous(housing_df, "median_income", "income_cat", [0., 1.5, 3.0, 4.5, 6, np.inf],
                                       [1, 2, 3, 4, 5])

# Sample training v. testing data by strata
train_housing, test_housing = enrich.split_train_test_strat(housing_df, "income_cat", 1, 0.2, 42)
# Remove strata, as it is no longer needed
train_housing = train_housing.drop("income_cat", axis=1)
test_housing = test_housing.drop("income_cat", axis=1)

# Create a copy of the training data to work with
hds = train_housing.copy()
hds_labels = train_housing["median_house_value"].copy()

# Create a starting correlation matrix
corr_matrix = hds.corr()
print("Housing Data Training Subset Correlation Matrix: median_house_value")
print(corr_matrix["median_house_value"].sort_values(ascending=False))
print("\n")


# Create pipelines
numeric_transformer = sk_pipe.Pipeline(steps=[
    ('attribs_adder', learn.HousingAttributesAdder()),
    ('imputer', sk_impute.SimpleImputer(strategy='median')),
    ('scaler', skp.StandardScaler())])
categorical_transformer = sk_pipe.Pipeline(steps=[
    ('onehot', skp.OneHotEncoder())])

# Create full transformation, including both pipelines
full_transformer = sk_compose.ColumnTransformer([
    ("num", numeric_transformer, list(hds.drop("ocean_proximity", axis=1))),
    # NOTE: The column passed will be DROPPED by default after transformation.
    ("cat", categorical_transformer, ["ocean_proximity"])
])

# Prepare the data by fitting the full pipeline to copy of training data,
# and transforming it
# N.B. You must cast this back to DataFrame, because the return value is of type Numpy array
tcols_list = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
              'households', 'median_income', 'median_house_value', 'rooms_per_household', 'population_per_household',
              'bedrooms_per_room', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
hds_prepared = pd.DataFrame(full_transformer.fit_transform(hds), columns=tcols_list)


# Show the correlation matrix for the prepared data
corr_matrix = hds_prepared.corr()
print("Transformed / Prepared Housing Data Correlation Matrix: median_house_value")
print(corr_matrix["median_house_value"].sort_values(ascending=False))
print("\n")

