# Hands-On Machine Learning
# Chapter 2
# Housing Database: Regression

import utils
import enrich
import learn
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.impute as ski
import sklearn.preprocessing as skp


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
HOUSING_CSV = "housing.csv"

utils.fetch_data(HOUSING_URL, HOUSING_PATH)
housing_df = utils.load_data(HOUSING_PATH, HOUSING_CSV)
print(housing_df.head())

# Show data about the DataFrame, including number of rows with data for each column
print("All Data: Info & Filled Values: ")
print(housing_df.info())

# Categorical distribution table
print("Categorical Distribution Table: ocean_proximity")
print(housing_df["ocean_proximity"].value_counts())

# Numerical attributes summary
print("All Data: Numeric Attributes Summary: ")
print(housing_df.describe())

# Histograms
housing_df.hist(bins=50, figsize=(20, 15))
plt.title("All Data: Distribution Histograms")
plt.show()

# Create categorical variable based on median income for Stratification Sampling
enrich.add_categorical_from_continuous(housing_df, "median_income", "income_cat", [0., 1.5, 3.0, 4.5, 6, np.inf],
                                       [1, 2, 3, 4, 5])

# Sample training v. testing data by strata
train_housing, test_housing = enrich.split_train_test_strat(housing_df, "income_cat", 1, 0.2, 42)
# Remove strata, as it is no longer needed
train_housing = train_housing.drop("income_cat", axis=1)
test_housing = test_housing.drop("income_cat", axis=1)

# Create a copy of the training data to work with
hds = train_housing.drop("median_house_value", axis=1)
hds_labels = train_housing["median_house_value"].copy()

# Visualize scatterplot with high-density areas
hds.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.title("Training Data: Scatterplot")
plt.show()

# Visualize prices
hds.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=hds["population"]/100, label="Population",
         figsize=(10,7), c=hds_labels, cmap=plt.get_cmap("jet"), colorbar=True)
plt.title("Training Data: Median Housing Prices")
plt.legend()
plt.show()

# Create a starting correlation matrix
corr_matrix = train_housing.corr()
print("Training Data Correlation Matrix: median_house_value")
print(corr_matrix["median_house_value"].sort_values(ascending=False))
print("\n")

# Add Attributes
haa = learn.HousingAttributesAdder()
haa.transform(hds)

# One Hot Encode Categorical Variable
one_hot = pd.get_dummies(hds['ocean_proximity'])
hds = hds.drop('ocean_proximity', axis=1)
hds = hds.join(one_hot)

# Impute NULL Values
imp = ski.SimpleImputer(strategy="median")
hds = pd.DataFrame(imp.fit_transform(hds), columns=hds.columns, index=hds.index)

# Standardize
scale = skp.StandardScaler()
hds = pd.DataFrame(scale.fit_transform(hds), columns=hds.columns, index=hds.index)

# Do everything you just did on the test set to keep them congruent but DON'T CALL fit_transform, use transform
hds_test = test_housing.drop("median_house_value", axis=1)
hds_test_labels = test_housing["median_house_value"].copy()
haa.transform(hds_test)
one_hot2 = pd.get_dummies(hds_test["ocean_proximity"])
hds_test = hds_test.drop("ocean_proximity", axis=1)
hds_test = hds_test.join(one_hot2)
hds_test = pd.DataFrame(imp.transform(hds_test), columns=hds_test.columns, index=hds_test.index)
hds_test = pd.DataFrame(scale.transform(hds_test), columns=hds_test.columns, index=hds_test.index)


# Attributes List: Tweaked after Exploration with Random Forest Regression
attributes = ["median_income", "INLAND", "population_per_household", "bedrooms_per_room", "rooms_per_household", "housing_median_age"]


# SLR - Try to use the most promising attributes to predict median_house_value
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
hds_subset = pd.DataFrame(hds, columns=attributes, index=hds.index)
lin_reg.fit(hds_subset, hds_labels)
print("Slope Coefficients: ", lin_reg.coef_)
print("\nIntercept Value: ", lin_reg.intercept_)

# Try to predict median_house_value using the test set
test_subset = pd.DataFrame(hds_test, columns=attributes, index=hds_test.index)

print("Predictions: ", lin_reg.predict(test_subset))
print("Reality: ", hds_test_labels)
print("\n")

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(test_subset)
lin_mse = mean_squared_error(hds_test_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Root Mean Square Error: ")
print(lin_rmse)
print("\n")

from sklearn.tree import DecisionTreeRegressor
tr = DecisionTreeRegressor()
tr.fit(hds_subset, hds_labels)

housing_predictions2 = tr.predict(test_subset)
print("Decision Tree Regressor: ")
print("Predictions: ", housing_predictions2)
tree_mse = mean_squared_error(hds_test_labels, housing_predictions2)
tree_rmse = np.sqrt(tree_mse)
print("Root Mean Square Error: ")
print(tree_rmse)
print("\n")

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tr, test_subset, hds_test_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print("Scores: ", tree_rmse_scores)
print("Mean: ", tree_rmse_scores.mean())
print("Standard deviation: ", tree_rmse_scores.std())

# Try Random Forest Regression on all of the items in the DataFrame
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(hds, hds_labels)
housing_predictions3 = forest_reg.predict(hds_test)
print("Random Forest Regressor: ")
print("Predictions: ", housing_predictions3)
print(hds_test_labels)
rf_mse = mean_squared_error(hds_test_labels, housing_predictions3)
rf_rmse = np.sqrt(rf_mse)
print("Root Mean Square Error: ")
print(rf_rmse)
print("\n")

cv_score = cross_val_score(forest_reg, hds_test, hds_test_labels, scoring="neg_mean_squared_error", cv=10)
rf_rmse_scores = np.sqrt(-cv_score)
print("Scores: ", rf_rmse_scores)
print("Mean: ", rf_rmse_scores.mean())
print("Standard deviation: ", rf_rmse_scores.std())

# Tune Random Forest Regressor with GridSearchCV - Takes a while
from sklearn.model_selection import GridSearchCV
param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
              {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]
fr2 = RandomForestRegressor()
grid_search = GridSearchCV(fr2, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(hds, hds_labels)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

feature_importances = grid_search.best_estimator_.feature_importances_
print(sorted(zip(feature_importances, hds.columns), reverse=True))

final_model = grid_search.best_estimator_
final_predictions = final_model.predict(hds_test)
final_mse = mean_squared_error(hds_test_labels, final_predictions)
final_rmse = np.sqrt(final_mse)
print("Final: ", final_predictions)
print("Final RMSE: ", final_rmse)