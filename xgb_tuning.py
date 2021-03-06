# Using Ensemble Boosting Regression on an Unknown (Toy) Data Set using Scikit Learn
# Assignment 2 - HDS 805
# KPR 3/5/2021
#

import time
import enrich
import learn
import numpy as np
import pandas as pd
import xgboost as xgb
import category_encoders as ce
import sklearn.pipeline as pipe
import sklearn.preprocessing as skp
import sklearn.compose as compose
import sklearn.ensemble as ens
import sklearn.model_selection as ms
import sklearn.metrics as metrics
from sklearn.experimental import enable_halving_search_cv

import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)


# Begin Main
#
df = pd.read_csv("data/synthetic_data_regression.csv")

# Drop ID and Y3.  Y3 data did not have any significant linear or non-linear relationship with the other variables.
df.drop(["ID", "Y3"], axis=1, inplace=True)

# Define targets (Y1 and Y2), categorical and numeric features (Xs)
targets = ["Y1", "Y2"]
cat_features = ["CatVar0", "CatVar1", "CatVar2", "CatVar3", "CatVar4", "CatVar5", "CatVar6", "CatVar7"]
num_features = list(df.drop(cat_features + targets, axis=1).columns)

# The distribution of Y1 and Y2 are both normal. Split data randomly.
train, test = enrich.split_train_test_rand(df, 0.2, 123)

# Define pipelines
numeric_transformer = pipe.Pipeline(steps=[
    ('scaler', skp.StandardScaler())])

categorical_transformer = pipe.Pipeline(steps=[
    ('onehot', ce.OneHotEncoder(cols=cat_features, drop_invariant=True))])

# Create full transformation, including both pipelines
full_transformer = compose.ColumnTransformer([
    ("cat", categorical_transformer, cat_features),
    ("num", numeric_transformer, num_features)
])

# Prepare the data by fitting the full pipeline to the training data, and transforming it
# N.B. You must cast this back to DataFrame, because the return value is of type numpy array
oh_names = ["CatVar0_1", "CatVar0_2", "CatVar0_3", "CatVar0_4",
            "CatVar1_1", "CatVar1_2", "CatVar1_3", "CatVar1_4",
            "CatVar2_1", "CatVar2_2",
            "CatVar3_1", "CatVar3_2", "CatVar3_3",
            "CatVar4_1", "CatVar4_2", "CatVar4_3",
            "CatVar5_1", "CatVar5_2", "CatVar5_3", "CatVar5_4",
            "CatVar6_1", "CatVar6_2", "CatVar6_3", "CatVar6_4",
            "CatVar7_1", "CatVar7_2", "CatVar7_3"]

# drop targets prior to transformation with pipeline
x_train = train.drop(targets, axis=1)
x_test = test.drop(targets, axis=1)

# fit-transform x_train, and transform x_test
x_train_np = full_transformer.fit_transform(x_train)
x_test_np = full_transformer.transform(x_test)

# restore column titles / recast to data frame
x_train = pd.DataFrame(x_train_np, columns=num_features + oh_names)
x_test = pd.DataFrame(x_test_np, columns=num_features + oh_names)


for y in targets:
    print("The target is ", y, "\n")
    y_train = train[y].copy()
    y_test = test[y].copy()

    # XGBRegressor
    print("XGBRegressor: ")

    xg_params = {'reg_alpha': np.arange(3, 15, 1, dtype=int),
                 'reg_lambda': np.arange(0.000000001, 1, 0.05, dtype=float),
                 'n_estimators': np.arange(50, 200, 50, dtype=int)}

    start_time = time.time()

    xgsc = ms.HalvingGridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', booster='gblinear',
                                   eval_metric='rmse', n_jobs=-1, verbosity=3),
        param_grid=xg_params,
        cv=5,
        random_state=123,
        scoring='neg_mean_squared_error',
        aggressive_elimination=True,
        verbose=3,
        n_jobs=-1)


    xgsc.fit(x_train, y_train)
    y_true, y_pred = y_test, xgsc.predict(x_test)

    best_params = xgsc.best_params_
    print("Best Params: ", best_params)

    print("R-Squared: ", metrics.r2_score(y_true, y_pred))
    print("Mean Squared Error: ", metrics.mean_squared_error(y_true, y_pred))

    learn.displayCVResults(xgsc)

    end_time = time.time()

    print("---------------------------------")
    print("total time taken = ", end_time - start_time)