# Conducting OLS and Regularized Regression on an Unknown (Toy) Data Set using Scikit Learn
# Assignment 2 - HDS 805
# KPR 3/3/2021
#

import enrich
import learn
import numpy as np
import pandas as pd
import category_encoders as ce
import sklearn.linear_model as lm
import sklearn.pipeline as pipe
import sklearn.preprocessing as skp
import sklearn.metrics as metrics
import sklearn.compose as compose

import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)


# A function to get the best alpha and l1 ratio for Elastic Net regression
def getParametersElasticNet(X, y, target, k_folds=10):
    if target == "Y1":
        enCV = lm.ElasticNetCV(max_iter=100000,
                           l1_ratio=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
                           alphas=np.logspace(-4, -2, 9),
                           cv=k_folds,
                           n_jobs=-1,
                           verbose=1,
                           fit_intercept=False,
                           )
    else:
        enCV =  compose.TransformedTargetRegressor(regressor=lm.ElasticNetCV(max_iter=100000,
                           l1_ratio=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
                           alphas=np.logspace(-4, -2, 9),
                           cv=k_folds,
                           n_jobs=-1,
                           verbose=1,
                           fit_intercept=False,
                           ), transformer=skp.StandardScaler())

    # train and get hyperparams
    enCV.fit(X, y)
    print("Best Params: ", enCV.get_params())
    if target == "Y1":
        return enCV.alpha_, enCV.l1_ratio_
    else:
        return enCV.regressor_.alpha_, enCV.regressor_.l1_ratio_


# Begin Main
#
df = pd.read_csv("data/synthetic_data_regression.csv")

# Drop ID and Y3.  Y3 data did not have any significant linear or non-linear relationship with the other variables.
df.drop(["ID", "Y3"], axis=1, inplace=True)

# Initialize an output DataFrame, to hold information about the models tuned and fitted.
output_df = pd.DataFrame(
    columns=["Target", "Model", "Lambda", "Alpha", "L1 Ratio", "Feature Reduction", "Training Score", "Testing Score",
             "Test MSE", "Test MAE"])

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

    # LinearRegression
    print("Linear Regression / OLS: ")
    lin_reg = lm.LinearRegression()
    _ = lin_reg.fit(x_train, y_train)
    preds = lin_reg.predict(x_test)

    output_df = learn.printScores(lin_reg.score(x_train, y_train),
                                  lin_reg.score(x_test, y_test),
                                  metrics.mean_squared_error(y_test, preds),
                                  metrics.mean_absolute_error(y_test, preds),
                                  {'Target': y, 'Model': 'OLS'},
                                  output_df)

    # RidgeCV and Ridge
    print("Ridge Regression: ")
    ridgecv = compose.TransformedTargetRegressor(regressor=lm.RidgeCV(alphas=np.arange(0.5, 100, 5), cv=10), transformer=skp.StandardScaler())
    _ = ridgecv.fit(x_train, y_train)
    print("The cross-validated lambda for Ridge is ", ridgecv.regressor_.alpha_)

    ridge = compose.TransformedTargetRegressor(regressor=lm.Ridge(alpha=ridgecv.regressor_.alpha_, fit_intercept=False), transformer=skp.StandardScaler())
    _ = ridge.fit(x_train, y_train)
    preds = ridge.predict(x_test)

    output_df = learn.printScores(ridge.score(x_train, y_train),
                                  ridge.score(x_test, y_test),
                                  metrics.mean_squared_error(y_test, preds),
                                  metrics.mean_absolute_error(y_test, preds),
                                  {'Target': y, 'Model': 'Ridge', 'Lambda': ridgecv.regressor_.alpha_},
                                  output_df)

    # LassoCV and Lasso
    print("LASSO Regression: ")
    if y == "Y1":
        # use MinMaxScaler on the target
        lassocv = compose.TransformedTargetRegressor(
            regressor=lm.LassoCV(alphas=np.arange(0.000000001, 1, 0.05), cv=10), transformer=skp.MinMaxScaler())
    else:
        # use Standard
        lassocv = compose.TransformedTargetRegressor(
            regressor=lm.LassoCV(alphas=np.arange(0.000000001, 1, 0.05), cv=10), transformer=skp.StandardScaler())

    _ = lassocv.fit(x_train, y_train)
    print("The cross-validated lambda for LASSO is ", lassocv.regressor_.alpha_)
    if y == "Y1":
        lasso = compose.TransformedTargetRegressor(regressor=lm.Lasso(alpha=lassocv.regressor_.alpha_, fit_intercept=False),
                                                   transformer=skp.MinMaxScaler())
    else:
        lasso = compose.TransformedTargetRegressor(regressor=lm.Lasso(alpha=lassocv.regressor_.alpha_, fit_intercept=False),
                                                   transformer=skp.StandardScaler())
    lasso.fit(x_train, y_train)
    preds = lasso.predict(x_test)

    # Output a graph of the features reduced, and get the number of features removed
    lcoefs = lasso.regressor_.coef_
    feat_red = learn.printLassoCoefs(lcoefs, x_train)

    output_df = learn.printScores(lasso.score(x_train, y_train),
                                  lasso.score(x_test, y_test),
                                  metrics.mean_squared_error(y_test, preds),
                                  metrics.mean_absolute_error(y_test, preds),
                                  {'Target': y, 'Model': 'LASSO', 'Lambda': lassocv.regressor_.alpha_, 'Feature Reduction': feat_red},
                                  output_df)


    # ElasticNetCV and ElasticNet
    print("Elastic Net Regression: ")

    # find the best alpha and l1 ratio: default cv=10 (uses ElasticNetCV)
    bestAlpha, bestRatio = getParametersElasticNet(x_train, y_train, y)
    print(f'The ideal alpha for Elastic Net is {bestAlpha} and the best ratio is {bestRatio}')

    if y == "Y1":
        elastic = lm.ElasticNet(alpha=bestAlpha, l1_ratio=bestRatio, fit_intercept=False)
    else:
        elastic = compose.TransformedTargetRegressor(
            regressor=lm.ElasticNet(alpha=bestAlpha, l1_ratio=bestRatio, fit_intercept=False),
            transformer=skp.StandardScaler())

    _ = elastic.fit(x_train, y_train)
    preds = elastic.predict(x_test)

    output_df = learn.printScores(elastic.score(x_train, y_train),
                                  elastic.score(x_test, y_test),
                                  metrics.mean_squared_error(y_test, preds),
                                  metrics.mean_absolute_error(y_test, preds),
                                  {'Target': y, 'Model': 'ElasticNet', 'Alpha': bestAlpha, 'L1 Ratio': bestRatio},
                                  output_df)

print(output_df.head(10))
# Write tuning and model report to file
output_df.to_csv("data/FinalRegressionComparison.csv")
