# Using Ensemble Modeling with Random Forest Regression on an Unknown (Toy) Data Set using Scikit Learn
# Assignment 2 - HDS 805
# KPR 3/5/2021
#

import time
import enrich
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
import sklearn.pipeline as pipe
import sklearn.preprocessing as skp
import sklearn.compose as compose
import sklearn.ensemble as ens
import sklearn.linear_model as lm
import sklearn.model_selection as ms


import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)


# Scikit Learn Function for Displaying Results from Stacking Model
def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    title = title + '\n Evaluation in {:.2f} seconds'.format(elapsed_time)
    ax.set_title(title)


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

    # Stacking, using previously-tuned Regression models and Random Forest Regressor
    estimators = [('Random Forest', ens.RandomForestRegressor(max_features=None, max_leaf_nodes=100,
                                                              n_estimators=300, min_samples_leaf=10,
                                                              max_depth=15, min_samples_split=5,
                                                              random_state=123, n_jobs=-1, verbose=1)),
                  ('Lasso', lm.LassoCV(random_state=123, fit_intercept=False))]

    stacking_regressor = ens.StackingRegressor(estimators=estimators,
                                               final_estimator=lm.RidgeCV(fit_intercept=False))


    stacking_regressor.fit(x_train, y_train)
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    axs = np.ravel(axs)

    for ax, (name, est) in zip(axs, estimators + [('Stacking Regressor', stacking_regressor)]):
        start_time = time.time()
        score = ms.cross_validate(est, x_train, y_train, scoring=['r2', 'neg_mean_absolute_error'], n_jobs=-1, verbose=0)
        elapsed_time = time.time() - start_time

        y_pred = ms.cross_val_predict(est, x_train, y_train, n_jobs=-1, verbose=0)

        plot_regression_results(
            ax, y_train, y_pred,
            name,
            (r'$R^2={:.2f} \pm {:.2f}$' + '\n' + r'$MAE={:.2f} \pm {:.2f}$')
                .format(np.mean(score['test_r2']),
                        np.std(score['test_r2']),
                        -np.mean(score['test_neg_mean_absolute_error']),
                        np.std(score['test_neg_mean_absolute_error'])),
            elapsed_time)

    plt.suptitle('Single predictors versus stacked predictors')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()





