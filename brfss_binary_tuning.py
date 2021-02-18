# BINARY CLASSIFICATION - BRFSS
# Outcome variable: COMORB_1
# Candidate model: Random Forest
#
import manip
import enrich
import learn

import time
import warnings
import numpy as np
import pandas as pd

import sklearn.metrics as smet
import sklearn.model_selection as ms

from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv

np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings('ignore')

# define columns to use
binary_columns = ["B_SMOKER", "B_HLTHPLN", "B_COUPLED", "B_VEGGIE", "B_EXERCISE", "B_EXER30", "B_SLEEPOK", "B_BINGER",
                  "B_CHECKUP", "B_GOODHLTH", "B_POORHLTH", "B_SEATBLT", "B_HIVRISK"]
label_columns = ["L_SEX", "L_AGE_G", "L_EMPLOY1", "L_INCOMG", "L_EDUCAG", "L_BMI5CAT", "L_IMPRACE"]
labeled_states = ["L_STATEAB"]
target = ["COMORB_1"]

# read in the 300,000 row sample
raw = pd.read_csv("data/BRFSS_Medium.csv")
brfss_med = pd.DataFrame(raw, columns=binary_columns + label_columns + labeled_states + target)
brfss_med = manip.rebalanceSample(brfss_med, "COMORB_1", 0, 1, .5, 123)
X, Y = enrich.prepareXY(brfss_med, "COMORB_1")
X = manip.doCleanupEncode(X, binary=label_columns + labeled_states)
X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=0.2)

param_grid = {'n_estimators': [100, 200, 500],
              'min_samples_leaf': [10, 15, 20, 30],
              'criterion': ['gini', 'entropy'],
              'max_depth': [10, 15, 20],
              'min_samples_split': [10, 20],
              'n_jobs': [-1],
              'verbose': [1]}

start_time = time.time()

scores = ['accuracy', 'f1_macro', 'roc_auc_ovo']

for score in scores:
    print("Tuning for %s" % score)
    print("----------------------------------")

    rf_new = ms.HalvingRandomSearchCV(RandomForestClassifier(), param_grid, scoring='%s' % score)
    rf_new.fit(X_train, Y_train)

    print("Best parameters set found is:")
    print(rf_new.best_params_)

    print("Grid scores on training set:")

    means = rf_new.cv_results_['mean_test_score']
    stds = rf_new.cv_results_['std_test_score']

    print("Average scores are ", means)
    print("SD for the scores are ", stds)

    print("Detailed classification report:")
    y_true, y_pred = Y_test, rf_new.predict(X_test)
    print(smet.classification_report(y_true, y_pred))

    learn.displayCVResults(rf_new)

end_time = time.time()

print("---------------------------------")
print("total time taken = ", end_time - start_time)