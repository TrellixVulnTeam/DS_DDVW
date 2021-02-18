# Multiclass Classifier Pipeline Script
# Compare all the models and run the best one on the large data set
#

import manip
import enrich
import importlib
import pandas as pd

# classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# accuracy scoring, and data partitioning
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def getInstance(module, classname, params):
    my_module = importlib.import_module(module)
    my_class = getattr(my_module, classname)
    if params is not None:
        instance = my_class(**params)
    else:
        instance = my_class()
    print("class instanced:", instance)
    return instance


# Classifiers we'd like to compare
classifiers = {
    "MLPClassifier": ["sklearn.neural_network", "MLPClassifier", {"alpha":1e-5,
                                                                  "solver":"lbfgs",
                                                                  "hidden_layer_sizes": (15,),}],
    "Gaussian Naive Bayes": ["sklearn.naive_bayes", "GaussianNB", None],
    "K-Nearest Neighbors": ["sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors": 3}],
    "Random Forest": ["sklearn.ensemble", "RandomForestClassifier", None],
    "Gradient Boosting": ["sklearn.ensemble", "GradientBoostingClassifier", None],
    "XGBoost": ["xgboost", "XGBClassifier", {"use_label_encoder": False}]
}

# define columns to use
binary_columns = ["B_SMOKER", "B_HLTHPLN", "B_COUPLED", "B_VEGGIE", "B_EXERCISE", "B_EXER30", "B_SLEEPOK", "B_BINGER",
                  "B_CHECKUP", "B_GOODHLTH", "B_POORHLTH", "B_SEATBLT", "B_HIVRISK"]
label_columns = ["L_SEX", "L_AGE_G", "L_EMPLOY1", "L_INCOMG", "L_EDUCAG", "L_BMI5CAT", "L_IMPRACE"]
labelled_states = ["L_STATEAB"]
target = ["CHRONICGRP"]

print("MULTICLASS CLASSIFICATION: BRFSS DATASET 2017-2019 \n")
print("OUTCOMES: NO CHRONIC CONDITIONS (0), ONE CHRONIC CONDITION (1), TWO CHRONIC CONDITIONS (2), THREE OR MORE (3)\n\n")

# read in the weighted 50,000 row preliminary sample
raw = pd.read_csv("BRFSS_SAMPLE_WEIGHTED.csv")
brfss_small = pd.DataFrame(raw, columns=binary_columns + label_columns + labelled_states + target)
print("Preliminary 50,000 row sub-set:\n")
print(brfss_small.info())
print("CHRONICGRP Value Counts:\n")
print(brfss_small["CHRONICGRP"].value_counts())
print("\n")
brfss_small = manip.rebalanceMulticlass(brfss_small, "CHRONICGRP", 4, [0, 1, 2, 3], 123)
print("After Balancing:\n")
print(brfss_small["CHRONICGRP"].value_counts())
print("\n")

X, Y = enrich.prepareXY(brfss_small, "CHRONICGRP")


# Do categorical encoding
print("One-hot encoding all categorical columns.\n")
X = manip.doCleanupEncode(X, binary=label_columns + labelled_states)
print("FINAL preliminary dataset:\n")
print(X.info())

# train and check the model against the test data for each classifier
iterations = 10
results = {}
for itr in range(iterations):

    # Reshuffle training/testing datasets by sampling randomly 80/20% of the input data
    print("Shuffling training/testing datasets...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    for classifier in classifiers:
        clf = getInstance(classifiers[classifier][0], classifiers[classifier][1], classifiers[classifier][2])
        print("Fitting model...")
        clf.fit(X_train, Y_train)

        # predict on test data
        prediction = clf.predict(X_test)

        # compute accuracy and sum it to the previous ones
        accuracy = accuracy_score(Y_test, prediction)
        cm = confusion_matrix(Y_test, prediction)

        if classifier in results:
            results[classifier] = results[classifier] + accuracy
        else:
            results[classifier] = accuracy

        print(itr, ": Classifier: ", classifier, "\nAccuracy: ", accuracy, "\nConfusion Matrix:\n", cm, "\n\n")


print("Classifiers average scores over", iterations, "iterations are:")
halloffame = [(v, k) for k, v in results.items()]
halloffame.sort(reverse=True)
for v, k in halloffame:
    print("\t-", k, "with", v / iterations * 100, "% accuracy")

# using the best to train the full dataset
raw = pd.read_csv("BRFSS_Clean_Combo.csv")
brfss = pd.DataFrame(raw, columns=binary_columns + label_columns + labelled_states + target)
print("CHRONICGRP Value Counts:\n")
print(brfss["CHRONICGRP"].value_counts())
print("\n")
brfss = manip.rebalanceMulticlass(brfss, "CHRONICGRP", 4, [0, 1, 2, 3], 123)
print("After Balancing:\n")
print(brfss["CHRONICGRP"].value_counts())
print("\n")
X, Y = enrich.prepareXY(brfss, "CHRONICGRP")
X = manip.doCleanupEncode(X, binary=label_columns + labelled_states)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


print("Retraining the Full Data-Set with the Top 2 Classifiers:", halloffame[0], ", ", halloffame[1])
print("\n")
bestclf1 = getInstance(classifiers[halloffame[0][1]][0], classifiers[halloffame[0][1]][1],
                      classifiers[halloffame[0][1]][2])
print("FITTING...")
bestclf1.fit(X_train, Y_train)
print("PREDICTING...")
prediction = bestclf1.predict(X_test)
print("SCORES:")
accuracy = accuracy_score(Y_test, prediction)
cm = confusion_matrix(Y_test, prediction)

print(bestclf1, "Accuracy: ", accuracy, "\n")
print(bestclf1, "Confusion Matrix:\n", accuracy, "\n")

bestclf2 = getInstance(classifiers[halloffame[1][1]][0], classifiers[halloffame[1][1]][1],
                      classifiers[halloffame[1][1]][2])
print("FITTING...")
bestclf2.fit(X_train, Y_train)
print("PREDICTING...")
prediction = bestclf2.predict(X_test)
print("CRUDE SCORES (pre-tuning):")
accuracy = accuracy_score(Y_test, prediction)
cm = confusion_matrix(Y_test, prediction)

print(bestclf2, "Accuracy: ", accuracy, "\n")
print(bestclf1, "Confusion Matrix:\n", accuracy, "\n")

print("CLASSIFICATION COMPLETE.")
