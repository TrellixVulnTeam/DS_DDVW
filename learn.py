#
# K. Rasku RN BSN & Programmer / Analyst
# My python data science library
# Initiated 1/29/2020
#
# LEARN.py
#
#### Model training & testing functions
#
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn.base as skbase
import sklearn.metrics as smet
import sklearn.pipeline as skpipe
import sklearn.preprocessing as skpre
import sklearn.linear_model as lm
import sklearn.model_selection as ms

# from Hands On ML Book
# HousingAttributesAdder
#
class HousingAttributesAdder(skbase.BaseEstimator, skbase.TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["rooms_per_household"] = X["total_rooms"] / X["households"]
        X["population_per_household"] = X["population"] / X["households"]
        if self.add_bedrooms_per_room:
            X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"]
        # print(X.columns)
        return X

# These next two plotting functions by Kevin Arvai github.com/arvkevi
#
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')


# NOTE: this is for a binary model
#
def plot_roc_curve(fpr, tpr, label=None):

    plt.figure(figsize=(8,8))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')


# Plot data columns to see distribution before scaling
#
def plot_for_scale_type(df, cols):
    for c in cols:
        plt.title = "Before scaling: "
        sb.kdeplot(df[c])
        plt.show()


# Train and test a Logistic Regression model for a large data set
# using SGDClassifier, loss="log"
#
def initTrainLargeLogistic(x, y, test_x, test_y):

    # NOTE: Always examine and scale data prior to LR
    # Attempting to use SGDClassifier for Logistic Regression on large dataset
    # Preliminary model - increase max_iter, allow early_stopping for performance
    logModel = lm.SGDClassifier(loss="log", penalty="l2", max_iter=5000, early_stopping=True)

    # Cross Validate using 10 iterations
    scores = ms.cross_val_score(logModel, x, y, cv = 10)
    print("Initial accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Evaluate hyperparameters (do not change loss type)
    params = {
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "penalty": ["l2", "l1", "elasticnet", "none"],
        "max_iter": [2500, 5000],
        "early_stopping": [True]
    }

    logModel2 = lm.SGDClassifier(loss="log")
    grid = ms.GridSearchCV(logModel2, param_grid=params, cv=10)
    grid.fit(x, y)
    print("Params: ")
    print(grid.best_params_)


# Show all Model metrics # And plot PR and ROC curves
# Only works on models with 'decision_function';
# does not work for Random Forest Classifier.
#
#
def showMetrics(model, yTest, yPredicted, X=None, y=None, cv=None):

    # Confusion Matrix
    cm = smet.confusion_matrix(yTest, yPredicted)
    print("Confusion Matrix: ")
    print("Row 1 - true negatives, false positives")
    print("Row 2 - false negatives, true positives")
    print(cm)
    print("\n")

    # Classification Report - Precision, Recall, F1 Score, Support
    #
    # Precision (TPR) = true positives / (true positives + false positives)
    # Recall (Sensitivity) = true positives / (true positives + false negatives)
    # F1 Score: harmonic mean of precision and recall
    # Obviously favors classifiers with similar precision & recall (not always the goal)
    # Increasing precision reduced recall (precision / recall trade-off)
    # F1 = (2 / (1/precision + 1/recall)) = tp / tp + ((fn + fp)/ 2)
    #
    cr = smet.classification_report(yTest, yPredicted)
    print(cr)
    print("\n")

    # Full Decision Function Analysis for model
    y_scores = yPredicted
    precisions = []
    recalls = []
    thresholds = []
    if cv is not None:
        # do cross validation, re-define yscores
        y_scores =ms.cross_val_predict(model, X, y, cv=cv, method="decision_function")
        precisions, recalls, thresholds = smet.precision_recall_curve(y, y_scores)
    else:
        precisions, recalls, thresholds = smet.precision_recall_curve(yTest, y_scores)

    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.show()

    # The ROC: sensitivity (recall) vs. 1 - specificity (TNR)
    fpr = []
    tpr = []
    thresholds = []
    if cv is not None:
        #use cross-validated predictions
        fpr, tpr, thresholds = smet.roc_curve(y, y_scores)
    else:
        fpr, tpr, thresholds = smet.roc_curve(yTest, y_scores)

    plot_roc_curve(fpr, tpr)
    plt.show()

    # Compute AUC
    if cv is not None:
        auc_score = smet.roc_auc_score(y, y_scores)
    else:
        auc_score = smet.roc_auc_score(yTest, y_scores)

    print("AUC Score: ")
    print(auc_score)
    print("\n")


def showRFFeatures(rfModel):
    # random forest for feature importance on a regression problem
    from sklearn.ensemble import RandomForestRegressor
    from matplotlib import pyplot

    # get importance
    importance = rfModel.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


# Use predict_proba to show ROC curves for multiclass Random Forest model
#
def multiclassROC(fittedModel, X_test, Y_test, classes_list, classes_names):
    from sklearn.preprocessing import label_binarize
    y_score = fittedModel.predict_proba(X_test)
    y_test_bin = label_binarize(Y_test, classes=classes_list)
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = smet.roc_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(fpr[i], tpr[i], color='darkorange', lw=2)
        print('AUC for Class {}, {}: {}'.format(i + 1,classes_names[i], smet.auc(fpr[i], tpr[i])))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves')
    plt.show()


def displayCVResults(searchObject):
    results = pd.DataFrame(searchObject.cv_results_)
    results['params_str'] = results.params.apply(str)
    results.drop_duplicates(subset=('params_str', 'iter'), inplace=True)
    mean_scores = results.pivot(index='iter', columns='params_str',
                                values='mean_test_score')
    ax = mean_scores.plot(legend=False, alpha=.6)

    labels = [
        f'iter={i}\nn_samples={searchObject.n_resources_[i]}\n'
        f'n_candidates={searchObject.n_candidates_[i]}'
        for i in range(searchObject.n_iterations_)
    ]

    ax.set_xticks(range(searchObject.n_iterations_))
    ax.set_xticklabels(labels, rotation=45, multialignment='left')
    ax.set_title('Scores of candidates over iterations')
    ax.set_ylabel('mean test score', fontsize=15)
    ax.set_xlabel('iterations', fontsize=15)
    plt.tight_layout()
    plt.show()


def printScores(trainR2, testR2, testMSE, testMAE, params, output_df):

    print("Training score:", trainR2)
    print("Testing score:", testR2)
    print("Test MSE:", testMSE)
    print("Test MAE:", testMAE, '\n')

    scores = {'Training Score': trainR2, 'Testing Score': testR2, 'Test MSE': testMSE, 'Test MAE': testMAE}
    params.update(scores)
    return output_df.append(params, ignore_index=True)


def printLassoCoefs(lasso_coefs, x_train):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(x_train.columns, lasso_coefs, color='#111111')
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.show()

    print("Number of features before Lasso:", len(lasso_coefs))
    print("Number of features after fitting Lasso:", len(lasso_coefs[lasso_coefs > 0]))
    print("\n")
    return len(lasso_coefs)-len(lasso_coefs[lasso_coefs > 0])

