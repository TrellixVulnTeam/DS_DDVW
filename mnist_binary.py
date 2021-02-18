#
# MNIST Binary Classification: Is it a 5?
# Chapter 3 Hands On ML
#

import utils
import learn
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import sklearn.ensemble as sk_ensemble
import sklearn.model_selection as sk_ms
import sklearn.linear_model as sk_linear

mnist = utils.fetch_sklearn('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
digit = X.to_numpy()[0]
d_image = digit.reshape(28, 28)
plt.imshow(d_image, cmap="binary")
plt.axis("off")
plt.show()

# Convert y to an integer
y = y.astype(np.uint8)

# Segment train vs. test data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Target vectors for '5' classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Stochastic Gradient Descent Classifier
sgd_clf = sk_linear.SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
prediction = sgd_clf.predict([digit])
print("Is it a 5? ", prediction)

# Cross-Validation Folds Scores (not very useful)
scores = sk_ms.cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print("Accuracy Scores: ")
print(scores)
print("\n")

# Confusion Matrix (more useful)
y_train_pred = sk_ms.cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = sk_metrics.confusion_matrix(y_train_5, y_train_pred)
print("Confusion Matrix: ")
print("Row 1 - Things that were NOT 5s: true negatives, false positives")
print("Row 2 - Things that were 5s: false negatives, true positives")
print(cm)
print("\n")

# Precision (TPR) = true positives / (true positives + false positives)
# Recall (Sensitivity) = true positives / (true positives + false negatives)
precision = sk_metrics.precision_score(y_train_5, y_train_pred)
print("Precision tp / (tp + fp): ")
print(precision)
recall = sk_metrics.recall_score(y_train_5, y_train_pred)
print("Recall tp / (tp + fp): ")
print(recall)
print("\n")

# F1 Score: harmonic mean of precision and recall
# Obviously favors classifiers with similar precision & recall (not always the goal)
# Increasing precision reduced recall (precision / recall trade-off)
# F1 = (2 / (1/precision + 1/recall)) = tp / tp + ((fn + fp)/ 2)
f1 = sk_metrics.f1_score(y_train_5, y_train_pred)
print("F1 Score: ")
print(f1)
print("\n")

# Tweaking the Threshold of the Decision Function
y_scores = sk_ms.cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = sk_metrics.precision_recall_curve(y_train_5, y_scores)

# Plot precision vs. recall
learn.plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# Aim for 90% precision
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
y_train_predict_90 = (y_scores >= threshold_90_precision)
print("Precision then Recall Score for 90% precision classifier:")
print(sk_metrics.precision_score(y_train_5, y_train_predict_90))
print(sk_metrics.recall_score(y_train_5, y_train_predict_90))
print("\n")

# The ROC: sensitivity (recall) vs. 1 - specificity (TNR)
fpr, tpr, thresholds = sk_metrics.roc_curve(y_train_5, y_scores)
learn.plot_roc_curve(fpr, tpr)
plt.show()

# Compute AUC
auc_score = sk_metrics.roc_auc_score(y_train_5, y_scores)
print("AUC Score: ")
print(auc_score)
print("\n")

# If the positive class is rare, prefer PR curve
# If you want to avoid false positives more than false negatives
# Otherwise, use AUC.  In this case, there are few 5s, so AUC
# doesn't tell us much.

# Set the threshold to -4000 (?)
threshold = -4000
dprediction = (y_scores > threshold)
print(dprediction)

############################
# Random Forest Classifier
############################
forest_clf = sk_ensemble.RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train_5)
forest_prediction = forest_clf.predict([digit])
print("Forest prediction. Is it a 5? ")
print(forest_prediction)
print("\n")

y_probas_forest = sk_ms.cross_val_predict(forest_clf, X_train, y_train_5,
                                          cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]
print("Accuracy Scores: ")
print(y_scores_forest)

fpr_forest, tpr_forest, thresholds_forest = sk_metrics.roc_curve(y_train_5, y_scores_forest)
plt.plot(fpr, tpr, "b:", label="SGD")
learn.plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

print("AUC Score for Random Forest Classification: ")
auc_forest = sk_metrics.roc_auc_score(y_train_5, y_scores_forest)
print(auc_forest)
print("\n")

y_forest_pred = sk_ms.cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
forest_cm = sk_metrics.confusion_matrix(y_train_5, y_forest_pred)
print("Forest Confusion Matrix: ")
print("Row 1 - Things that were NOT 5s: true negatives, false positives")
print("Row 2 - Things that were 5s: false negatives, true positives")
print(forest_cm)
print("\n")

print("Forest Precision, Recall & F1: ")
print(sk_metrics.precision_score(y_train_5, y_forest_pred))
print(sk_metrics.recall_score(y_train_5, y_forest_pred))
print(sk_metrics.f1_score(y_train_5, y_forest_pred))
print("\n")








