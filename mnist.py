#
# MNIST - Multiclass Classification
# What number is it?
# Chapter 3 Hands On ML
#

import utils
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as sk_svm
import sklearn.metrics as sk_metrics
import sklearn.multiclass as sk_multi
import sklearn.neighbors as skn
import sklearn.model_selection as sk_ms
import sklearn.linear_model as sk_linear
import sklearn.preprocessing as skp

mnist = utils.fetch_sklearn('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

digit = X.to_numpy()[0]
digit2 = X.to_numpy()[1]
digit5 = X.to_numpy()[4]
d_image = digit.reshape(28, 28)
d_image2 = digit2.reshape(28, 28)
d_image5 = digit5.reshape(28, 28)

plt.imshow(d_image, cmap="binary")
plt.axis("off")
plt.show()

plt.imshow(d_image2, cmap="binary")
plt.axis("off")
plt.show()

plt.imshow(d_image5, cmap="binary")
plt.axis("off")
plt.show()

# Convert y to an integer
y = y.astype(np.uint8)

# Segment train vs. test data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

###########################################
# Support Vector Machine Classifier - OvO
###########################################

svm_clf = sk_svm.SVC()
svm_clf.fit(X_train, y_train)
svm_prediction = svm_clf.predict([digit])
print("What is it? ")
print(svm_prediction)

svm_prediction2 = svm_clf.predict([digit2])
print("What is it? ")
print(svm_prediction2)

svm_prediction3 = svm_clf.predict([digit5])
print("What is it? ")
print(svm_prediction3)
print("\n")

# One score per class for each prediction:
print("Decision Scores")
print("----------------")
print("First prediction (5):")
scores = svm_clf.decision_function([digit])
print(scores)
print("Max score: ", np.argmax(scores))

print("Second prediction (0):")
scores2 = svm_clf.decision_function([digit2])
print(scores2)
print("Max score: ", np.argmax(scores2))

print("Third prediction (9):")
scores5 = svm_clf.decision_function([digit5])
print(scores5)
print("Max score: ", np.argmax(scores5))
print("\n")

###########################################
# OneVsRestClassifier - OvR
###########################################
ovr_clf = sk_multi.OneVsRestClassifier(sk_svm.SVC())
ovr_clf.fit(X_train, y_train)
ovr_prediction = ovr_clf.predict([digit5])
print("OvR - What is it? ")
print(ovr_prediction)
print("\n")

#len(ovr_clf.estimators_)

############################################
# SGDClassifier
############################################

sgd_clf = sk_linear.SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
sgd_prediction = sgd_clf.predict([digit2])
print("SGD - What is it? ")
print(sgd_prediction)
print("\n")

print("SGD prediction (0):")
sgd_scores = sgd_clf.decision_function([digit2])
print(scores2)
print("Max score: ", np.argmax(scores2))
print("\n")

sgd_cv_scores = sk_ms.cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print("Accuracy scores: ", sgd_cv_scores)

# Improve accuracy by scaling inputs
scaler = skp.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
scaled_cv_scores = sk_ms.cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
print("Scaled accuracy scores: ", scaled_cv_scores)

# Plot Confusion Matrix
y_train_pred = sk_ms.cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = sk_metrics.confusion_matrix(y_train, y_train_pred)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# Plot Errors
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

###################################
# K Neighbors Classifier
###################################
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = skn.KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_pred = knn_clf.predict([digit5])
print("Is the digit large and/or odd? ")
print(knn_pred)
print("\n")
