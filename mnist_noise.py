#
# MNIST Classification: Little Noise Exercise
# Chapter 3 Hands On ML
#

import utils
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as skn


mnist = utils.fetch_sklearn('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Convert y to an integer
y = y.astype(np.uint8)

# Segment train vs. test data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

# Check out a digit from the test set
some_digit = X_test_mod.to_numpy()[2]
sd_image = some_digit.reshape(28, 28)
plt.imshow(sd_image, cmap="binary")
plt.axis("off")
plt.show()

# TODO: Fix this
# Something went wrong...this did not show anything in the plot ?
knn_cli = skn.KNeighborsClassifier()
knn_cli.fit(X_train_mod, y_train_mod)
clean_digit = knn_cli.predict([some_digit])
plt.imshow(clean_digit, cmap="binary")
plt.axis("off")
plt.show()


