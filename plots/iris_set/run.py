#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# import some data to play with
iris = datasets.load_iris()

X = iris.data
y = iris.target


X = X[y != 0, :2] # we only take the first two features. We could avoid this ugly slicing by using a two-dim dataset
y = y[y != 0]

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(np.float)

X_train = X[:int(.9 * n_sample)]
y_train = y[:int(.9 * n_sample)]
X_test = X[int(.9 * n_sample):]
y_test = y[int(.9 * n_sample):]

kernel = sys.argv[1]
clf = svm.SVC(kernel=kernel, gamma=10)
clf.fit(X_train, y_train)

plt.subplot(1, 1, 1)

plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolor='k', s=20)

# Circle out the test data
plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',  zorder=10, edgecolor='k')

plt.axis('tight')


x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()


XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])


# Put the result into a color plot
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)


plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
plt.title(kernel)

#plt.savefig('plot.pgf')

plt.savefig(sys.argv[1]+'.png', figsize=(8, 6), dpi=400,)

#plt.show()
