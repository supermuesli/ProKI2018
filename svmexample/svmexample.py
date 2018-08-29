#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import svm, datasets

# Seed for linear.pgf = 1529502007

t = int(time.time())
print ("Time:", t)
np.random.seed(t)


kernel = sys.argv[1]

# Generate datasets

# Random sets
#X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
#Y = [0] * 20 + [1] * 20

# Two circles
X, Y = datasets.make_circles(noise=0.2, factor=0.2, random_state=1529503094)

# Train the kernel
clf = svm.SVC(kernel=kernel, gamma=10, C=10)
clf.fit(X, Y)

# Plot the kernel linear
if kernel == "linear":

	# Get the separating hyperplane
	w = clf.coef_[0]
	a = -w[0] / w[1]
	xx = np.linspace(-2, 2) # TODO maybe remove
	yy = np.linspace(-2, 2) # TODO maybe remove

	# Parallels through support vectors
	margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
	yy_down = yy - np.sqrt(1 + a ** 2) * margin
	yy_up = yy + np.sqrt(1 + a ** 2) * margin

	plt.figure()
	plt.clf()
	plt.plot(xx, yy, 'k-')
	plt.plot(xx, yy_down, 'k--')
	plt.plot(xx, yy_up, 'k--')

	plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
	            facecolors='none', zorder=10, edgecolors='k')

# Plot the kernel rbf
if kernel == "rbf":

	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	h = .02

	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
            edgecolors='k')

plt.axis('tight')

#plt.savefig('plot.pgf')

# TODO production pgf files
plt.savefig(sys.argv[1]+'.png', figsize=(8, 6), dpi=400,)
plt.savefig(sys.argv[1]+'.pgf')
#plt.show()
