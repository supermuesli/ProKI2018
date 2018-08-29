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
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20


# Train the kernel
clf = svm.SVC(kernel="linear")
clf.fit(X, Y)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-4, 4) # TODO maybe remove
yy = np.linspace(-4, 4) # TODO maybe remove

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

plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
            edgecolors='k')

plt.axis('tight')

# TODO production pgf files
plt.savefig(sys.argv[1]+'.png', figsize=(8, 6), dpi=400,)
plt.savefig(sys.argv[1]+'.pgf')
#plt.show()
