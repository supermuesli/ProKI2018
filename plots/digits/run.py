#!/usr/bin/env python3

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

def plot_learning_curve(names, estimators, title, X, y, ylim=None, cv=None,  n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
	plt.figure()
	plt.title(title)
	if ylim is not None: plt.ylim(*ylim)

	plt.xlabel("Training examples")
	plt.ylabel("Accuracy")

	for name, estimator in zip(names, estimators):
		print ("Plotting:", name)
		train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		plt.grid()

		print ("Accuracy:", test_scores_mean[-1])

		#plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1)
		#plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1)
		#plt.plot(train_sizes, train_scores_mean, 'o-', label="Training accuracy")
		plt.plot(train_sizes, test_scores_mean, 'o-', label="%s Accuracy"%name)
		plt.legend(loc="best")


	return plt


# The digits dataset
digits = datasets.load_digits()

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
print("NSAMPLES:", n_samples)
data = digits.images.reshape((n_samples, -1))

#pca = PCA(n_components=64)
#data = pca.fit_transform(data)

X, y = (data[:n_samples],  digits.target[:n_samples])

# Create a classifier: a support vector classifier
#classifier = svm.SVC(kernel="rbf", gamma=0.001)
classifiers = [svm.SVC(kernel="linear"), svm.SVC(kernel="rbf", gamma=0.001), MLPClassifier(), SGDClassifier()]

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(["Linear SVM", "RBF SVM", "FFNN", "SGDC"], classifiers, "Comparison: Learning Rates", X, y, (0.7, 1.01), cv=cv, n_jobs=4)

print ("Saving figures")
plt.savefig('pcalr.png', figsize=(8, 6), dpi=400,)
plt.savefig('pcalr.pgf')
