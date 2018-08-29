#!/usr/bin/env python3

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

with open("data.txt") as f:
    data = np.loadtxt(f)

X = data[:, :5]
y = data[:, 5]

print (X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


tuned_parameters = {'alpha': [1e-8, 1e-2] }

"""
reg = linear_model.BayesianRidge()
reg.fit(X, Y)

print (reg.score(X, Y))
"""

score = "precision"
print("# Tuning hyper-parameters for %s" % score)
print()

clf = GridSearchCV(linear_model.Lasso(), tuned_parameters, cv=5, scoring='%s_macro' % score)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)