#!/usr/bin/env python3


import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv("countries.csv", header=0)

X = data.loc[:, ['Agriculture', 'Industry', 'Service']]
y = data.loc[:, ['Phones (per 1000)']]

print (X.shape)
print (y.shape)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(X, y)     
