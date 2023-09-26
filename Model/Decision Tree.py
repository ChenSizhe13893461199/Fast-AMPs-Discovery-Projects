# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(matrix[:,:], T[:,0])

predictions_p = model.predict(matrix1[:])
import pickle

with open('decision_tree_model1.pkl', 'wb') as f:
    pickle.dump(model, f)