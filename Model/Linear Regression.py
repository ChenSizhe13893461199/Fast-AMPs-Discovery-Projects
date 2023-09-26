# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(matrix[:,:], T[:,0])

predictions_p = model.predict(matrix1[:])
import pickle

with open('linear_regression_model1.pkl', 'wb') as f:
    pickle.dump(model, f)