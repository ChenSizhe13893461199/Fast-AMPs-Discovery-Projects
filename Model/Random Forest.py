# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
model = RandomForestClassifier(n_estimators=100, random_state=42)#can be set as 100, 500 and 1000, respectively

model.fit(matrix[:,:], T[:,0])

predictions_p = model.predict(matrix1[:])
import pickle

with open('random_forest_model1.pkl', 'wb') as f:
    pickle.dump(model, f)