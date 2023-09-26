# -*- coding: utf-8 -*-
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

model = xgb.XGBClassifier(n_estimators=500, random_state=42)#can be set as 100, 500 and 1000, respectively

model.fit(matrix[:,:], T[:,0])

predictions_p = model.predict(matrix1[:])
import pickle

with open('xgboost_model1.pkl', 'wb') as f:
    pickle.dump(model, f)