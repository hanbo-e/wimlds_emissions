# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:42:55 2021

@author: Oswin
"""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from emissions.data import load_data, clean_data, split
from emissions.trainer import MakeTransformer

from explainerdashboard import ClassifierExplainer, ExplainerDashboard

# get the data and split
df = load_data()
df = clean_data(df)
X_train, X_test, y_train, y_test = split(df)

# choose initial columns to be check feature importance
cols = ['MODEL_YEAR','VEHICLE_AGE','MILE_YEAR', 'ENGINE_WEIGHT_RATIO','MAKE']

# transform rare MAKE into other
mt = MakeTransformer().fit(X_train[cols])
print("\nMAKEs don't belong to other:", mt.makes_keep)
X_train_update = mt.transform(X_train[cols])
print('\nNumber of unique makes in train', X_train_update.MAKE.nunique())
X_test_update = mt.transform(X_test[cols])
print('\nNumber of unique makes in test', X_test_update.MAKE.nunique())
        
# transform MAKE into one-hot numeric array
enc = OneHotEncoder(handle_unknown='ignore')
MAKE_train = pd.DataFrame(enc.fit_transform(X_train_update[['MAKE']]).toarray())
MAKE_train = MAKE_train.add_prefix('MAKE_')
MAKE_test = pd.DataFrame(enc.fit_transform(X_test_update[['MAKE']]).toarray())
MAKE_test = MAKE_test.add_prefix('MAKE_')

# drop MAKE and add the one-hot numeric array to form one new data frame
X_train_rel = X_train_update.drop('MAKE',axis=1)
X_train_rel.reset_index(drop=True, inplace=True)
MAKE_train.reset_index(drop=True, inplace=True)
X_train_rel = pd.concat([X_train_rel, MAKE_train],axis=1)
X_test_rel = X_test_update.drop('MAKE',axis=1)
X_test_rel.reset_index(drop=True, inplace=True)
MAKE_test.reset_index(drop=True, inplace=True)
X_test_rel = pd.concat([X_test_rel, pd.DataFrame(MAKE_test)],axis=1)

# fit a standard RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, max_depth=10)
model.fit(X_train_rel, y_train)

# use Explainer Dashboard
explainer = ClassifierExplainer(model, X_test_rel, y_test)
db = ExplainerDashboard(explainer)
db.run()