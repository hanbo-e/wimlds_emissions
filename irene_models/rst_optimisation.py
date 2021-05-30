# -*- coding: utf-8 -*-
"""
Created on Sun May 30 16:53:43 2021

@author: Oswin
"""

import pandas as pd

from numpy import mean

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV

from emissions.data import load_data, clean_data, split
from emissions.trainer import MakeTransformer

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
model = RandomForestClassifier(n_estimators=100,n_jobs=18,
                               class_weight='balanced')
model.fit(X_train_rel, y_train)
y_pred = model.predict(X_test_rel)
tmp = confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))

# simple cross validation
model = RandomForestClassifier(n_estimators=100,n_jobs=18,
                               class_weight='balanced')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X_train_rel, y_train, 
                         scoring='accuracy', cv=cv, n_jobs=18)
mean(scores)


# use a Grid Search
n_estimators = [50, 100, 250, 500, 1000]
max_depth = [5, 10, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
              min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(RandomForestClassifier(class_weight='balanced'), hyperF, verbose = 1,  
                     scoring=['accuracy', 'recall', 'precision'],
                     refit='precision',
                     return_train_score=True,
                     n_jobs = 18)

bestF = gridF.fit(X_train_rel, y_train)
