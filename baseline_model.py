#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 20:57:19 2021

@author: hanbo
"""


import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

DATA_DIR = Path("data/clean_data_20210505.csv")
df = pd.read_csv(DATA_DIR, low_memory=False, sep=',',index_col='RecordID', usecols=['RecordID','RESULT', 'ODOMETER', 'VEHICLE_AGE'])
df.shape
df.nunique()
df.info()
df.head()

#train-test-split
X = df[['ODOMETER', 'VEHICLE_AGE']]
y = df['RESULT']
X.head()
y.head()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=24)
DT = DecisionTreeClassifier(max_depth =4)
DT.fit(Xtrain, ytrain)
DT.score(Xtrain, ytrain)

tree.plot_tree(DT) #can't make out the plot
import graphviz
dot_data = tree.export_graphviz(DT, out_file=None, \
                                feature_names=['Odometer', 'Age'],
                                filled=True,\
                                rounded=True)
graph = graphviz.Source(dot_data)
graph.render("emissions")

#trying out the attributes and methods
DT.classes_
DT.feature_importances_
DT.max_features_
DT.n_classes_
DT.n_features_
DT.n_outputs_
DT.tree_


X = [[5,1]]
print (DT.decision_path(X))
DT.get_depth()
DT.get_n_leaves()
DT.predict(X)
DT.predict_proba(X)
DT.get_params()
