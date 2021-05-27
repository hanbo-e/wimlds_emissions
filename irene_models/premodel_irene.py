# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:50:55 2021

@author: Oswin
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
# from sklearn.base import BaseEstimator, TransformerMixin

from emissions.data import load_data, clean_data, split
from emissions.transformer import MakeTransformer

def scoring_table(search, param_index,cols):
    """
    takes grid search output and index of best params
    returns a scoring table
    """
    result = search.cv_results_
    tmp = pd.DataFrame({'train':{'accuracy': result['mean_train_accuracy'][param_index], 
                           'recall': result['mean_train_recall'][param_index],
                           'precision': result['mean_train_precision'][param_index]}, 
                  'val':{'accuracy': result['mean_test_accuracy'][param_index], 
                           'recall': result['mean_test_recall'][param_index],
                           'precision': result['mean_test_precision'][param_index]}
                 })

    y_pred = search.best_estimator_.predict(X_test[cols])
    y_true = y_test
    tmp.loc['accuracy', 'test'] = accuracy_score(y_true, y_pred)
    tmp.loc['recall', 'test'] = recall_score(y_true, y_pred)
    tmp.loc['precision', 'test'] = precision_score(y_true, y_pred)
    return tmp.round(3)

def plot_learning_curve(model, X_train, y_train, name='test', scoring='recall'):
    """takes a model, X_train, y_train and plots learning curve"""

   
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(model.best_estimator_, 
                                                            X_train, 
                                                            y_train, 
                                                            train_sizes=np.linspace(0.05, 1, 20),
                                                            cv=cv,
                                                            scoring=scoring,
                                                            n_jobs=-1
                                                           )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, label = 'Train')
    plt.fill_between(train_sizes, 
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, 
                     alpha=0.1)

    plt.plot(train_sizes, test_scores_mean, label = 'Val')
    plt.fill_between(train_sizes, 
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, 
                     alpha=0.1)
    plt.legend()
    plt.ylabel('score')
    plt.xlabel('train sizes')
    if scoring=='recall':
        plt.ylim(0.6, 1)
    plt.savefig('../tree_figs/' + name + '.png', bbox_inches='tight')
    plt.close()
        
## Get and clean data

# first of all: get the data
df = load_data()
df = clean_data(df)

# second of all: split the data
X_train, X_test, y_train, y_test = split(df)

# interesting columns: THIS DOES NOT WORK WITH MAKE FOR NOW
col_names = ['MODEL_YEAR','VEHICLE_AGE','MILE_YEAR','GVWR','ENGINE_SIZE',
             'TRANS_TYPE','TEST_TYPE','ENGINE_WEIGHT_RATIO','VEHICLE_TYPE'] 
            # ,'BEFORE_2000','SPORT','MAKE_VEHICLE_TYPE','MAKE'

m = 20

## full feature tree
cols = col_names
cat_cols = []
if 'TRANS_TYPE' in cols:
    cat_cols.extend(['TRANS_TYPE'])
if 'TEST_TYPE' in cols:
    cat_cols.extend(['TEST_TYPE'])
if 'MAKE_VEHICLE_TYPE' in cols:
    cat_cols.extend(['MAKE_VEHICLE_TYPE'])
if ('MAKE' in cols):
    # transform make
    make_processor = Pipeline([
        ('make_transformer', MakeTransformer()),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('make_processor', make_processor, ['MAKE']),
        ('encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)], 
        remainder='passthrough'
    ) 
    # Combine preprocessor and linear model in pipeline
    pipe = Pipeline([
        ('preprocessing', preprocessor),
        ('model', DecisionTreeClassifier(class_weight='balanced'))
    ])
elif cat_cols != []: 
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)], 
        remainder='passthrough'
    )
    # Combine preprocessor and linear model in pipeline
    pipe = Pipeline([
        ('preprocessing', preprocessor),
        ('model', DecisionTreeClassifier(class_weight='balanced'))
    ])
else: 
    pipe = Pipeline([
        ('model', DecisionTreeClassifier(class_weight='balanced'))
    ])

# Hyperparameter Grid
grid = {'model__max_depth': np.arange(2, m, 1)}

# Instanciate Grid Search
search = GridSearchCV(pipe, 
                      grid, 
                      scoring=['accuracy', 'recall', 'precision'],
                      cv=10,
                      refit='recall',
                      return_train_score=True,
                      n_jobs=18
                     ) 
search.fit(X_train[cols], y_train)

result = search.cv_results_

pd.DataFrame(result)[['param_model__max_depth', 
                      'mean_test_recall', 
                      'mean_train_recall']].sort_values('mean_test_recall', ascending=False).head(5)
tmp = scoring_table(search, search.best_index_, cols) 

name = 'DT_all_' + str(search.best_params_['model__max_depth'])

plot_learning_curve(search, X_train[cols], y_train, name=name)

df_res = pd.DataFrame()
df_all = pd.DataFrame()

data = {'features': name,
       	'no features': len(cols),
        'depth': search.best_params_['model__max_depth'],
       	'acc': tmp.test[0],
       	'rec': tmp.test[1],
       	'prc': tmp.test[2]}

df_res = df_res.append(data, ignore_index=True)
df_all = df_all.append(data, ignore_index=True)

# save depth and recall as comparison
vgl = [tmp.test[1], search.best_index_]

## checking all possible combinations and compare to full feature tree

# for i in range(len(col_names)-1,2,-1):
for i in range(3,len(col_names)):
    col_combs = list(itertools.combinations(col_names, i))
    print("----- No of features: " + str(i) + "; possible combinations: " + str(len(col_combs)) + " -----")
    x = 0
    for col_tup in col_combs:
        x += 1
        print(str(x) + "/" + str(len(col_combs)))
        cols = list(col_tup)
        cat_cols = []
        if 'TRANS_TYPE' in cols:
            cat_cols.extend(['TRANS_TYPE'])
        if 'TEST_TYPE' in cols:
            cat_cols.extend(['TEST_TYPE'])
        if 'MAKE_VEHICLE_TYPE' in cols:
            cat_cols.extend(['MAKE_VEHICLE_TYPE'])
        if ('MAKE' in cols):
            # transform make
            make_processor = Pipeline([
                ('make_transformer', MakeTransformer()),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            # Preprocessor
            preprocessor = ColumnTransformer([
                ('make_processor', make_processor, ['MAKE']),
                ('encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)], 
                remainder='passthrough'
            ) 
            # Combine preprocessor and linear model in pipeline
            pipe = Pipeline([
                ('preprocessing', preprocessor),
                ('model', DecisionTreeClassifier(class_weight='balanced'))
            ])
        elif cat_cols != []: 
            # Preprocessor
            preprocessor = ColumnTransformer([
                ('encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)], 
                remainder='passthrough'
            )
            # Combine preprocessor and linear model in pipeline
            pipe = Pipeline([
                ('preprocessing', preprocessor),
                ('model', DecisionTreeClassifier(class_weight='balanced'))
            ])
        else: 
            pipe = Pipeline([
                ('model', DecisionTreeClassifier(class_weight='balanced'))
            ])
        
        # Hyperparameter Grid
        grid = {'model__max_depth': np.arange(2, m, 1)}
        
        # Instanciate Grid Search
        search = GridSearchCV(pipe, 
                              grid, 
                              scoring=['accuracy', 'recall', 'precision'],
                              cv=10,
                              refit='recall',
                              return_train_score=True,
                              n_jobs=18
                             ) 
        search.fit(X_train[cols], y_train)
        
        result = search.cv_results_
        
        pd.DataFrame(result)[['param_model__max_depth', 
                              'mean_test_recall', 
                              'mean_train_recall']].sort_values('mean_test_recall', ascending=False).head(5)
        tmp = scoring_table(search, search.best_index_, cols) 
        strts = ''
        for s in cols:
            if s == 'ENGINE_WEIGHT_RATIO':
                strts = strts + 'EW_'
            elif s == 'MAKE_VEHICLE_TYPE':
                strts = strts + 'MV_'
            else:
                strts = strts + str(s[0:2]) + '_'
        name = 'DT_' + strts + str(search.best_params_['model__max_depth'])
        
        row = {'features': name,
               	'no features': len(cols),
                'depth': search.best_params_['model__max_depth'],
               	'acc': tmp.test[0],
               	'rec': tmp.test[1],
               	'prc': tmp.test[2]}
        
        df_all = df_all.append(row, ignore_index=True)
        
        # EVALUATE MODEL: df for 
        if (tmp.test[1] > vgl[0]) or ((tmp.test[1] == vgl[0]) & (search.best_index_ != vgl[1])):
            
            plot_learning_curve(search, X_train[cols], y_train, name=name)
            
            df_res = df_res.append(row, ignore_index=True)
            
            # update comarison values
            if tmp.test[1] > vgl[0]:
                vgl[0] = tmp.test[1]
                vgl[1] = search.best_index_
                    
    df_res.to_csv('../DT_' + str(i) + '.csv',index=False)
    df_all.to_csv('../DT_all_' + str(i) + '.csv',index=False)
            
df_res.to_csv('../summary_DT.csv',index=False)
df_all.to_csv('../summary_DT_all.csv',index=False)