# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:50:55 2021

@author: Oswin
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from emissions.data import load_data, clean_data, split
from emissions.transformer import MakeTransformer

def make_transform_get(df, make_threshhold="0.01"):
    '''
    Take cleaned training data and return a list of makes to be converted to 'other'
    '''
    #set make to string and to lower case, strip trailing and internal whitespace
    #df['MAKE'] = df['MAKE'].astype('string').str.strip().str.lower().str.replace(' ', '')
    #create a make label 'other' for all makes that only account for less than 1% of cars each and together aprox <10% of cars
    value_counts_norm = df['MAKE'].value_counts(normalize = True)
    to_other = value_counts_norm[value_counts_norm < float(make_threshhold)]
    print(f"\n{len(to_other)} make labels each account for less than {round((float(make_threshhold) *100), 2)}% of cars and together account for {(round(to_other.sum(), 4)) *100}% of cars")
    #print("Grouping these car makes into one category called 'other'")
    #df['MAKE'] = df['MAKE'].replace(to_other.index, 'other')
    list_to_other = list(to_other.index)
    list_to_other.sort()
    return list_to_other

def make_transform_set(df, list_to_other):
    '''
    Take test df and replace make labels from list with 'other'
    '''
    df = df.copy()
    df['MAKE'] = df['MAKE'].replace(list_to_other, 'other')
    return df

def make_engineer(df):
    df = df.copy()
    #df['MAKE'] = df['MAKE'].astype('string').str.strip().str.lower().str.replace(' ', '')
    df['MAKE_VEHICLE_TYPE'] = df['MAKE'] + df.VEHICLE_TYPE.astype('str')
    return df

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

list_to_other = make_transform_get(X_train)
X_train = make_transform_set(X_train, list_to_other)
X_train = make_engineer(X_train)
X_test = make_transform_set(X_test, list_to_other)
X_test = make_engineer(X_test)

# mt = MakeTransformer.fit(X_train)
# X_train_update = mt.transform(X_train)
# X_test_update = mt.transform(X_test)

m = 20

df_all = pd.DataFrame()

## checking good models with 'MAKE' added

col_combs = [('MODEL_YEAR', 'VEHICLE_AGE', 'MILE_YEAR', 'GVWR', 'ENGINE_SIZE', 'TRANS_TYPE', 'TEST_TYPE', 'ENGINE_WEIGHT_RATIO', 'VEHICLE_TYPE'),
             ('MODEL_YEAR', 'VEHICLE_AGE', 'TRANS_TYPE'),
             ('MODEL_YEAR', 'MILE_YEAR', 'ENGINE_SIZE'),
             ('VEHICLE_AGE', 'MILE_YEAR', 'GVWR'),
             ('VEHICLE_AGE', 'MILE_YEAR', 'ENGINE_SIZE')]

# for i in range(len(col_names)-1,2,-1):
for col_tup in col_combs:
    cols = list(col_tup) + ['MAKE']
    print(cols)
    cat_cols = ['MAKE']
    if 'TRANS_TYPE' in cols:
        cat_cols.extend(['TRANS_TYPE'])
    if 'TEST_TYPE' in cols:
        cat_cols.extend(['TEST_TYPE'])
    if 'MAKE_VEHICLE_TYPE' in cols:
            cat_cols.extend(['MAKE_VEHICLE_TYPE'])
    if cat_cols != []: 
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
    name = 'DT_' + strts + str(search.best_index_)
    
    plot_learning_curve(search, X_train[cols], y_train, name=name)
    
    row = {'features': [cols],
           	'no features': len(cols),
            'depth': search.best_index_,
           	'acc': tmp.test[0],
           	'rec': tmp.test[1],
           	'prc': tmp.test[2]}
    
    df_all = df_all.append(row, ignore_index=True)
        
## checking good models with 'MAKE_VEHICLE_TYPE' added

col_combs = [('MODEL_YEAR', 'VEHICLE_AGE', 'MILE_YEAR', 'GVWR', 'ENGINE_SIZE', 'TRANS_TYPE', 'TEST_TYPE', 'ENGINE_WEIGHT_RATIO'),
             ('MODEL_YEAR', 'VEHICLE_AGE', 'TRANS_TYPE'),
             ('MODEL_YEAR', 'MILE_YEAR', 'ENGINE_SIZE'),
             ('VEHICLE_AGE', 'MILE_YEAR', 'GVWR'),
             ('VEHICLE_AGE', 'MILE_YEAR', 'ENGINE_SIZE')]

# for i in range(len(col_names)-1,2,-1):
for col_tup in col_combs:
    cols = list(col_tup) + ['MAKE_VEHICLE_TYPE']
    print(cols)
    cat_cols = ['MAKE_VEHICLE_TYPE']
    if 'TRANS_TYPE' in cols:
        cat_cols.extend(['TRANS_TYPE'])
    if 'TEST_TYPE' in cols:
        cat_cols.extend(['TEST_TYPE'])
    if cat_cols != []: 
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
    name = 'DT_' + strts + str(search.best_index_)
    
    plot_learning_curve(search, X_train[cols], y_train, name=name)
    
    row = {'features': [cols],
           	'no features': len(cols),
            'depth': search.best_index_,
           	'acc': tmp.test[0],
           	'rec': tmp.test[1],
           	'prc': tmp.test[2]}
    
    df_all = df_all.append(row, ignore_index=True)
            
df_all.to_csv('../summary_DT_all_make.csv',index=False)
            