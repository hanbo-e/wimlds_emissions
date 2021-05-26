from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score
from emissions.data import load_data, clean_data


def scoring_table(search, 
                  X_test,
                  y_test):
    """
    takes grid search output and index of best params
    returns a scoring table
    """
    result = search.cv_results_
    tmp = pd.DataFrame({'train':{'accuracy': result['mean_train_accuracy'][search.best_index_], 
                           'recall': result['mean_train_recall'][search.best_index_],
                           'precision': result['mean_train_precision'][search.best_index_]}, 
                  'val':{'accuracy': result['mean_test_accuracy'][search.best_index_], 
                           'recall': result['mean_test_recall'][search.best_index_],
                           'precision': result['mean_test_precision'][search.best_index_]}
                 })

    y_pred = search.best_estimator_.predict(X_test)
    y_true = y_test
    tmp.loc['accuracy', 'test'] = accuracy_score(y_true, y_pred)
    tmp.loc['recall', 'test'] = recall_score(y_true, y_pred)
    tmp.loc['precision', 'test'] = precision_score(y_true, y_pred)
    return tmp.round(3)

def plot_learning_curve(model, X_train, y_train, scoring='recall'):
    """takes a model, X_train, y_train and plots learning curve"""

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(model, 
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
    
def make_transform_get(df, make_threshhold="0.01"):
    '''
    Take cleaned training data and return a list of makes to be converted to 'other'
    '''
    #create a make label 'other' for all makes that only account for less than 1% of cars each and together aprox <10% of cars
    value_counts_norm = df['MAKE'].value_counts(normalize = True)
    to_other = value_counts_norm[value_counts_norm < float(make_threshhold)]
    print(f"\n{len(to_other)} make labels each account for less than {round((float(make_threshhold) *100), 2)}% of cars and together account for {(round(to_other.sum(), 4)) *100}% of cars")
    to_keep = value_counts_norm[value_counts_norm >= float(make_threshhold)]
    makes_keep = list(to_keep.index)
    makes_keep.sort()
    return makes_keep

if __name__=="__main__":
    df = load_data()
    df = clean_data(df)
    print('Makes to keep:', make_transform_get(df))