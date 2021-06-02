import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from emissions.trainer import Trainer
from emissions.data import load_data, clean_data
from sklearn.metrics import precision_score
from termcolor import colored


class Implementer():
    """ 
    this class is built to used to facilitate analysis for answering following question:
    How different {year} could have been with implementation of our solution?
    """
    cols = ['VEHICLE_AGE', 'MILE_YEAR', 'MAKE', 'MODEL_YEAR', 'ENGINE_WEIGHT_RATIO']
    cat_col = ['MAKE']
    def __init__(self, plot='both'):
        """
        plot: possible values 'recall', 'precision', 'both'
        """
        self.plot = plot
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.total_tests = None
        self.total_fails = None
        
    def load_data(self):
        """
        1. loads clean data and save it as class attribute self.df
        2. adds counter columns for all the tests and failed tests
            count_fail: 1 if the test result is fail else 0
            count_test: 1 for each test
        """
        df = load_data()
        df = clean_data(df)
        df['count_test'] = 1
        df['count_fail'] = df.RESULT
        self.df = df
    
    def train_test_split(self, year):
        ''' 
        splits the data into train and test sets using the given year
        '''
        train = self.df[self.df.TEST_SDATE.dt.year < year].sort_values('TEST_SDATE')
        test = self.df[self.df.TEST_SDATE.dt.year == year].sort_values('TEST_SDATE')
        self.y_train = train.pop('RESULT')
        self.X_train = train
        self.y_test = test.pop('RESULT')
        self.X_test = test
        self.total_tests = self.X_test.shape[0]
        self.total_fails = self.y_test.sum()

    def get_estimator(self, metric):  
        '''
        uses Trainer class from trainer.py to get best estimator based on the given metric
        prints the evluation scores
        plots learning curve
        ''' 
        trainer = Trainer(self.X_train[self.cols],
                          self.y_train,
                          metric=metric,
                          with_categorical=self.cat_col,
                          max_depth=np.arange(3, 10, 1)
         )
        trainer.grid_search()
        print('\nBest max_depth:', trainer.search_result.best_params_['model__max_depth'])
        tmp = trainer.evaluate(self.X_test[self.cols], self.y_test)
        print(tmp)
        # trainer.learning_curve()
        return trainer
    
    def get_counter_table(self):
        '''
        creates a counter table with TEST_SDATE as index and having columns:
            n_tests: cumulative number of tests along the year
            n_fails: cumulative number of failed tests along the year
        '''
        df = self.X_test[['TEST_SDATE', 'count_fail', 'count_test']].copy()
        df.set_index(pd.DatetimeIndex(df.TEST_SDATE), inplace=True)
        df['n_tests'] = df.count_test.cumsum()
        df['n_fails'] = df.count_fail.cumsum()
        df.drop(columns=['count_fail', 'count_test', 'TEST_SDATE'], inplace=True)
        return df
    
    def count_fails_captured(self, 
                             data, 
                             precision, 
                             predicted_fails,
                             true_fails):
        """ 
        params:
        data: output df from get_cumulated_tests
        precision: precision score on test set
        predicted_fails: number of fails predicted
        true_fails: number of predicted failed vehicles that acutally fail the test
        returns a panda series: number of fails captured along the year
        """
        df = data.copy()
        # first test all the predicted fails
        df['fails_captured'] = df.n_tests * precision
        # changes after finish testing predicted fails
        tests_left = df[df.n_tests > predicted_fails].shape[0]
        fails_left = self.total_fails - true_fails
        avg_fail_per_test = fails_left/tests_left
        df.loc[df.n_tests > predicted_fails, 'fails_captured'] = true_fails +\
            (df[df.n_tests > predicted_fails]['n_tests'] - predicted_fails)*avg_fail_per_test
        return df.fails_captured
        
    def plot_comparison(self, year, df, predicted_fails1, predicted_fails2):
        '''
        for given year plots:
        1. how many failed tests were detected along that year
        2. how many failed tests could have been deen detected along that year if our solution is implemented
        '''
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, 
                df.n_fails, 
                label='with current policy', c='red')
            
        if self.plot == 'both' or self.plot == 'recall':
            
            plt.plot(df.index, 
                     df.fails_captured_r, 
                     label='with solution favoring recall', c='green')
            # mark the time when all the predicted fail tests from r favored model are completed
            t1 = df[df.n_tests==predicted_fails1].index[0]
            cap_fails1 = df[df.n_tests==predicted_fails1].fails_captured_r[0]
            true_fails1 = df[df.n_tests==predicted_fails1].n_fails[0]
            print(f'\nBy the time {str(t1)[:10]}, {round(true_fails1)} vehicles were off the road in reality')
            print(f'By the time {str(t1)[:10]}, {round(cap_fails1)} vehicles could have been off the road using model favoring recall')
            plt.plot([t1 for i in range(100)], 
                    np.linspace(0, cap_fails1, 100), c='grey')
            plt.plot(df[df.index < t1].index, 
                    [cap_fails1 for i in range(df[df.index < t1].shape[0])], 
                    c='grey')
            # fill the area corresponding to reduced pollution
            plt.fill_between(df.index,
                            df.n_fails,
                            df.fails_captured_r,
                            color='green', alpha=0.1
                            )
            
        if self.plot == 'both' or self.plot == 'precision':
            plt.plot(df.index, 
                     df.fails_captured_p, 
                     label='with solution favoring precision', c='blue')
            # mark the time when all the predicted fail tests from precision favored model are completed
            t2 = df[df.n_tests==predicted_fails2].index[0]
            cap_fails2 = df[df.n_tests==predicted_fails2].fails_captured_p[0]
            true_fails2 = df[df.n_tests==predicted_fails2].n_fails[0]
            print(f'\nBy the time {str(t2)[:10]}, {round(true_fails2)} vehicles were off the road in reality')
            print(f'By the time {str(t2)[:10]}, {round(cap_fails2)} vehicles could have been off the road using model favoring precision')
            plt.plot([t2 for i in range(100)], 
                    np.linspace(0, cap_fails2, 100), c='grey')
            plt.plot(df[df.index < t2].index, 
                    [cap_fails2 for i in range(df[df.index < t2].shape[0])], 
                    c='grey')
            # fill in the area corresponding to reduced pollution
            plt.fill_between(df.index,
                            df.n_fails,
                            df.fails_captured_p,
                            color='green', alpha=0.1
                            )
        # fill in the area corresponding to total pollution
        plt.fill_between(df.index,
                        df.n_fails,
                        [self.total_fails for i in range(df.shape[0])],
                        color='grey', alpha=0.1
                        )
        plt.ylim(0, self.total_fails)
        plt.xlim(df.index.min(), df.index.max())
        plt.ylabel('number of polluting vehicles')
        plt.xlabel('date')
        plt.legend()
        plt.title(f'Number of polluting vehicles detected over the year {year}')
        
    def implement(self, year):
        # train set split
        self.train_test_split(year)
        # get counter table
        df = self.get_counter_table()
        # get best estimator favoring recall
        print(colored('----------------- getting best estimator favoring recall ----------------', 'green'))
        trainer_r = self.get_estimator('recall')
        # get all the numbers
        y_pred_r = trainer_r.search_result.predict(self.X_test[self.cols])
        precision_r = precision_score(self.y_test, y_pred_r)
        true_fails_r = sum([i for i, j in zip (y_pred_r, self.y_test) if i + j == 2])
        predicted_fails_r = y_pred_r.sum()
        # add new columns storing number of failed test captured along the year
        df['fails_captured_r'] = self.count_fails_captured(df, 
                                                        precision_r, 
                                                        predicted_fails_r, 
                                                        true_fails_r
                                                        )
        
        # get best estimator favoring precision
        print(colored('--------------- getting best estimator favoring precision --------------', 'green'))
        trainer_p = self.get_estimator('precision')
        # get all the number 
        y_pred_p = trainer_p.search_result.predict(self.X_test[self.cols])
        precision_p = precision_score(self.y_test, y_pred_p)
        true_fails_p = sum([i for i, j in zip(y_pred_p, self.y_test) if i + j == 2])
        predicted_fails_p = y_pred_p.sum()
        # add new columns storing number of failed test captured along the year
        df['fails_captured_p'] = self.count_fails_captured(df, 
                                                        precision_p, 
                                                        predicted_fails_p, 
                                                        true_fails_p
                                                        )
        
        # plot the comparision
        self.plot_comparison(year, df, predicted_fails_r, predicted_fails_p)
        
if __name__ == "__main__":
    imp = Implementer()
    imp.load_data()
    imp.train_test_split(2020)
    print('Total tests in 2020', imp.total_tests)
    print('Total failed test in 2020', imp.total_fails)