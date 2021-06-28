import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from emissions.trainer import Trainer
from emissions.data import load_data, clean_data
from sklearn.metrics import precision_score


class ImpSearch():
    """ 
    this class is built to facilitate analysis for answering following question:
    How different {year} could have been with implementation of our solution?
    What ImSearch do:
    1. For each param, 
       it trains the model, performs implement analysis on test year 
       and then collects total pollution quantity in that year caused by vehicles that failed the test. 
    2. After finishing the above steps for all possible params, 
       it will select the params that gave the smallest pollution quantity
       as best param and plot the implementation outcome for that year
    check out notebooks/what_if_2020.ipynb for usage 
    """
    
    cols = ['VEHICLE_AGE', 'MILE_YEAR', 'MAKE', 
            'MODEL_YEAR', 'ENGINE_WEIGHT_RATIO']
    cat_col = ['MAKE']
    
    def __init__(self):
        """
        """
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.total_tests = None
        self.total_fails = None
        self.year = None
        self.max_depth = None
        self.n_estimators = 1
        self.best_depth = None
        self.pollutions = None
        self.total_predicted_fails = None
        self.anaylsis_table = None
        
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
        for a given year, splits data to train (before that year) and test (in that year)
        '''
        train = self.df[self.df.TEST_SDATE.dt.year < year].sort_values('TEST_SDATE')
        test = self.df[self.df.TEST_SDATE.dt.year == year].sort_values('TEST_SDATE')
        self.y_train = train.pop('RESULT')
        self.X_train = train
        self.y_test = test.pop('RESULT')
        self.X_test = test
        self.total_tests = self.X_test.shape[0]
        self.total_fails = self.y_test.sum()

    def get_estimator(self, depth):  
        '''
        uses Trainer class from trainer.py to get the fitted estimator
        prints the evluation scores
        if you want to plot learning curve, uncomment the last line 
        ''' 
        trainer = Trainer(self.X_train[self.cols],
                          self.y_train,
                          metric='precision',
                          n_estimators = self.n_estimators,
                          with_categorical=self.cat_col,
                          max_depth=depth
         )
        trainer.grid_search()
        print('\nmax_depth:', trainer.search_result.best_params_['model__max_depth'])
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
        df['dayofyear'] = df.index.dayofyear
        df.drop(columns=['count_fail', 'count_test', 'TEST_SDATE'], inplace=True)
        return df
    
    def plot_heuristic_curve(self, df):
        ''' 
        this function is used in plot_simulation_plot():
        it plots the cumulative number of failed vehicles along the year with heuristic decision 
        The heuristic decision is that we predict all the vehicles with age > 16 fail the emissions test
        '''
        # get heuristic prediction
        y_pred = (self.X_test.VEHICLE_AGE > 16).astype('int')
        true_fails = sum([i for i, j in zip (y_pred, self.y_test) if i + j == 2])
        predicted_fails = y_pred.sum()
        self.total_predicted_fails['heuristic'] = predicted_fails
        # create df with prediction outcomes
        pred_df = pd.DataFrame.from_dict({'y_true':self.y_test, 'y_pred':y_pred})
        pred_df = pred_df.sort_values('y_pred', ascending=False)
        # merge the prediction with counter table
        pred_df.index = df.n_tests
        df = df.merge(pred_df, how='left', left_on='n_tests', right_index=True)
        # add new columns storing number of failed test captured along the year
        df['fails_captured_heuristic'] = df.y_true.cumsum()
        df['fails_left_heuristic'] = self.total_fails - df['fails_captured_heuristic']
        plt.plot(df.index, 
                df['fails_captured_heuristic'], 
                label='with heuristic')
        # mark the time when all the predicted fail tests from precision favored model are completed
        t = df[df.n_tests==predicted_fails].index[0]
        cap_fails = df[df.n_tests==predicted_fails]['fails_captured_heuristic'][0]
        true_fails = df[df.n_tests==predicted_fails].n_fails[0]
        print('\n With heuristic decision:')
        print(f'''\nBy the time {str(t)[:10]}, 
                - {round(true_fails)} vehicles were off the road in reality
                - {round(cap_fails)} vehicles could have been off the road with heuristic decision''')
        # what about dayofyear = 100
        t2 = df[df.dayofyear==100].index[0]
        cap_fails2 = df[df.index==t2]['fails_captured_heuristic'][0]
        true_fails2 = df[df.index==t2].n_fails[0]
        print(f'''\nBy the time {str(t2)[:10]}, 
                - {round(true_fails2)} vehicles were off the road in reality
                - {round(cap_fails2)} vehicles could have been off the road with heuristic decision''') 
        # store the pollution quantity in pollution
        self.pollutions['heuristic'] = df['fails_left_heuristic'].sum()
        return df
        
    def plot_simulation_curve(self, year, df):
        '''
        for given year, plots:
        1. cumulative number of failed tests in reality along that year 
        2. cumulative number of failed tests along that year if our solution was implemented
        3. cumulative number of failed tests along that year if heuristic decision was made
        '''
        # find the best max_depth and ploting 
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, 
                df.n_fails, 
                label='with current policy', c='red')
        # create dictionary to store pollution quantity
        self.pollutions = {}
        self.total_predicted_fails = {}
        for depth in self.max_depth:
            trainer = self.get_estimator([depth])
            # get all the numbers
            y_pred = trainer.search_result.predict(self.X_test[self.cols])
            true_fails = sum([i for i, j in zip (y_pred, self.y_test) if i + j == 2])
            predicted_fails = y_pred.sum()
            self.total_predicted_fails[f'{depth}'] = predicted_fails
            # sort testing order by proba
            y_proba = trainer.search_result.predict_proba(self.X_test[self.cols])
            pred_df = pd.DataFrame.from_dict({'y_true':self.y_test, 
                                              'y_pred':y_pred, 
                                              'y_proba':y_proba[:,1]})
            pred_df = pred_df.sort_values('y_proba', ascending=False)
            pred_df.index = df.n_tests
            df = df.merge(pred_df, how='left', left_on='n_tests', right_index=True)
            # add new columns storing number of failed test captured along the year
            df[f'fails_captured_{depth}'] = df.y_true.cumsum()
            df.drop(columns=['y_true', 'y_pred', 'y_proba'], inplace=True)
            df[f'fails_left_{depth}'] = self.total_fails - df[f'fails_captured_{depth}']
            plt.plot(df.index, 
                    df[f'fails_captured_{depth}'], 
                    label=f'with model max_depth = {depth}')
            # mark the time when all the predicted fail tests from precision favored model are completed
            t = df[df.n_tests==predicted_fails].index[0]
            cap_fails = df[df.n_tests==predicted_fails][f'fails_captured_{depth}'][0]
            true_fails = df[df.n_tests==predicted_fails].n_fails[0]
            print(f'''\nBy the time {str(t)[:10]}, 
                    - {round(true_fails)} vehicles were off the road in reality
                    - {round(cap_fails)} vehicles could have been off the road using model max_depth = {depth}''')
            # what about dayofyear = 100
            t2 = df[df.dayofyear==100].index[0]
            cap_fails2 = df[df.index==t2][f'fails_captured_{depth}'][0]
            true_fails2 = df[df.index==t2].n_fails[0]
            print(f'''\nBy the time {str(t2)[:10]}, 
                    - {round(true_fails2)} vehicles were off the road in reality
                    - {round(cap_fails2)} vehicles could have been off the road using model max_depth = {depth}''') 
            # store the pollution quantity in pollution
            self.pollutions[depth] = df[f'fails_left_{depth}'].sum()
        # plot the curve of heuristic decision
        df = self.plot_heuristic_curve(df)
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
        plt.show()
        return df
        
    def implement(self, year, n_estimators=[1], max_depth=[2, 3]):
        '''
        params:
        year: split train (before that year) and test (in that year)
        max_depth and n_estimators: hyper params for grid search with random forest classifier 
        '''
        # train set split
        self.train_test_split(year)
        # get update the class attributes         
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        # get counter table
        df = self.get_counter_table()
        # plotting simulation curve for each max_depth
        df = self.plot_simulation_curve(year, df)
        # pollution quantity correspoding to max depth value   
        tmp = pd.Series(self.pollutions).sort_values()
        print(tmp)
        self.best_depth = tmp.index[0]
        # save df
        self.anaylsis_table = df
        df.to_csv(f'../data/implementation_analysis_{year}_best_{self.best_depth}.csv') 
        print(f'\nSaved implementation_analysis_{year}_best_{self.best_depth}.csv in data folder')
        self.year = year 
        
    def plot_clean(self):
        ''' 
        similar to plot simulation method, 
        only that this one plots the curve creatd by the best max_depth
        '''
        df = self.anaylsis_table
        plt.figure(figsize=(10, 6))
        # reality curve
        plt.plot(df.index, 
                df.n_fails, 
                label='with current policy', c='red')
        # simulated curve
        plt.plot(df.index, 
                     df[f'fails_captured_{self.best_depth}'], 
                     label='with our solution', c='green')
        
        col = f'fails_captured_{self.best_depth}'
        t = df[df.dayofyear==100].index[0]
        # horizontal grey line - simulated curve
        cap_fails = df[df.index==t][col].values[0]
        plt.plot(df[df.index < t].index, 
                 [cap_fails for i in range(df[df.index < t].shape[0])], 
                 c='grey')
        # horizontal grey line - reality curve
        true_fails = df[df.index==t].n_fails.values[0]
        plt.plot(df[df.index < t].index, 
                 [true_fails for i in range(df[df.index < t].shape[0])], 
                 c='grey')
        # vertical grey line
        plt.plot([t for i in range(100)], np.linspace(0, cap_fails, 100), c='grey')
        # fill in the area corresponding to reduced pollution
        plt.fill_between(df.index,
                        df.n_fails,
                        df[col],
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
        plt.title(f'Number of polluting vehicles detected over the year {self.year}')
        
if __name__ == "__main__":
    imp = ImpSearch()
    imp.load_data()
    imp.train_test_split(2020)
    print('Total tests in 2020', imp.total_tests)
    print('Total failed test in 2020', imp.total_fails)