
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from emissions.transformer import MakeTransformer
from emissions.utils import scoring_table, plot_learning_curve
from emissions.data import split
import numpy as np

class Trainer():
    def __init__(self, X, y, **kwargs):
        """ 
        params:
        1. X_train
        2. y_train
        3. max_depth, takes a list of possible max_depths for grid search, ex. max_depth=[2, 4, 6]
            by default max_depth=np.arange(2, 10, 1)
        3. metric, used for finding best params in grid search and in ploting learning curve
            by default metric='recall', it also takes 'precision', 'accurarcy'
        4. estimator, model used for training
            by default estimator='tree', it can also take 'forest'
        5. with_categorical, used in set_pipeline
            by default with_categorical='False', meaning X_train doesn't include categorical columns
            if X_train includes categorical columns, please give a list of categorical column names
            ex. with_categorical=['MAKE', 'TEST_TYPE']
        """
        self.estimator = kwargs.get('estimator', 'tree')
        self.X = X
        self.y = y
        self.max_depth = kwargs.get('max_depth', np.arange(2, 10, 1))
        self.metric = kwargs.get('metric', 'recall')
        self.with_categorical = kwargs.get('with_categorical', 'False')
        self.pipeline = None
        self.search_result = None
        self.cat_cols = None
            
    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        if self.estimator == 'tree':
            model = DecisionTreeClassifier(class_weight='balanced')
        elif self.estimator == 'forest':
            model = RandomForestClassifier()
        # transform make
        make_processor = Pipeline([
            ('make_transformer', MakeTransformer()),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        # preprocessor
        if not self.with_categorical: # without categorical cols
            preprocessor = ColumnTransformer([], remainder='passthrough')
        elif self.with_categorical==['MAKE']: # only with MAKE
            preprocessor = ColumnTransformer([
                ('make_processor', make_processor, ['MAKE'])
            ], remainder='passthrough')      
        elif 'MAKE' not in self.with_categorical: # categorical cols w/o MAKE
            preprocessor = ColumnTransformer([
                ('encoder', 
                    OneHotEncoder(handle_unknown='ignore'), 
                    self.with_categorical
                )
            ], remainder='passthrough')    
        else: # categorical cols with MAKE
            preprocessor = ColumnTransformer([
                ('make_processor', make_processor, ['MAKE']),
                ('encoder', 
                    OneHotEncoder(handle_unknown='ignore'), 
                    self.with_categorical
                )
            ], remainder='passthrough')
        # Combine preprocessor and linear model in pipeline
        self.pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('model', model)
        ])
    
    def grid_search(self):
        '''
        performs grid search
        saves grid search result as class attribute self.search_result
        '''
        self.set_pipeline()
        grid = {'model__max_depth': self.max_depth}
        search = GridSearchCV(self.pipeline, 
                      param_grid=grid, 
                      scoring=['accuracy', 'recall', 'precision'],
                      cv=10,
                      refit=self.metric,
                      return_train_score=True,
                      n_jobs=-1
                     ) 
        search.fit(self.X, self.y)
        self.search_result = search
    
    def evaluate(self, X_test, y_test):
        """
        takes X_test and y_test
        returns a scoring table including precision, recall, 
        and accurarcy scores on both train and test sets
        """
        tmp = scoring_table(self.search_result, X_test, y_test)
        return tmp
    
    def learning_curve(self):
        """
        using self.search_result and self.metric plots learning curve
        """
        #print('\nFeatures included:', ' + '.join(self.X.columns.values))
        #print('Metric used:', self.metric)
        plot_learning_curve(self.search_result.best_estimator_, 
                            self.X, 
                            self.y, 
                            scoring=self.metric)
        

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split(df=None, test_size=0.2)
    cols = ['VEHICLE_AGE', 'MILE_YEAR', 'MAKE']
    trainer = Trainer(X_train[cols], y_train)
    trainer.grid_search()
    tmp = trainer.evaluate(X_test[cols], y_test)
    print(tmp)