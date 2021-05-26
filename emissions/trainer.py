
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from emissions.transformer import MakeTransformer
from emissions.utils import scoring_table, plot_learning_curve
from emissions.data import load_data, clean_data, split
import numpy as np

class Trainer():
    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.estimator = kwargs.get('estimator', 'tree')
        self.X = X
        self.y = y
        self.pipeline = None
        self.grid = kwargs.get('grid', {'model__max_depth': np.arange(2, 20, 1)})
        self.search_result = None
        self.metric = kwargs.get('metric', 'recall')
            
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
        # Preprocessor
        preprocessor = ColumnTransformer([
            ('make_processor', make_processor, ['MAKE'])
        ], remainder='passthrough')

        # Combine preprocessor and linear model in pipeline
        self.pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('model', model)
        ])
    
    def grid_search(self):
        '''
        performs grid search and saves best params 
        and estimator as class attributes
        '''
        self.set_pipeline()
        search = GridSearchCV(self.pipeline, 
                      param_grid=self.grid, 
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
        takes grid search output and index of best params
        returns a scoring table
        """
        tmp = scoring_table(self.search_result, X_test, y_test)
        return tmp
    
    def learning_curve(self):
        plot_learning_curve(self.search_result.best_estimator_, 
                            self.X, self.y, scoring=self.metric)
        

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split(df=None, test_size=0.2)
    trainer = Trainer(X_train[['VEHICLE_AGE', 'MILE_YEAR', 'MAKE']], y_train)
    trainer.grid_search()
    tmp = trainer.evaluate(X_test[['VEHICLE_AGE', 'MILE_YEAR', 'MAKE']], y_test)
    print(tmp)