from sklearn.base import BaseEstimator, TransformerMixin
from emissions.utils import make_transform_get
from emissions.data import load_data, clean_data, split

class MakeTransformer(BaseEstimator, TransformerMixin):
    """
        updates MAKE column of a given dataframe
        1. based on train set decides which makes to keep and which makes will be comebined into other
        2. transform the train or test set based on previous decision
    """

    def __init__(self):
        self.makes_keep = None
        
    def fit(self, X, y=None):
        self.makes_keep = make_transform_get(X)
        return self

    def transform(self, X, y=None):
        df = X.copy()
        other_makes = [make for make in df['MAKE'].unique() if make not in self.makes_keep]
        df['MAKE'] = df['MAKE'].replace(other_makes, 'other')
        return df

if __name__=="__main__":
    X_train, X_test, y_train, y_test = split()
    mt = MakeTransformer().fit(X_train)
    print("\nMAKEs don't belong to other:", mt.makes_keep)
    X_train_update = mt.transform(X_train)
    print('\nNumber of unique makes in train', X_train_update.MAKE.nunique())
    X_test_update = mt.transform(X_test)
    print('\nNumber of unique makes in test', X_test_update.MAKE.nunique())
    