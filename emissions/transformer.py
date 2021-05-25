from sklearn.base import BaseEstimator, TransformerMixin
from emissions.utils import make_transform_get

class MakeTransformer(BaseEstimator, TransformerMixin):
    """
        Returns a copy of the DataFrame with only one column.
    """

    def __init__(self):
        self.other_makes = None
        
    def fit(self, X, y=None):
        self.other_makes = make_transform_get(X)
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df['MAKE'] = df['MAKE'].replace(self.other_makes, 'other')
        return df