import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.utils import convert_input


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols , impute_missing=True,
                 handle_unknown='impute', min_samples_leaf=1, smoothing=1):
        
        self.cols = cols
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self._dim = None
        self.mapping = None
        self.impute_missing = impute_missing
        self.handle_unknown = handle_unknown
        self._mean = None
        
    def fit(self, X, y, **kwargs):

        # first check the type
        X = convert_input(X)
        y = pd.Series(y, name='target')
        assert X.shape[0] == y.shape[0]

        _, self.mapping = self.target_encode(
            X, y,
            mapping=self.mapping,
            cols=self.cols,
            impute_missing=self.impute_missing,
            handle_unknown=self.handle_unknown
        )
        
        self._dim = X.shape[1]
        
        return self

    def predict(self, X, y=None):
        
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
        X = convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))
        assert (y is None or X.shape[0] == y.shape[0])

        array , _ = self.target_encode(
            X, y,
            cols=self.cols,
            mapping=self.mapping,
            impute_missing=self.impute_missing,
            handle_unknown=self.handle_unknown, 
            min_samples_leaf=self.min_samples_leaf,
            smoothing_in=self.smoothing
        )

        return array
    
    def target_encode(self, X_in, y, cols , mapping=None, impute_missing=True,
            handle_unknown='impute', min_samples_leaf=1, smoothing_in=1):
        X = X_in.copy(deep=True)
        
        if mapping is not None:
            mapping_out = mapping
            for switch in mapping:
                X[str(switch.get('col')) + '_tmp'] = np.nan
                for val in switch.get('mapping'):
                    if switch.get('mapping')[val]['count'] == 1:
                        X.loc[X[switch.get('col')] == val, str(switch.get('col')) + '_tmp'] = self._mean
                    else:
                        X.loc[X[switch.get('col')] == val, str(switch.get('col')) + '_tmp'] = \
                            switch.get('mapping')[val]['smoothing']
                del X[switch.get('col')]
                X.rename(columns={str(switch.get('col')) + '_tmp': switch.get('col')}, inplace=True)

                if impute_missing:
                    if handle_unknown == 'impute':
                        X[switch.get('col')].fillna(self._mean, inplace=True)
                    elif handle_unknown == 'error':
                        if X[~X['D'].isin([str(x[1]) for x in switch.get('mapping')])].shape[0] > 0:
                            raise ValueError('Unexpected categories found in %s' % (switch.get('col'),))

                X[switch.get('col')] = X[switch.get('col')].astype(float).values.reshape(-1, )
                array = X[switch.get('col')].values
        
        else:
            self._mean = y.mean()
            prior = self._mean
            mapping_out = []
            
            tmp = y.groupby(X[cols]).agg(['sum', 'count'])
            tmp['mean'] = tmp['sum'] / tmp['count']
            tmp = tmp.to_dict(orient='index')

            X[str(cols) + '_tmp'] = np.nan
            for val in tmp:
                tmp[val]['mean'] = tmp[val]['sum']/tmp[val]['count']
                if tmp[val]['count'] == 1:
                    X.loc[X[cols] == val, str(cols) + '_tmp'] = self._mean
                else:
                    smoothing = smoothing_in
                    smoothing = 1 / (1 + np.exp(-(tmp[val]["count"] - min_samples_leaf) / smoothing))
                    cust_smoothing = prior * (1 - smoothing) + tmp[val]['mean'] * smoothing
                    X.loc[X[cols] == val, str(cols) + '_tmp'] = cust_smoothing
                    tmp[val]['smoothing'] = cust_smoothing

            if impute_missing:
                if handle_unknown == 'impute':
                    X[str(cols) + '_tmp'].fillna(self._mean, inplace=True)

            array = X[str(cols) + '_tmp'].astype(float).values.reshape(-1, )
            del X[str(cols) + '_tmp']
            mapping_out.append({'col': cols, 'mapping': tmp})

        return array, mapping_out
