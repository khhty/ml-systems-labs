import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore", category=FutureWarning)


class Cleaner(BaseEstimator, TransformerMixin):
    """Median (num) / mode (cat) imputation + clipping to train percentiles."""

    def __init__(self, num_cols, cat_cols, low=0.01, high=0.99):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.low = low
        self.high = high

    def _prep(self, X):
        X = X.copy()
        X[self.cat_cols] = X[self.cat_cols].astype(object).replace("?", np.nan)
        X[self.num_cols] = X[self.num_cols].astype(float)
        return X

    def fit(self, X, y=None):
        X = self._prep(X)
        self.medians_ = X[self.num_cols].median()
        self.modes_ = X[self.cat_cols].mode().iloc[0]
        Xn = X[self.num_cols].fillna(self.medians_)
        self.lo_ = Xn.quantile(self.low)
        self.hi_ = Xn.quantile(self.high)
        return self

    def transform(self, X):
        X = self._prep(X)
        X[self.num_cols] = (
            X[self.num_cols].fillna(self.medians_).clip(self.lo_, self.hi_, axis=1)
        )
        X[self.cat_cols] = X[self.cat_cols].fillna(self.modes_)
        return X