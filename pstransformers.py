import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore, skew

class SkewnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5, methods=['log', 'sqrt'], variance_threshold=1e-4):
        """
        Initialize the transformer with specified parameters:
        - threshold: Skewness threshold above which to apply transformation.
        - methods: List of transformation methods to consider.
        - variance_threshold: Minimum variance required to consider a feature for skewness correction.
        """
        self.threshold = threshold
        self.methods = methods
        self.variance_threshold = variance_threshold

    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.columns_ = X.columns
        self.transformation_methods_ = {}
        for col in self.columns_:
            col_variance = X[col].var()
            if col_variance < self.variance_threshold:
                # Feature has very low variance; skip transformation
                self.transformation_methods_[col] = 'none'
                continue
            col_skewness = skew(X[col].dropna())
            if abs(col_skewness) > self.threshold:
                # Decide on the best transformation method
                best_method = self._select_best_transformation(X[col])
                self.transformation_methods_[col] = best_method
            else:
                self.transformation_methods_[col] = 'none'
        return self

    def transform(self, X):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_)
        X_transformed = X.copy()
        for col in self.columns_:
            method = self.transformation_methods_.get(col, 'none')
            if method == 'log':
                # Apply log transformation, handle non-positive values
                X_transformed[col] = np.log1p(X_transformed[col].clip(lower=0) + 1e-9)
            elif method == 'sqrt':
                # Apply square root transformation, handle non-positive values
                X_transformed[col] = np.sqrt(X_transformed[col].clip(lower=0))
            # No transformation for 'none'
        return X_transformed.values

    def _select_best_transformation(self, column_data):
        # Evaluate possible transformations and select the best one
        min_skewness = abs(skew(column_data.dropna()))
        best_method = 'none'
        for method in self.methods:
            if method == 'log':
                transformed = np.log1p(column_data.clip(lower=0) + 1e-9)
            elif method == 'sqrt':
                transformed = np.sqrt(column_data.clip(lower=0))
            else:
                continue
            transformed_skewness = abs(skew(transformed.dropna()))
            if transformed_skewness < min_skewness:
                min_skewness = transformed_skewness
                best_method = method
        return best_method
    
class OutlierReplacementTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='none', zscore_threshold=3, iqr_multiplier=1.5):
        self.method = method
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.columns_ = X.columns
        if self.method == 'zscore':
            self.means_ = X.mean()
            self.stds_ = X.std()
        elif self.method == 'iqr':
            self.Q1_ = X.quantile(0.25)
            self.Q3_ = X.quantile(0.75)
            self.IQR_ = self.Q3_ - self.Q1_
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_)
        X_transformed = X.copy()
        if self.method == 'zscore':
            z_scores = (X_transformed - self.means_) / self.stds_
            mask = (np.abs(z_scores) > self.zscore_threshold)
            X_transformed[mask] = np.nan
        elif self.method == 'iqr':
            lower_bound = self.Q1_ - self.iqr_multiplier * self.IQR_
            upper_bound = self.Q3_ + self.iqr_multiplier * self.IQR_
            mask = (X_transformed < lower_bound) | (X_transformed > upper_bound)
            X_transformed[mask] = np.nan
        # No change for 'none'
        return X_transformed.values  # Return as NumPy array for pipeline compatibility

# Define a custom transformer for outlier removal
class OutlierRemovalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='none', zscore_threshold=3, iqr_multiplier=1.5, n_neighbors=20):
        self.method = method
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if X is a DataFrame; if not, convert it to DataFrame for consistent processing
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_transformed = X.copy()
        if self.method == 'zscore':
            # Z-score method
            z_scores = np.abs(zscore(X_transformed, nan_policy='omit'))
            X_transformed = X_transformed[(z_scores < self.zscore_threshold).all(axis=1)]
        elif self.method == 'iqr':
            # IQR method
            Q1 = X_transformed.quantile(0.25)
            Q3 = X_transformed.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            X_transformed = X_transformed[((X_transformed >= lower_bound) & (X_transformed <= upper_bound)).all(axis=1)]
        elif self.method == 'knn':
            # K-Nearest Neighbors method
            lof = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=False)
            is_inlier = lof.fit_predict(X_transformed) == 1
            X_transformed = X_transformed[is_inlier]
        return X_transformed.values  # Return as NumPy array for pipeline compatibility