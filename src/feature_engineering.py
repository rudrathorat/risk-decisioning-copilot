"""
Advanced feature engineering utilities for credit risk modeling.
Includes WOE transformation, IV calculation, optimal binning, and feature interactions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class WOETransformer:
    """Weight of Evidence (WOE) transformer for categorical and binned features."""
    
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.woe_dict = {}
        self.iv_dict = {}
        
    def fit(self, X: pd.Series, y: pd.Series) -> 'WOETransformer':
        """Calculate WOE for each category/bin."""
        df = pd.DataFrame({'feature': X, 'target': y})
        
        # Calculate distribution
        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()
        
        woe_dict = {}
        iv = 0
        
        for category in df['feature'].unique():
            category_data = df[df['feature'] == category]
            good_count = (category_data['target'] == 0).sum()
            bad_count = (category_data['target'] == 1).sum()
            
            # Avoid division by zero
            good_dist = good_count / (total_good + self.epsilon)
            bad_dist = bad_count / (total_bad + self.epsilon)
            
            # Calculate WOE
            if bad_dist > 0 and good_dist > 0:
                woe = np.log(bad_dist / good_dist)
                iv += (bad_dist - good_dist) * woe
            else:
                woe = 0
                
            woe_dict[category] = woe
        
        self.woe_dict = woe_dict
        self.iv_dict[X.name] = iv
        
        return self
    
    def transform(self, X: pd.Series) -> pd.Series:
        """Transform feature values to WOE."""
        # Handle Categorical types from pd.cut()
        if isinstance(X.dtype, pd.CategoricalDtype):
            X_mapped = X.astype(str).map(self.woe_dict)
        else:
            X_mapped = X.map(self.woe_dict)
        
        # Fill missing values with 0, convert to numeric if needed
        result = pd.to_numeric(X_mapped, errors='coerce').fillna(0)
        return result
    
    def fit_transform(self, X: pd.Series, y: pd.Series) -> pd.Series:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class OptimalBinner:
    """Optimal binning using monotonic WOE."""
    
    def __init__(self, n_bins: int = 5, min_samples: int = 20):
        self.n_bins = n_bins
        self.min_samples = min_samples
        self.bin_edges = None
        self.woe_transformer = WOETransformer()
        
    def fit(self, X: pd.Series, y: pd.Series) -> 'OptimalBinner':
        """Create optimal bins based on WOE monotonicity."""
        # Sort by feature value
        sorted_indices = X.argsort()
        X_sorted = X.iloc[sorted_indices].reset_index(drop=True)
        y_sorted = y.iloc[sorted_indices].reset_index(drop=True)
        
        n_samples = len(X_sorted)
        samples_per_bin = max(self.min_samples, n_samples // self.n_bins)
        
        bin_edges = [X_sorted.min()]
        
        # Create bins with minimum samples
        for i in range(1, self.n_bins):
            idx = i * samples_per_bin
            if idx < n_samples:
                bin_edges.append(X_sorted.iloc[idx])
        
        bin_edges.append(X_sorted.max() + 1e-6)  # Add small buffer
        self.bin_edges = sorted(set(bin_edges))
        
        # Ensure at least 2 unique bin edges
        if len(self.bin_edges) < 2:
            self.bin_edges = [X_sorted.min(), X_sorted.max() + 1e-6]
        
        # Bin the data and fit WOE
        X_binned = pd.cut(X, bins=self.bin_edges, include_lowest=True, duplicates='drop')
        self.woe_transformer.fit(X_binned, y)
        
        return self
    
    def transform(self, X: pd.Series) -> pd.Series:
        """Transform feature to binned WOE values."""
        X_binned = pd.cut(X, bins=self.bin_edges, include_lowest=True, duplicates='drop')
        return self.woe_transformer.transform(X_binned)
    
    def fit_transform(self, X: pd.Series, y: pd.Series) -> pd.Series:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class FeatureEngineer:
    """Main feature engineering pipeline."""
    
    def __init__(self, config=None):
        self.config = config
        self.woe_transformers = {}
        self.binners = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        
    def calculate_iv(self, X: pd.Series, y: pd.Series) -> float:
        """Calculate Information Value for a feature."""
        woe_transformer = WOETransformer()
        woe_transformer.fit(X, y)
        return woe_transformer.iv_dict.get(X.name, 0.0)
    
    def calculate_iv_all(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calculate IV for all features."""
        iv_scores = []
        
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                # Categorical feature
                iv = self.calculate_iv(df[col], y)
            else:
                # Numeric feature - bin first
                binner = OptimalBinner(n_bins=5)
                try:
                    X_binned = binner.fit_transform(df[col], y)
                    iv = self.calculate_iv(X_binned, y)
                except:
                    iv = 0.0
            
            iv_scores.append({
                'feature': col,
                'iv': iv,
                'predictive_power': self._iv_interpretation(iv)
            })
        
        iv_df = pd.DataFrame(iv_scores).sort_values('iv', ascending=False)
        return iv_df
    
    def _iv_interpretation(self, iv: float) -> str:
        """Interpret IV value."""
        if iv < 0.02:
            return "Not useful"
        elif iv < 0.1:
            return "Weak"
        elif iv < 0.3:
            return "Medium"
        elif iv < 0.5:
            return "Strong"
        else:
            return "Very Strong"
    
    def fit_transform_numeric(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform numeric features."""
        X_processed = pd.DataFrame(index=X.index)
        
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                continue
                
            # Use optimal binning + WOE for numeric features
            binner = OptimalBinner(n_bins=5)
            X_woe = binner.fit_transform(X[col], y)
            
            self.binners[col] = binner
            X_processed[f'{col}_woe'] = X_woe
            self.feature_names.append(f'{col}_woe')
        
        return X_processed
    
    def transform_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform numeric features using fitted transformers."""
        X_processed = pd.DataFrame(index=X.index)
        
        for col, binner in self.binners.items():
            if col in X.columns:
                X_woe = binner.transform(X[col])
                X_processed[f'{col}_woe'] = X_woe
        
        return X_processed
    
    def fit_transform_categorical(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform categorical features."""
        X_processed = pd.DataFrame(index=X.index)
        
        for col in X.columns:
            if X[col].dtype not in ['object', 'category']:
                continue
            
            # WOE transformation for categorical
            woe_transformer = WOETransformer()
            X_woe = woe_transformer.fit_transform(X[col], y)
            
            self.woe_transformers[col] = woe_transformer
            X_processed[f'{col}_woe'] = X_woe
            self.feature_names.append(f'{col}_woe')
        
        return X_processed
    
    def transform_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features using fitted transformers."""
        X_processed = pd.DataFrame(index=X.index)
        
        for col, transformer in self.woe_transformers.items():
            if col in X.columns:
                X_woe = transformer.transform(X[col])
                X_processed[f'{col}_woe'] = X_woe
        
        return X_processed
    
    def create_interactions(self, X: pd.DataFrame, feature_pairs: List[Tuple[str, str]] = None,
                            skip_append: bool = False) -> pd.DataFrame:
        """Create interaction features. Use skip_append=True when calling from transform()."""
        if feature_pairs is None:
            # Default interactions for credit risk
            feature_pairs = [
                ('duration', 'credit_amount'),
                ('age', 'credit_amount'),
                ('installment_rate', 'credit_amount')
            ]
        
        X_interactions = pd.DataFrame(index=X.index)
        
        for feat1, feat2 in feature_pairs:
            if feat1 in X.columns and feat2 in X.columns:
                # Ratio interaction
                if X[feat2].abs().min() > 1e-6:
                    interaction_name = f'{feat1}_div_{feat2}'
                    X_interactions[interaction_name] = X[feat1] / (X[feat2] + 1e-6)
                    if not skip_append:
                        self.feature_names.append(interaction_name)
                
                # Product interaction
                interaction_name = f'{feat1}_mul_{feat2}'
                X_interactions[interaction_name] = X[feat1] * X[feat2]
                if not skip_append:
                    self.feature_names.append(interaction_name)
        
        return X_interactions
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, 
                     use_woe: bool = True, use_interactions: bool = True) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        X_processed_list = []
        
        # Numeric features with WOE
        if use_woe:
            X_num = X.select_dtypes(include=[np.number])
            if not X_num.empty:
                X_num_processed = self.fit_transform_numeric(X_num, y)
                X_processed_list.append(X_num_processed)
        
        # Categorical features with WOE
        if use_woe:
            X_cat = X.select_dtypes(include=['object', 'category'])
            if not X_cat.empty:
                X_cat_processed = self.fit_transform_categorical(X_cat, y)
                X_processed_list.append(X_cat_processed)
        
        # Interactions
        if use_interactions:
            X_interactions = self.create_interactions(X)
            if not X_interactions.empty:
                X_processed_list.append(X_interactions)
        
        # Combine all features
        if X_processed_list:
            X_final = pd.concat(X_processed_list, axis=1)
        else:
            X_final = X.copy()
        
        return X_final
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformers."""
        X_processed_list = []
        
        # Transform numeric features
        X_num = X.select_dtypes(include=[np.number])
        if not X_num.empty and self.binners:
            X_num_processed = self.transform_numeric(X_num)
            X_processed_list.append(X_num_processed)
        
        # Transform categorical features
        X_cat = X.select_dtypes(include=['object', 'category'])
        if not X_cat.empty and self.woe_transformers:
            X_cat_processed = self.transform_categorical(X_cat)
            X_processed_list.append(X_cat_processed)
        
        # Create interactions so validation/test match training (even when config is None)
        use_interactions = getattr(self.config, 'use_interactions', True) if self.config else True
        if use_interactions:
            X_interactions = self.create_interactions(X, skip_append=True)
            if not X_interactions.empty:
                X_processed_list.append(X_interactions)
        
        # Combine all features
        if X_processed_list:
            X_final = pd.concat(X_processed_list, axis=1)
            # Only return features that were in training
            X_final = X_final[[col for col in self.feature_names if col in X_final.columns]]
        else:
            X_final = X.copy()
        
        return X_final

