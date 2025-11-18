"""
Feature Selection Module
Implements various feature selection techniques
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel, chi2
)
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


class FeatureSelector:
    """Feature selection using multiple techniques"""
    
    def __init__(self, method='mutual_info', k=None):
        """
        Initialize feature selector
        Args:
            method: 'mutual_info', 'f_classif', 'rfe', 'model_based', 'chi2'
            k: Number of features to select (if None, uses default)
        """
        self.method = method
        self.k = k
        self.selector = None
        self.selected_features = None
        self.feature_scores = None
        
    def fit(self, X, y):
        """Fit the feature selector"""
        if self.k is None:
            # Default to 80% of features
            self.k = max(1, int(len(X.columns) * 0.8))
        
        if self.method == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_classif, k=self.k)
        elif self.method == 'f_classif':
            self.selector = SelectKBest(score_func=f_classif, k=self.k)
        elif self.method == 'chi2':
            self.selector = SelectKBest(score_func=chi2, k=self.k)
        elif self.method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            self.selector = RFE(estimator=estimator, n_features_to_select=self.k)
        elif self.method == 'model_based':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            estimator.fit(X, y)
            self.selector = SelectFromModel(estimator, prefit=True, max_features=self.k)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.selector.fit(X, y)
        self.selected_features = X.columns[self.selector.get_support()].tolist()
        
        # Get feature scores
        if hasattr(self.selector, 'scores_'):
            self.feature_scores = pd.DataFrame({
                'feature': X.columns,
                'score': self.selector.scores_
            }).sort_values('score', ascending=False)
        elif hasattr(self.selector, 'feature_importances_'):
            self.feature_scores = pd.DataFrame({
                'feature': X.columns,
                'score': self.selector.feature_importances_
            }).sort_values('score', ascending=False)
        
        return self
    
    def transform(self, X):
        """Transform data to selected features"""
        if self.selector is None:
            raise ValueError("Selector must be fitted first")
        return pd.DataFrame(
            self.selector.transform(X),
            columns=self.selected_features,
            index=X.index
        )
    
    def fit_transform(self, X, y):
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_selected_features(self):
        """Get list of selected feature names"""
        return self.selected_features
    
    def get_feature_scores(self):
        """Get feature importance scores"""
        return self.feature_scores
    
    def save(self, path='models/feature_selector.pkl'):
        """Save the feature selector"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
    
    @staticmethod
    def load(path='models/feature_selector.pkl'):
        """Load a saved feature selector"""
        return joblib.load(path)

