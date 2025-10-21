"""
Machine Learning model for Boston Housing price prediction
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os


class BostonHousingModel:
    """Ridge Regression model for Boston Housing price prediction"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.is_trained = False
        
    def train(self, X_train, y_train):
        """Train the Ridge regression model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def save_model(self, path='models/boston_housing_model.pkl'):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        
    def load_model(self, path='models/boston_housing_model.pkl'):
        """Load a saved model"""
        self.model = joblib.load(path)
        self.is_trained = True
        
    def get_feature_importance(self):
        """Get feature importance (coefficients)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        return self.model.coef_
