"""
Data loader for Boston Housing dataset
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


class BostonHousingDataLoader:
    """Data loader for Boston Housing dataset"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and preprocess the Boston Housing dataset"""
        # Load the dataset
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        y = pd.Series(boston.target, name='MEDV')
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def get_feature_names(self):
        """Get feature names"""
        boston = load_boston()
        return boston.feature_names
    
    def save_scaler(self, path='models/scaler.pkl'):
        """Save the scaler for later use"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        
    def load_scaler(self, path='models/scaler.pkl'):
        """Load the saved scaler"""
        return joblib.load(path)
