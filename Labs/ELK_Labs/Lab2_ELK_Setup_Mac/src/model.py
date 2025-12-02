"""
Random Forest model for Wine Quality classification with ELK logging
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os


class WineQualityModel:
    """Random Forest model for Wine Quality classification"""
    
    def __init__(self, n_estimators=200, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        self.is_trained = False
        
    def train(self, X_train, y_train):
        """Train the Random Forest model"""
        print(f"Training Random Forest with {self.n_estimators} estimators...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training completed!")
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        return metrics
    
    def save_model(self, path='models/wine_quality_model.pkl'):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path='models/wine_quality_model.pkl'):
        """Load a saved model"""
        self.model = joblib.load(path)
        self.is_trained = True
        print(f"Model loaded from {path}")
        
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        return self.model.feature_importances_

