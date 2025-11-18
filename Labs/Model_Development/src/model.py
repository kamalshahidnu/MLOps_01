"""
XGBoost model for Heart Disease classification
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os


class HeartDiseaseModel:
    """XGBoost classifier for Heart Disease prediction"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        self.is_trained = False
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the XGBoost model"""
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        self.is_trained = True
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics
    
    def save_model(self, path='models/heart_disease_model.pkl'):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path='models/heart_disease_model.pkl'):
        """Load a saved model"""
        self.model = joblib.load(path)
        self.is_trained = True
        return self.model
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        return self.model.feature_importances_
    
    def get_best_iteration(self):
        """Get best iteration from training (if early stopping was used)"""
        if hasattr(self.model, 'best_iteration'):
            return self.model.best_iteration
        return None

