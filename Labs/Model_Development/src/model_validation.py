"""
Model Validation Module
Comprehensive model validation with train/val/test splits
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os
from datetime import datetime


class ModelValidator:
    """Comprehensive model validation"""
    
    def __init__(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Initialize model validator
        Args:
            model: Trained model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.validation_results = {}
        
    def validate(self):
        """Run comprehensive validation"""
        results = {
            'train_metrics': self._evaluate(self.X_train, self.y_train, 'train'),
            'val_metrics': self._evaluate(self.X_val, self.y_val, 'val'),
            'test_metrics': self._evaluate(self.X_test, self.y_test, 'test'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Check for overfitting
        results['overfitting_check'] = self._check_overfitting(
            results['train_metrics'], results['val_metrics']
        )
        
        # Performance comparison
        results['performance_comparison'] = self._compare_performance(
            results['train_metrics'], results['val_metrics'], results['test_metrics']
        )
        
        self.validation_results = results
        return results
    
    def _evaluate(self, X, y, split_name):
        """Evaluate model on a dataset"""
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
        
        return metrics
    
    def _check_overfitting(self, train_metrics, val_metrics):
        """Check for overfitting"""
        threshold = 0.1  # 10% difference threshold
        
        train_acc = train_metrics['accuracy']
        val_acc = val_metrics['accuracy']
        
        diff = train_acc - val_acc
        
        return {
            'is_overfitting': diff > threshold,
            'accuracy_difference': diff,
            'threshold': threshold,
            'severity': 'high' if diff > 0.2 else 'medium' if diff > 0.1 else 'low'
        }
    
    def _compare_performance(self, train_metrics, val_metrics, test_metrics):
        """Compare performance across splits"""
        return {
            'train_val_diff': {
                'accuracy': train_metrics['accuracy'] - val_metrics['accuracy'],
                'f1': train_metrics['f1'] - val_metrics['f1']
            },
            'val_test_diff': {
                'accuracy': val_metrics['accuracy'] - test_metrics['accuracy'],
                'f1': val_metrics['f1'] - test_metrics['f1']
            },
            'is_consistent': abs(val_metrics['accuracy'] - test_metrics['accuracy']) < 0.05
        }
    
    def is_model_valid(self, min_accuracy=0.7, min_f1=0.7, max_overfitting=0.15):
        """Check if model meets validation criteria"""
        if not self.validation_results:
            self.validate()
        
        val_metrics = self.validation_results['val_metrics']
        overfitting = self.validation_results['overfitting_check']
        
        meets_accuracy = val_metrics['accuracy'] >= min_accuracy
        meets_f1 = val_metrics['f1'] >= min_f1
        not_overfitting = overfitting['accuracy_difference'] <= max_overfitting
        
        return {
            'is_valid': meets_accuracy and meets_f1 and not_overfitting,
            'meets_accuracy': meets_accuracy,
            'meets_f1': meets_f1,
            'not_overfitting': not_overfitting,
            'details': {
                'accuracy': val_metrics['accuracy'],
                'f1': val_metrics['f1'],
                'overfitting_diff': overfitting['accuracy_difference']
            }
        }
    
    def save_validation_report(self, path='models/validation_report.json'):
        """Save validation report"""
        import json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(self.validation_results)
        
        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Validation report saved to {path}")

