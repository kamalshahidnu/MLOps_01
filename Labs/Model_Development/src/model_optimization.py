"""
Model Optimization Module
Quantization and Pruning for model compression
"""
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier


class ModelQuantizer:
    """Model quantization for size reduction"""
    
    @staticmethod
    def quantize_model(model, precision='int8'):
        """
        Quantize model weights
        Args:
            model: Model to quantize
            precision: 'int8', 'int16', 'float16'
        Returns:
            Quantized model info
        """
        # This is a simplified version
        # In production, use proper quantization libraries (TensorFlow Lite, ONNX, etc.)
        
        if hasattr(model, 'get_booster'):  # XGBoost
            # XGBoost models can be saved in different formats
            return {
                'method': 'xgboost_native',
                'precision': precision,
                'note': 'XGBoost models are already efficient. Use model.save_model() for binary format.'
            }
        
        return {
            'method': 'generic',
            'precision': precision,
            'note': 'For full quantization, convert to ONNX or TensorFlow Lite format'
        }


class ModelPruner:
    """Model pruning for size reduction"""
    
    def __init__(self, model, pruning_ratio=0.3):
        """
        Initialize model pruner
        Args:
            model: Model to prune
            pruning_ratio: Fraction of parameters to prune (0.3 = 30%)
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
    
    def prune(self):
        """
        Prune model by removing less important features/parameters
        Returns:
            Pruned model
        """
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models (XGBoost, Random Forest)
            importances = self.model.feature_importances_
            threshold = np.percentile(importances, (1 - self.pruning_ratio) * 100)
            
            # Create mask for important features
            important_features = importances >= threshold
            
            return {
                'method': 'feature_importance',
                'pruning_ratio': self.pruning_ratio,
                'features_kept': int(np.sum(important_features)),
                'features_pruned': int(np.sum(~important_features)),
                'note': 'Model structure unchanged. Use feature selection before training for actual pruning.'
            }
        
        elif hasattr(self.model, 'coef_'):
            # Linear models - prune coefficients
            coef = self.model.coef_
            threshold = np.percentile(np.abs(coef), (1 - self.pruning_ratio) * 100)
            coef_pruned = np.where(np.abs(coef) < threshold, 0, coef)
            
            # Create new model with pruned coefficients
            pruned_model = type(self.model)()
            pruned_model.coef_ = coef_pruned
            if hasattr(self.model, 'intercept_'):
                pruned_model.intercept_ = self.model.intercept_
            
            return pruned_model
        
        else:
            return {
                'method': 'not_supported',
                'note': 'Pruning not directly supported for this model type'
            }
    
    def get_pruning_stats(self):
        """Get statistics about pruning"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return {
                'total_features': len(importances),
                'mean_importance': np.mean(importances),
                'std_importance': np.std(importances),
                'min_importance': np.min(importances),
                'max_importance': np.max(importances)
            }
        return None

