"""
Bias Checking and Fairness Analysis Module
Implements data slicing and fairness metrics
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class BiasChecker:
    """Bias checking and fairness analysis"""
    
    def __init__(self, model, X, y, sensitive_features=None):
        """
        Initialize bias checker
        Args:
            model: Trained model
            X: Features
            y: True labels
            sensitive_features: List of sensitive feature names to analyze
        """
        self.model = model
        self.X = X
        self.y = y
        self.sensitive_features = sensitive_features or []
        self.bias_report = {}
        
    def check_bias(self, sensitive_feature):
        """
        Check bias for a specific sensitive feature
        Args:
            sensitive_feature: Name of sensitive feature to analyze
        Returns:
            Dictionary with bias metrics
        """
        if sensitive_feature not in self.X.columns:
            raise ValueError(f"Sensitive feature '{sensitive_feature}' not found in data")
        
        # Get predictions
        y_pred = self.model.predict(self.X)
        y_pred_proba = self.model.predict_proba(self.X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Group by sensitive feature
        groups = self.X[sensitive_feature].unique()
        group_metrics = {}
        
        for group in groups:
            mask = self.X[sensitive_feature] == group
            y_group = self.y[mask]
            y_pred_group = y_pred[mask]
            
            if len(y_group) == 0:
                continue
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_group, y_pred_group),
                'precision': precision_score(y_group, y_pred_group, zero_division=0),
                'recall': recall_score(y_group, y_pred_group, zero_division=0),
                'f1': f1_score(y_group, y_pred_group, zero_division=0),
                'sample_size': len(y_group),
                'positive_rate': (y_pred_group == 1).mean(),
                'true_positive_rate': recall_score(y_group, y_pred_group, zero_division=0),
                'false_positive_rate': self._calculate_fpr(y_group, y_pred_group)
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_group, y_pred_proba[mask])
            
            group_metrics[group] = metrics
        
        # Calculate fairness metrics
        fairness_metrics = self._calculate_fairness_metrics(group_metrics)
        
        bias_result = {
            'sensitive_feature': sensitive_feature,
            'group_metrics': group_metrics,
            'fairness_metrics': fairness_metrics
        }
        
        self.bias_report[sensitive_feature] = bias_result
        return bias_result
    
    def _calculate_fpr(self, y_true, y_pred):
        """Calculate false positive rate"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0
    
    def _calculate_fairness_metrics(self, group_metrics):
        """Calculate fairness metrics across groups"""
        if len(group_metrics) < 2:
            return {}
        
        groups = list(group_metrics.keys())
        metrics_to_check = ['accuracy', 'precision', 'recall', 'f1', 'positive_rate', 
                          'true_positive_rate', 'false_positive_rate']
        
        fairness_metrics = {}
        
        for metric in metrics_to_check:
            values = [group_metrics[g][metric] for g in groups if metric in group_metrics[g]]
            if len(values) >= 2:
                # Calculate difference (max - min)
                diff = max(values) - min(values)
                # Calculate ratio (min / max)
                ratio = min(values) / max(values) if max(values) > 0 else 0
                
                fairness_metrics[f'{metric}_difference'] = diff
                fairness_metrics[f'{metric}_ratio'] = ratio
                
                # Fairness threshold (80% rule - common fairness metric)
                fairness_metrics[f'{metric}_fair'] = ratio >= 0.8
        
        return fairness_metrics
    
    def check_all_bias(self):
        """Check bias for all sensitive features"""
        if not self.sensitive_features:
            # Auto-detect categorical features as potential sensitive features
            self.sensitive_features = [
                col for col in self.X.columns 
                if self.X[col].dtype in ['object', 'category'] or 
                   self.X[col].nunique() <= 5
            ]
        
        results = {}
        for feature in self.sensitive_features:
            try:
                results[feature] = self.check_bias(feature)
            except Exception as e:
                print(f"Error checking bias for {feature}: {e}")
                continue
        
        return results
    
    def generate_bias_report(self):
        """Generate comprehensive bias report"""
        if not self.bias_report:
            self.check_all_bias()
        
        report = {
            'summary': {},
            'detailed_results': self.bias_report
        }
        
        # Summary statistics
        for feature, result in self.bias_report.items():
            fairness = result['fairness_metrics']
            unfair_metrics = [
                k.replace('_fair', '') 
                for k, v in fairness.items() 
                if k.endswith('_fair') and not v
            ]
            
            report['summary'][feature] = {
                'num_groups': len(result['group_metrics']),
                'unfair_metrics': unfair_metrics,
                'is_fair': len(unfair_metrics) == 0
            }
        
        return report
    
    def visualize_bias(self, sensitive_feature, save_path=None):
        """Visualize bias for a sensitive feature"""
        if sensitive_feature not in self.bias_report:
            self.check_bias(sensitive_feature)
        
        result = self.bias_report[sensitive_feature]
        group_metrics = result['group_metrics']
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        groups = list(group_metrics.keys())
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            values = [group_metrics[g][metric] for g in groups]
            
            ax.bar(groups, values, alpha=0.7)
            ax.set_title(f'{metric.capitalize()} by {sensitive_feature}')
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim(0, 1)
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        return fig

